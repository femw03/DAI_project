from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..interfaces import BoundingBox, ObjectType


@dataclass
class Detection:
    bounding_box: BoundingBox
    counter: int
    currently_seen: bool
    confidence: float


class DetectionStabilizer:
    def __init__(
        self,
        stability_threshold=0.4,  # IoU threshold for similar detections
        persistence_frames=3,  # Frames to maintain detection
        confidence_decay=0.95,  # Confidence decay rate
        max_tracked_instances=10,  # Max number of tracked instances per class
        min_detections=2,
    ):
        # Store multiple tracked instances per class
        self.tracked_instances: Dict[ObjectType, List[Detection]] = {}
        self.candidates: Dict[ObjectType, List[Detection]] = {}

        # Configuration parameters
        self.stability_threshold = stability_threshold
        self.persistence_frames = persistence_frames
        self.confidence_decay = confidence_decay
        self.max_tracked_instances = max_tracked_instances
        self.min_detections = min_detections

    def _calculate_iou(self, box1: BoundingBox, box2: BoundingBox):
        """Calculate Intersection over Union between two bounding boxes"""
        x1, y1, x2, y2 = box1.xyxy
        x3, y3, x4, y4 = box2.xyxy

        # Calculate intersection coordinates
        x_left = max(x1, x3)
        y_top = max(y1, y3)
        x_right = min(x2, x4)
        y_bottom = min(y2, y4)

        # Calculate intersection area
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area

    def stabilize_detections(
        self, current_detections: List[Tuple[ObjectType, float, BoundingBox]]
    ) -> List[Tuple[ObjectType, float, BoundingBox]]:
        """
        Stabilize detections by reducing flickering, supporting multiple instances

        :param current_detections: List of detections in current frame
        :return: Filtered and stabilized detections
        """

        for class_label, confidence, bounding_box in current_detections:
            self.process_detection(class_label, confidence, bounding_box)

        self.transfer()
        self.update_counters()
        self.clean_up()
        detections = []
        for class_label, values in self.tracked_instances.items():
            for detection in values:
                # Decay confidence
                detections.append(
                    (class_label, detection.confidence, detection.bounding_box)
                )
        return detections

    def find_overlapping(
        self, bounding_box: BoundingBox, reference: List[Detection]
    ) -> Optional[Tuple[float, int, Detection]]:
        matches = [
            self._calculate_iou(bounding_box, tracked.bounding_box)
            for tracked in reference
        ]
        if len(matches) == 0:
            return None
        best_iou = max(matches, default=0.0)
        best_match_idx = matches.index(best_iou)
        best_match = reference[best_match_idx]
        return best_iou, best_match_idx, best_match

    def process_detection(
        self, class_label, confidence, bounding_box
    ) -> Optional[List[Tuple[ObjectType, float, BoundingBox]]]:
        # Initialize tracking for this class if not exists
        if class_label not in self.tracked_instances:
            self.tracked_instances[class_label] = []
        if class_label not in self.candidates:
            self.candidates[class_label] = []

        # Find best matching existing tracked instance
        result = self.find_overlapping(
            bounding_box, self.tracked_instances[class_label]
        )

        if (
            result is not None
            and result[0] >= self.stability_threshold
            and not result[2].currently_seen
        ):
            best_iou, best_match_idx, best_match = result
            # Existing instance match found

            # Reset the counter to max_persistence frames
            best_match.counter = self.persistence_frames
            best_match.bounding_box = bounding_box  # Update bbox
            best_match.currently_seen = True
            best_match.confidence = confidence

            # Update the instance in tracked instances
            self.tracked_instances[class_label][best_match_idx] = best_match
            # Add to stabilized detections if meets criteria
            return None

        # New instance for this class
        result = self.find_overlapping(bounding_box, self.candidates[class_label])
        # Observation was also not a candidate
        if result is None or result[0] < self.stability_threshold:
            self.candidates[class_label].append(
                Detection(
                    bounding_box, counter=1, currently_seen=True, confidence=confidence
                )
            )
            return None
        best_iou, best_match_idx, best_match = result

        # Observation was already a candidate
        best_match.counter += 1
        best_match.currently_seen = True
        best_match.bounding_box = bounding_box
        best_match.confidence = confidence
        # Add new instance if we haven't exceeded max tracked
        return None

    def update_counters(self) -> None:
        detections: List[Detection] = []
        for values in self.tracked_instances.values():
            for detection in values:
                detections.append(detection)

        for values in self.candidates.values():
            for detection in values:
                detections.append(detection)

        for detection in detections:
            if not detection.currently_seen:
                detection.counter -= 1
            detection.currently_seen = False

    def clean_up(self) -> None:
        for class_label, detections in self.tracked_instances.items():
            self.tracked_instances[class_label] = [
                detection for detection in detections if detection.counter > 0
            ]
        for class_label, detections in self.candidates.items():
            self.candidates[class_label] = [
                detection for detection in detections if detection.counter > 0
            ]

    def transfer(self) -> None:
        for class_label, detections in self.candidates.items():
            for detection in detections:
                if detection.counter > self.min_detections:
                    detections.remove(detection)
                    detection.counter = self.persistence_frames
                    self.tracked_instances[class_label].append(detection)
