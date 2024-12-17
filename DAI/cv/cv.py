import os

from lightning import Trainer
from ultralytics import YOLO

from ..interfaces import (
    CarlaData,
    CarlaObservation,
    ComputerVisionModule,
    ObjectType,
)
from .object_detection import big_object_detection
from .road_marker_detection import detect_road_markers
from .stabilizer import DetectionStabilizer
from .traffic_light_detection import TrafficLight, detect_traffic_lights
from .traffic_sign_classification import TrafficSign, TrafficSignClassifier


class ComputerVisionModuleImp(ComputerVisionModule):
    """
    Processes the data using the following strategy:
    1. Use big_net to detect every interesting object available
    2. For all detected traffic_signs use the traffic_sign_classifier to classifiy the signs
        a. Use the classified list of traffic signs to extract the current maximum speed
    3. For all detected traffic_light use the traffic_light classifier to detect their relevance and color value
    """

    def __init__(
        self,
    ) -> None:
        # Load in models
        current_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "weights"
        )

        # Load big net
        self.big_net = YOLO(os.path.join(current_dir, "big_net.pt"), task="detect")
        self.big_tracker = DetectionStabilizer()
        # Load traffic light detection
        self.traffic_light_net = YOLO(
            os.path.join(current_dir, "traffic_light.pt"), task="detect"
        )
        self.traffic_light_stabalizer = DetectionStabilizer(
            min_detections=1, persistence_frames=5, stability_threshold=0.1
        )

        self.road_marker_net = YOLO(
            os.path.join(current_dir, "road_markers.pt"), task="detect"
        )

        # Load traffic sign classifier
        self.traffic_sign_classifier = TrafficSignClassifier.from_weights_file(
            os.path.join(current_dir, "traffic_sign.pth")
        )
        self.lightning_trainer = Trainer(
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            accelerator="gpu",
        )

    def process_data(self, data: CarlaData) -> CarlaObservation:
        # Use the big net to detect objects generally
        detected = big_object_detection(self.big_net, data, self.big_tracker)

        # Use the traffic sign classifier to get more details about the traffic signs
        traffic_signs = [
            detected_object
            for detected_object in detected
            if detected_object.type == ObjectType.TRAFFIC_SIGN
        ]
        traffic_signs = TrafficSignClassifier.classify(
            self.lightning_trainer,
            self.traffic_sign_classifier,
            traffic_signs,
            data.rgb_image,
        )
        max_speed = TrafficSign.speed_limit(traffic_signs)

        lights, traffic_light_objects = detect_traffic_lights(
            self.traffic_light_net, self.traffic_light_stabalizer, data
        )
        current_light = TrafficLight.should_stop(lights)
        detected = [obj for obj in detected if obj.type != ObjectType.TRAFFIC_LIGHT]
        detected.extend(traffic_light_objects)
        road_markers = detect_road_markers(self.road_marker_net, data=data)
        detected.extend(road_markers)
        # stop_lines = [
        #     road_marker
        #     for road_marker in road_markers
        #     if road_marker.type == ObjectType.STOP_LINE
        #     and road_marker.confidence >= 0.5
        # ]
        crossings = [
            road_marker
            for road_marker in road_markers
            if road_marker.type == ObjectType.CROSSING and road_marker.confidence >= 0.5
        ]

        return CarlaObservation(
            objects=detected,
            current_speed=data.current_speed,
            max_speed=max_speed,
            red_light=current_light,
            distance_to_pedestrian_crossing=crossings[0].distance
            if len(crossings) > 0
            else None,
            distance_to_stop=min(
                traffic_light_objects, key=lambda obj: obj.distance
            ).distance
            if len(traffic_light_objects) > 0
            else None,
            pedestrian_crossing_flag=len(crossings) > 0,
            angle=data.angle,
        )
