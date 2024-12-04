import os

from lightning import Trainer
from ultralytics import YOLO

from ..interfaces import (
    CarlaData,
    CarlaFeatures,
    ComputerVisionModule,
    ObjectType,
)
from .object_detection import big_object_detection
from .traffic_light_detection import TrafficLight, detect_traffic_lights
from .traffic_sign_classification import TrafficSign, TrafficSignClassifier


class ComputerVisionModuleImp(ComputerVisionModule):
    """
    Processes the data using the following strategy:
    1. Use big_net to detect every interesting object available
    2. For all detected traffic_signs use the traffic_sign_classifier to classifiy the signs
        a. Use the classified list of traffic signs to extract the current maximum speed
    3. For all detected traffic_light use the traffic_light classifier to detect their relevance and color value
        # TODO (implement)
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

        # Load traffic light detection
        self.traffic_light_net = YOLO(
            os.path.join(current_dir, "traffic_light.pt"), task="detect"
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

    def process_data(self, data: CarlaData) -> CarlaFeatures:
        # Use the big net to detect objects generally
        detected = big_object_detection(self.big_net, data)

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

        traffic_lights = [
            detected_object
            for detected_object in detected
            if detected_object.type == ObjectType.TRAFFIC_LIGHT
        ]
        current_light = None
        if len(traffic_lights) != 0:
            lights = detect_traffic_lights(self.traffic_light_net, data.rgb_image)
            current_light = TrafficLight.should_stop(lights)

        return CarlaFeatures(
            objects=detected,
            current_speed=data.current_speed,
            max_speed=max_speed,
            stop_flag=current_light,
            distance_to_pedestrian_crossing=0,  # TODO
            distance_to_stop=0,  # TODO
            pedestrian_crossing_flag=False,  # TODO
        )
