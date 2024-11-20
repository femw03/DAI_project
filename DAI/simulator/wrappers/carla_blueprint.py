from __future__ import annotations

from typing import List

import carla
from loguru import logger


class CarlaBlueprintLibrary:
    def __init__(self, blueprint_library=carla.BlueprintLibrary) -> None:
        assert isinstance(
            blueprint_library, carla.BlueprintLibrary
        ), f"Received {type(blueprint_library)} instead of carla.BlueprintLibrary"
        self.blueprint_library = blueprint_library

    def filter(self, search_string) -> List[CarlaBlueprint]:
        blueprints = [
            CarlaBlueprint(blueprint)
            for blueprint in self.blueprint_library.filter(search_string)
        ]
        logger.debug(
            f"The search_string '{search_string}' returned {len(blueprints)} blueprints: {blueprints}"
        )
        return blueprints


class CarlaBlueprint:
    def __init__(self, blueprint: carla.ActorBlueprint) -> None:
        assert isinstance(
            blueprint, carla.ActorBlueprint
        ), f"Received {type(blueprint)} but expected carla.ActorBlueprint"
        self.blueprint = blueprint

    def __getitem__(self, id: str) -> carla.ActorAttribute:
        return self.blueprint.get_attribute(id)

    def __setitem__(self, id: str, value: carla.ActorAttribute) -> None:
        self.blueprint.set_attribute(id, value)

    def contains(self, id: str) -> bool:
        return self.blueprint.has_attribute(id)

    @property
    def tags(self) -> List[str]:
        return self.blueprint.tags

    def __repr__(self) -> str:
        return self.blueprint.id


class CarlaRGBBlueprint(CarlaBlueprint):
    """A carla RGB blueprint check attributes to see what attributes are available"""

    ATTRIBUTES = {
        "black_clip",
        "blade_count",
        "bloom_intensity",
        "blur_amount",
        "blur_radius",
        "calibration_constant",
        "chromatic_aberration_intensity",
        "chromatic_aberration_offset",
        "enable_postprocess_effects",
        "exposure_compensation",
        "exposure_max_bright",
        "exposure_min_bright",
        "exposure_mode",
        "exposure_speed_down",
        "exposure_speed_up",
        "focal_distance",
        "fov",
        "fstop",
        "gamma",
        "image_size_x",
        "image_size_y",
        "iso",
        "lens_circle_falloff",
        "lens_circle_multiplier",
        "lens_flare_intensity",
        "lens_k",
        "lens_kcube",
        "lens_x_size",
        "lens_y_size",
        "min_fstop",
        "motion_blur_intensity",
        "motion_blur_max_distortion",
        "motion_blur_min_object_screen_size",
        "role_name",
        "sensor_tick",
        "shoulder",
        "shutter_speed",
        "slope",
        "temp",
        "tint",
        "toe",
        "white_clip",
    }

    def __init__(self, blueprint: carla.ActorBlueprint) -> None:
        super().__init__(blueprint)

    @staticmethod
    def from_blueprint(blueprint: CarlaBlueprint) -> CarlaRGBBlueprint:
        has_attribute = [
            blueprint.contains(attribute) for attribute in CarlaRGBBlueprint.ATTRIBUTES
        ]
        if not all(has_attribute):
            missing_keys = [
                attribute
                for attribute in CarlaRGBBlueprint.ATTRIBUTES
                if not blueprint.contains(attribute)
            ]
            raise Exception(
                f"Cannot downcast the blueprint to RGB blueprint because it does not have the following keys: {missing_keys}"
            )
        return CarlaRGBBlueprint(blueprint.blueprint)


class CarlaDepthBlueprint(CarlaBlueprint):
    ATTRIBUTES = {
        "fov",
        "image_size_x",
        "image_size_y",
        "lens_circle_falloff",
        "lens_circle_multiplier",
        "lens_k",
        "lens_kcube",
        "lens_x_size",
        "lens_y_size",
        "role_name",
        "sensor_tick",
    }

    def __init__(self, blueprint: carla.ActorBlueprint) -> None:
        super().__init__(blueprint)

    @staticmethod
    def from_blueprint(blueprint: CarlaBlueprint) -> CarlaDepthBlueprint:
        has_attribute = [
            blueprint.contains(attribute)
            for attribute in CarlaDepthBlueprint.ATTRIBUTES
        ]
        if not all(has_attribute):
            missing_keys = [
                attribute
                for attribute in CarlaDepthBlueprint.ATTRIBUTES
                if not blueprint.contains(attribute)
            ]
            raise Exception(
                f"Cannot downcast the blueprint to RGB blueprint because it does not have the following keys: {missing_keys}"
            )
        return CarlaDepthBlueprint(blueprint.blueprint)
