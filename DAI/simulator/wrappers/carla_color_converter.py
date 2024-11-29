from __future__ import annotations

import carla


class CarlaColorConverter:
    def __init__(self, converter: carla.ColorConverter) -> None:
        assert isinstance(converter, carla.ColorConverter)
        self.converter = converter

    @staticmethod
    def DEPTH() -> CarlaColorConverter:
        return CarlaColorConverter(carla.ColorConverter.Depth)
