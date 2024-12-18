from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Generic, List, Optional, Tuple, TypeVar

import carla
import numpy as np
from loguru import logger

from .carla_blueprint import CarlaBlueprint, CarlaBlueprintLibrary
from .carla_color_converter import CarlaColorConverter
from .carla_utils import CarlaLocation, CarlaVector3D, CarlaWaypoint


class CarlaWorld:
    """A wrapper class around the carla world object to allow Typing"""

    def __init__(self, world: carla.World) -> None:
        assert isinstance(
            world, carla.World
        ), f"Instantiate a CarlaWorld object with an actual carla world object, received {type(world)}"
        self.world = world
        self._map: Optional[CarlaMap] = None
        self._pededstrian_crossing_factor: Optional[float] = None

    def tick(self, timeout=10) -> int:
        """Signal the server to compute the next simulation tick, returns the ID of the computed frame"""
        logger.debug("Signaling world tick")
        tick_id = self.world.tick(timeout)
        logger.debug(f"Received world tick {tick_id}")
        return tick_id

    def spawn_actor(
        self,
        blueprint: CarlaBlueprint,
        location: carla.Transform,
        parent: Optional[CarlaActor] = None,
    ) -> CarlaActor:
        try:
            actor = self.world.spawn_actor(
                blueprint=blueprint.blueprint,
                transform=location,
                attach_to=parent.actor,
            )
        except Exception as e:
            logger.warning(
                f"The following exception occurred when spawning {blueprint}: {e}"
            )
            raise e
        return CarlaActor(actor, self)

    def spawn_vehicle(
        self, blueprint: CarlaBlueprint, location: carla.Transform
    ) -> CarlaVehicle:
        try:
            vehicle = self.world.spawn_actor(blueprint.blueprint, location)
        except Exception as e:
            logger.warning(
                f"The following exception occurred when spawning {blueprint}: {e}"
            )
            raise e
        return CarlaVehicle(vehicle, self)

    def spawn_walker(
        self, blueprint: CarlaBlueprint, location: carla.Transform
    ) -> CarlaWalker:
        try:
            if self._pededstrian_crossing_factor is None:
                logger.warning(
                    "The pedestrian crossing factor was not yet set while attempting to spawn a pedestrian"
                )
            vehicle = self.world.spawn_actor(blueprint.blueprint, location)
        except Exception as e:
            logger.warning(
                f"The following exception occurred when spawning {blueprint}: {e}"
            )
            raise e
        return CarlaWalker(vehicle, self)

    def get_random_walker_location(self) -> CarlaLocation:
        return CarlaLocation.from_native(
            self.world.get_random_location_from_navigation()
        )

    @property
    def blueprint_library(self) -> CarlaBlueprintLibrary:
        return CarlaBlueprintLibrary(self.world.get_blueprint_library())

    @property
    def map(self) -> CarlaMap:
        if self._map is None:
            logger.info("Fetching map from server")
            self._map = CarlaMap(self.world.get_map())
        return self._map

    @property
    def pedestrian_crossing_factor(self) -> float:
        return self._pededstrian_crossing_factor

    @pedestrian_crossing_factor.setter
    def pedestrian_crossing_factor(self, value: float) -> None:
        self.world.set_pedestrians_cross_factor(value)
        self._pededstrian_crossing_factor = value

    @property
    def settings(self) -> carla.WorldSettings:
        return self.world.get_settings()

    @settings.setter
    def settings(self, value: carla.WorldSettings) -> None:
        assert isinstance(value, carla.WorldSettings)
        self.world.apply_settings(value)

    @property
    def synchronous_mode(self) -> bool:
        return self.settings.synchronous_mode

    @property
    def delta_seconds(self) -> Optional[float]:
        """The amount of delt seconds that pass between ticks, None means it is variable"""
        delta: float = self.settings.fixed_delta_seconds
        return None if delta == 0.0 else delta

    @delta_seconds.setter
    def delta_seconds(self, value: Optional[float]):
        logger.info(f"Setting world delta seconds to {value}")
        settings = self.settings
        settings.fixed_delta_seconds = value if value is not None else 0.0
        self.settings = settings

    @synchronous_mode.setter
    def synchronous_mode(self, value: bool) -> None:
        logger.info(f"Setting synchronous mode to: {value}")
        settings = self.settings
        settings.synchronous_mode = value
        self.settings = settings

    def get_actors(self, filter: str) -> List[CarlaActor]:
        actors = self.world.get_actors().filter(filter)
        return [CarlaActor(actor,self) for actor in actors]


class CarlaMap:
    """Wrapper class around carla.Map"""

    def __init__(self, map: carla.Map) -> None:
        assert isinstance(map, carla.Map), f"Expected carla.Map but got {type(map)}"
        self.map = map

    @property
    def spawn_points(self) -> List[carla.Transform]:
        return self.map.get_spawn_points()

    def get_waypoint(self, location: CarlaLocation) -> CarlaWaypoint:
        """Gets a waypoint at location which will continue on to the nearest road"""
        return CarlaWaypoint(self.map.get_waypoint(location.native))

    def get_topology(self) -> List[Tuple[CarlaWaypoint, CarlaWaypoint]]:
        return [
            (CarlaWaypoint(begin), CarlaWaypoint(end))
            for begin, end in self.map.get_topology()
        ]


class CarlaActor:
    """Wrapper class around a carla actor"""

    def __init__(self, actor: carla.Actor, world: CarlaWorld) -> None:
        assert isinstance(
            actor, carla.Actor
        ), f"Instantiate a CarlaActor with a carla.Actor object instead of {type(actor)}"
        self.actor = actor
        self.world = world
        self.sensors: List[CarlaSensor] = []

    def __repr__(self) -> str:
        return str(self.actor)

    def destroy(self) -> bool:
        try:
            return self.actor.destroy()
        except Exception as e:
            logger.warning(f"Error when dispawning {self.actor}: {e}")
            return False

    def add_colision_detector(self) -> CarlaCollisionSensor:
        blueprint = self.world.blueprint_library.filter("sensor.other.collision")[0]
        actor = self.world.spawn_actor(blueprint, carla.Transform(), parent=self)
        sensor = CarlaCollisionSensor(actor.actor, self.world, self)
        self.sensors.append(sensor)
        return sensor

    @property
    def velocity(self) -> CarlaVector3D:
        return CarlaVector3D(self.actor.get_velocity())

    @velocity.setter
    def velocity(self, new_velocity: CarlaVector3D) -> None:
        self.actor.set_target_velocity(new_velocity.vector)

    @property
    def location(self) -> CarlaLocation:
        return CarlaLocation.from_native(self.actor.get_location())

    @location.setter
    def location(self, new_location: CarlaLocation):
        self.actor.set_location(new_location.native)

    @property
    def transform(self) -> carla.Transform:
        return self.actor.get_transform()

    @transform.setter
    def transform(self, new_transform: carla.Transform) -> None:
        self.actor.set_transform(new_transform)

    @property
    def state(self):
        return self.actor.actor_state

    def disable_physics(self, shoul_disable: bool = True) -> None:
        self.actor.set_simulate_physics(not shoul_disable)


class CarlaWalkerAI(CarlaActor):
    """An actor that controls a walker"""

    def __init__(
        self, controller: carla.Actor, world: CarlaWorld, walker: CarlaWalker
    ) -> None:
        super().__init__(controller, world)
        assert isinstance(
            controller, carla.WalkerAIController
        ), f"Instantiate CarlaWalker with a walker instead of {type(controller)}"
        self.walker = walker
        self._speed = 0.0

    def stop(self) -> None:
        self.actor.stop()

    def start(self) -> None:
        self.actor.start()

    def go_to_location(self, location: CarlaLocation) -> None:
        self.actor.go_to_location(location.native)

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        self._speed = value
        self.actor.set_max_speed(value)


class CarlaWalker(CarlaActor):
    """"""

    def __init__(self, walker: carla.Walker, world: CarlaWorld) -> None:
        super().__init__(walker, world)
        assert isinstance(
            walker, carla.Walker
        ), f"Instantiate CarlaWalker with a walker instead of {type(walker)}"
        self.controller: Optional[CarlaWalkerAI] = None

    def add_controller(self, blueprint: CarlaBlueprint) -> CarlaWalkerAI:
        controller_actor = self.world.spawn_actor(
            blueprint, carla.Transform(), parent=self
        )
        self.controller = CarlaWalkerAI(controller_actor.actor, self.world, self)
        return self.controller

    def destroy(self) -> bool:
        if self.controller is not None:
            self.controller.stop()
            self.controller.destroy()

        return super().destroy()


class CarlaVehicle(CarlaActor):
    """"""

    def __init__(self, vehicle: carla.Vehicle, world: CarlaWorld) -> None:
        super().__init__(vehicle, world)
        assert isinstance(
            vehicle, carla.Vehicle
        ), f"Instantiate CarlaVehicle with a vehicle instead of {type(vehicle)}"
        self._autopilot = False

    @property
    def autopilot(self) -> bool:
        return self._autopilot

    @autopilot.setter
    def autopilot(self, value: bool) -> None:
        self.actor.set_autopilot(value)
        self._autopilot = value

    def add_camera(
        self,
        sensor_blueprint: CarlaBlueprint,
        location: carla.Location = carla.Location(2, 0, 1),
        rotation: carla.Rotation = carla.Rotation(0, 0, 0),
    ) -> CarlaCamera:
        actor = self.world.spawn_actor(
            sensor_blueprint, carla.Transform(location, rotation), parent=self
        )
        sensor = CarlaCamera(actor.actor, self.world, self)
        self.sensors.append(sensor)
        return sensor

    def destroy(self) -> bool:
        for sensor in self.sensors:
            sensor.stop()
            sensor.destroy()
        return super().destroy()

    @property
    def control(self) -> CarlaVehicleControl:
        return CarlaVehicleControl(self.actor.get_control())

    @control.setter
    def control(self, new_control: CarlaVehicleControl) -> None:
        self.actor.apply_control(new_control.control)

    @property
    def current_max_speed(self) -> float:
        return self.actor.get_speed_limit()

    @property
    def get_traffic_light_state(self) -> CarlaTrafficLightState:
        return CarlaTrafficLightState.from_native(self.actor.get_traffic_light_state())

    @property
    def current_traffic_light(self) -> Optional[CarlaTrafficLight]:
        traffic_light = self.actor.get_traffic_light()
        return CarlaTrafficLight(traffic_light) if traffic_light is not None else None


class CarlaTrafficLight(CarlaActor):
    "A wrapper for: https://carla.readthedocs.io/en/latest/python_api/#carla.TrafficLight"

    def __init__(self, traffic_light: carla.TrafficLight, world: CarlaWorld):
        super().__init__(traffic_light, world)
        assert isinstance(traffic_light, carla.TrafficLight)
        self.traffic_light = traffic_light

    @property
    def state(self) -> CarlaTrafficLightState:
        return CarlaTrafficLightState.from_native(self.traffic_light.state)

    @property
    def stop_points(self) -> List[CarlaWaypoint]:
        return [CarlaWaypoint(wp) for wp in self.traffic_light.get_stop_waypoints()]

    @property
    def pole_index(self) -> int:
        """Returns the index of the pole that identifies it as part of the traffic light group of a junction."""
        return self.traffic_light.get_pole_index()

    @staticmethod
    def all(world: CarlaWorld) -> List[CarlaTrafficLight]:
        return [CarlaTrafficLight(actor.actor, world) for actor in world.get_actors('traffic.traffic_light')]
    
    @property
    def affected_waypoints(self) -> List[CarlaWaypoint]:
        waypoints = self.actor.get_affected_lane_waypoints()
        return [CarlaWaypoint(waypoint) for waypoint in waypoints]

class CarlaVehicleControl:
    """A wrapper for: https://carla.readthedocs.io/en/latest/python_api/#carla.VehicleControl"""

    def __init__(self, control: carla.VehicleControl) -> None:
        assert isinstance(control, carla.VehicleControl)
        self.control = control

    def clone(self) -> CarlaVehicleControl:
        new_control = CarlaVehicleControl.new()
        new_control.throttle = self.throttle
        new_control.steer = self.steer
        new_control.brake = self.brake
        new_control.hand_brake = self.hand_brake
        new_control.reverse = self.reverse
        new_control.manual_gear_shift = self.manual_gear_shift
        new_control.gear = self.gear
        return new_control

    @staticmethod
    def new() -> CarlaVehicleControl:
        return CarlaVehicleControl(carla.VehicleControl())

    @property
    def throttle(self) -> float:
        return self.control.throttle

    @throttle.setter
    def throttle(self, value: float) -> None:
        self.control.throttle = value

    @property
    def steer(self) -> float:
        return self.control.steer

    @steer.setter
    def steer(self, value: float) -> None:
        self.control.steer = value

    @property
    def brake(self) -> float:
        return self.control.brake

    @brake.setter
    def brake(self, value: float) -> None:
        self.control.brake = value

    @property
    def hand_brake(self) -> bool:
        return self.control.hand_brake

    @hand_brake.setter
    def hand_brake(self, value: bool) -> None:
        self.control.hand_brake = value

    @property
    def reverse(self) -> bool:
        return self.control.reverse

    @reverse.setter
    def reverse(self, value: bool) -> None:
        self.control.reverse = value

    @property
    def manual_gear_shift(self) -> bool:
        return self.control.manual_gear_shift

    @manual_gear_shift.setter
    def manual_gear_shift(self, value: bool) -> None:
        self.control.manual_gear_shift = value

    @property
    def gear(self) -> int:
        return self.control.gear

    @gear.setter
    def gear(self, value: int) -> None:
        self.control.gear = value

    def __str__(self) -> str:
        return f"throttle: {self.throttle}, steer: {self.steer}, brake: {self.brake}, hand_brake: {self.hand_brake}, reverse: {self.reverse}, manual: {self.manual_gear_shift}, gear: {self.gear},"

    def __repr__(self) -> str:
        return str(self)


T = TypeVar("T")


class CarlaSensor(Generic[T], CarlaActor):
    """A sensor often attached to a vehicle"""

    def __init__(
        self, sensor: carla.Sensor, world: CarlaWorld, vehicle: CarlaVehicle
    ) -> None:
        CarlaActor.__init__(self, sensor, world)
        assert isinstance(
            sensor, carla.Sensor
        ), f"Initiate CarlaSensor with carla.Sensor and not {type(sensor)}"
        self.vehicle = vehicle

    def stop(self) -> None:
        self.actor.stop()

    def listen(self, callback: Callable[[T], None]) -> None:
        self.actor.listen(callback)
        assert self.actor.is_listening


class CarlaCamera(CarlaSensor["CarlaImage"]):
    """A special type of sensor"""

    def __init__(
        self, sensor: carla.Sensor, world: CarlaWorld, vehicle: CarlaVehicle
    ) -> None:
        CarlaSensor.__init__(self, sensor, world, vehicle)

    def listen(self, callback: Callable[[CarlaImage], None]) -> None:
        def convert_and_listen(image: carla.Image) -> None:
            logger.debug("received image")
            converted = CarlaImage.from_native(image)
            callback(converted)

        super().listen(convert_and_listen)


class CarlaCollisionSensor(CarlaSensor["CarlaCollisionEvent"]):
    def __init__(
        self, sensor: carla.Sensor, world: CarlaWorld, vehicle: CarlaVehicle
    ) -> None:
        super().__init__(sensor, world, vehicle)


@dataclass
class CarlaImage:
    fov: float
    height: int
    width: int
    raw_data: List[int]
    """Flattened array of bytes, use reshape according to width and height"""
    native: carla.Image

    @staticmethod
    def from_native(carla_image: carla.Image):
        assert isinstance(carla_image, carla.Image)
        return CarlaImage(
            carla_image.fov,
            carla_image.height,
            carla_image.width,
            carla_image.raw_data,
            carla_image,
        )

    @property
    def numpy_image(self) -> np.ndarray:
        array = np.frombuffer(self.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (self.height, self.width, 4))  # Reshape into BGRA
        array = array[..., :3]  # Select the color channels
        array = array[:, :, ::-1]  # reverse to RGB
        return array

    def to_depth(self) -> np.ndarray:
        return self.native.convert(carla.ColorConverter.Depth)

    def convert(self, converter: CarlaColorConverter) -> CarlaImage:
        self.native.convert(converter.converter)

        return self


@dataclass
class CarlaCollisionEvent:
    actor: CarlaActor
    other_actor: CarlaActor

    @staticmethod
    def from_native(native: carla.CollisionEvent) -> CarlaCollisionEvent:
        return CarlaCollisionEvent(
            actor=CarlaActor(native.actor), other_actor=CarlaActor(native.other_actor)
        )


class CarlaTrafficLightState(Enum):
    RED = carla.TrafficLightState.Red
    YELLOW = carla.TrafficLightState.Yellow
    GREEN = carla.TrafficLightState.Green
    OFF = carla.TrafficLightState.Off
    UNKNOWN = carla.TrafficLightState.Unknown

    @staticmethod
    def from_native(native: carla.TrafficLightState):
        return next(
            (state for state in CarlaTrafficLightState if state.value == native), None
        )
