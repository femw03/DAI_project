"""Contains an abstract definition of the World object"""

from __future__ import annotations

import time
from abc import ABC
from threading import Event
from typing import Callable, List, Optional, final  # noqa: F401

from loguru import logger

from .data_carriers import CarlaData


class World(ABC):
    """
    The bridge between the world and the python environment.
    The latest frame data will be present in self.data or can be listened to in by adding a callback
    via self.add_listeners.
    """

    def __init__(self) -> None:
        super().__init__()
        self.__data_listeners: List[Callable[[CarlaData], None]] = []
        self.__tick_listeners: List[Callable[[], None]] = []
        self.data: CarlaData | None = None
        self._speed: float = 0.5

    @final
    def _set_data(self, data: CarlaData) -> None:
        """Sets self.data and notifies listeners

        Args:
            image (Image): the image that is added to the buffer
        """
        self.data = data
        for listener in self.__data_listeners:
            listener(data)

    @final
    def set_speed(self, speed: float) -> None:
        """Sets the ego vehicle speed"""
        self._speed = speed

    @final
    def add_listener(self, callback: Callable[[CarlaData], None]) -> None:
        self.__data_listeners.append(callback)

    @final
    def add_tick_listener(self, callback: Callable[[], None]) -> None:
        self.__tick_listeners.append(callback)

    @final
    def remove_tick_listener(self, callback: Callable[[], None]) -> None:
        try:
            self.__tick_listeners.remove(callback)
        except ValueError:
            logger.warning("Tried to remove a listener that was not present")

    @final
    def await_next_tick(self) -> None:
        """Blocks thread until the next tick has happened"""
        tick_event = Event()

        def set_has_ticked():
            logger.debug("New tick signalled")
            tick_event.set()

        self.add_tick_listener(set_has_ticked)
        tick_event.wait()
        self.remove_tick_listener(set_has_ticked)

    @final
    def _notify_tick_listeners(self) -> None:
        """Notify all tick listeners, MUST be called after every world tick"""
        logger.debug("Tick listeners notified")
        for listener in self.__tick_listeners:
            listener()
