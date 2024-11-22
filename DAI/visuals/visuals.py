from __future__ import annotations

import os
from enum import Enum
from threading import Thread
from typing import Callable, List, Tuple

import numpy as np
import pygame
from loguru import logger
from pygame.locals import K_ESCAPE, K_RIGHT, K_i

from ..interfaces import Object
from ..platform import list_tiger_vnc_displays
from .visual_utils import add_object_information


class VisualType(Enum):
    RGB = 0
    DEPTH = 1

    @property
    def next(self) -> VisualType:
        members = list(VisualType)
        current_index = members.index(self)
        next_index = (current_index + 1) % len(members)
        return members[next_index]


class Visuals(Thread):
    """
    Creates a pygame window of the given size at the given FPS that displays information of the agent
    controls:
     - ESC: close pygame window and trigger self.on_quit
     - ARROW_RIGHT: cycle the current view between the values of [VisualType]
     - i: toggle overlaying object Information
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        super().__init__()

        self.width = width
        """Width of the window"""
        self.height = height
        """Height of the window"""
        self.fps = fps
        """FPS of the window"""
        self.on_quit: Callable[[], None] = None
        """Is called when the user wants to quit"""

        self.rgb_image: np.ndarray = np.empty(
            (self.height, self.width, 3), dtype=np.uint8
        )
        """The rgb image to be blit onto the display if this option is selected"""

        self.depth_image: np.ndarray = np.empty(
            (self.height, self.width, 1), dtype=np.uint8
        )
        """The depth image to be blit onto the display if this option is selected"""

        self.detected_objects: List[Object] = []
        """The objects that are detected and used to draw bounding boxes"""

        self.running = False
        """Controls if the process is running and can be used to programmatically kill the game"""

        self.visuals_type = VisualType.RGB
        self.display_object_information: bool = False

    def setup(self) -> Tuple[pygame.Surface, pygame.time.Clock]:
        logger.info("Settting up pygame")
        vnc_display = list_tiger_vnc_displays()
        if len(vnc_display) == 0:
            raise RuntimeError("No screens where available to attach to")
        logger.info(f"Using display {vnc_display[0]}")
        os.environ["DISPLAY"] = vnc_display[0]

        pygame.init()
        display = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
        clock = pygame.time.Clock()
        return display, clock

    def run(self) -> None:
        display, clock = self.setup()
        logger.info("Starting pygame visuals")
        self.running = True
        while self.running:
            clock.tick(self.fps)
            self.process_events(pygame.event.get())
            self.blit_on_display(display)

            pygame.display.flip()

        logger.info("Closing the game window")
        pygame.quit()

    def blit_on_display(self, display: pygame.Surface) -> None:
        if self.visuals_type == VisualType.RGB:
            selected_image = self.rgb_image
        elif self.visuals_type == VisualType.DEPTH:
            selected_image = self.depth_image
        else:
            raise RuntimeError("Invalid image type setting")

        if self.display_object_information:
            selected_image = add_object_information(
                selected_image, self.detected_objects
            )

        display.blit(
            pygame.surfarray.make_surface(selected_image.swapaxes(0, 1)), (0, 0)
        )

    def process_events(self, events: List[pygame.event.Event]) -> None:
        for event in events:
            if event.type == pygame.KEYDOWN:
                self.process_key(event.key)

    def process_key(self, key: int) -> None:
        if key == K_ESCAPE:
            logger.info("Stopping game")
            self.on_quit()
            self.running = False

        if key == K_RIGHT:
            logger.info("Cycling view")
            self.visuals_type = self.visuals_type.next

        if key == K_i:
            self.display_object_information = not self.display_object_information
            logger.info(
                f"{'enabling' if self.display_object_information else 'disabling'} detected object information"
            )
