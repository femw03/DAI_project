"""Contains the Definition of The Visuals class"""

from __future__ import annotations

import os
from enum import Enum
from threading import Thread
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pygame
import pygame_widgets
from loguru import logger
from pygame.locals import K_ESCAPE, K_RIGHT, K_i, K_r
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox

from ..platform import list_tiger_vnc_displays
from .visual_utils import (
    ObjectDTO,
    add_object_information,
    add_static_information,
    draw_trajectory_line,
)


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

        self.on_reset: Callable[[], None] = None
        """Is called when the user wants to reset the simulation"""
        self.on_pause: Callable[[bool], None] = None
        """Is called when the user wants to pause the simulation"""
        self.is_paused = False
        self.rgb_image: np.ndarray = np.empty(
            (self.height, self.width, 3), dtype=np.uint8
        )
        """The rgb image to be blit onto the display if this option is selected"""

        self.depth_image: np.ndarray = np.empty(
            (self.height, self.width, 1), dtype=np.uint8
        )
        """The depth image to be blit onto the display if this option is selected"""

        self.detected_objects: Optional[List[ObjectDTO]] = None
        """The objects that are detected and used to draw bounding boxes"""
        self.information: Optional[Dict[str, str]] = None

        self.running = False
        """Controls if the process is running and can be used to programmatically kill the game"""

        self.visuals_type = VisualType.RGB
        self.display_object_information: bool = False
        self.angle = 0
        self.correction_factor = 0.109
        self.boost_factor = 45
        self.margin = 845
        self.horizon_factor = 0.48

    def setup(self) -> Tuple[pygame.Surface, pygame.time.Clock]:
        logger.info("Settting up pygame")
        vnc_display = list_tiger_vnc_displays()
        if len(vnc_display) == 0:
            raise RuntimeError("No screens where available to attach to")
        logger.info(f"Using display {vnc_display[0]}")
        os.environ["DISPLAY"] = vnc_display[0]

        pygame.init()
        display = pygame.display.set_mode((self.width, self.height), pygame.DOUBLEBUF)
        self.slider_1 = Slider(
            display,
            self.width - 300,
            40,
            200,
            40,
            min=1e-3,
            max=5,
            step=1e-5,
            initial=self.correction_factor,
        )
        self.slider_value_1 = TextBox(display, self.width - 100, 120, 40, 40)
        self.slider_value_1.disable()
        self.slider_2 = Slider(
            display,
            self.width - 300,
            200,
            200,
            40,
            min=1,
            max=100,
            step=1e-5,
            initial=self.boost_factor,
        )
        self.slider_value_2 = TextBox(display, self.width - 100, 280, 40, 40)
        self.slider_value_2.disable()
        self.slider_3 = Slider(
            display,
            self.width - 300,
            360,
            200,
            40,
            min=0,
            max=self.width,
            step=1,
            initial=self.margin,
        )
        self.slider_value_3 = TextBox(display, self.width - 100, 440, 40, 40)
        self.slider_value_3.disable()

        clock = pygame.time.Clock()
        return display, clock

    def run(self) -> None:
        display, clock = self.setup()
        logger.info("Starting pygame visuals")
        self.running = True
        while self.running:
            clock.tick(self.fps)
            self.blit_on_display(display)
            self.correction_factor = self.slider_1.value
            self.slider_value_1.setText(self.slider_1.value)

            self.boost_factor = self.slider_2.value
            self.slider_value_2.setText(self.slider_2.value)

            self.margin = self.slider_3.value
            self.slider_value_3.setText(self.slider_3.value)

            self.process_events(pygame.event.get())

            pygame.display.flip()

        logger.info("Closing the game window")
        pygame.quit()

    def blit_on_display(self, display: pygame.Surface) -> None:
        if self.visuals_type == VisualType.RGB:
            selected_image = self.rgb_image
        elif self.visuals_type == VisualType.DEPTH:
            # depth is given as float between 0 and 1
            depth = (self.depth_image * 255).astype(np.uint8)
            selected_image = np.stack([depth] * 3, axis=-1)
        else:
            raise RuntimeError("Invalid image type setting")

        if self.display_object_information:
            if self.detected_objects is not None:
                selected_image = add_object_information(
                    selected_image, self.detected_objects
                )
            if self.information is not None:
                selected_image = add_static_information(
                    selected_image,
                    self.information,
                )
            if self.depth_image is not None:
                selected_image = draw_trajectory_line(
                    selected_image,
                    self.depth_image,
                    self.angle,
                    self.correction_factor,
                    self.boost_factor,
                    margin=self.margin,
                    at_y=self.horizon_factor,
                )

        display.blit(
            pygame.surfarray.make_surface(selected_image.swapaxes(0, 1)), (0, 0)
        )

    def process_events(self, events: List[pygame.event.Event]) -> None:
        for event in events:
            if event.type == pygame.KEYDOWN:
                self.process_key(event.key)
        pygame_widgets.update(events)

    def process_key(self, key: int) -> None:
        if key == K_ESCAPE:
            logger.info("Stopping game")
            if self.on_pause is None and self.is_paused:
                self.on_pause(False)
            self.on_quit()  # error: TypeError: stop() missing 1 required positional argument: 'world'
            self.running = False

        if key == K_RIGHT:
            logger.info("Cycling view")
            self.visuals_type = self.visuals_type.next

        if key == K_i:
            self.display_object_information = not self.display_object_information
            logger.info(
                f"{'enabling' if self.display_object_information else 'disabling'} detected object information"
            )
        if key == pygame.K_p and self.on_pause is not None:
            self.is_paused = not self.is_paused
            self.on_pause(self.is_paused)
        if key == K_r:
            logger.info("Resetting simulation")
            self.on_reset()
