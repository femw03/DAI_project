import os
from threading import Thread
from typing import Callable, Tuple

import numpy as np
import pygame
from loguru import logger
from pygame.locals import K_ESCAPE

from ..platform import list_tiger_vnc_displays


class Visuals(Thread):
    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        on_quit: Callable[[], None],
    ) -> None:
        super().__init__()

        self.width = width
        """Width of the window"""
        self.height = height
        """Height of the window"""
        self.fps = fps
        """FPS of the window"""
        self.on_quit = on_quit
        """Is called when the user wants to quit"""

        self.rgb_image: np.ndarray = None
        """The rgb image to be blit onto the display if this option is selected"""

        self.running = False
        """Controls if the process is running and can be used to programmatically kill the game"""

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

        self.rgb_image = np.empty((self.height, self.width, 3), dtype=np.uint8)
        return display, clock

    def process_keys(self, keys: Tuple[bool]) -> None:
        if keys[K_ESCAPE]:
            logger.info("Stopping game")
            self.on_quit()
            self.running = False

    def blit_on_display(self, display: pygame.Surface) -> None:
        display.blit(
            pygame.surfarray.make_surface(self.rgb_image.swapaxes(0, 1)), (0, 0)
        )

    def run(self) -> None:
        display, clock = self.setup()
        logger.info("Starting pygame visuals")
        self.running = True
        while self.running:
            clock.tick(self.fps)
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            self.process_keys(keys)
            self.blit_on_display(display)

            pygame.display.flip()

        logger.info("Closing the game window")
        pygame.quit()

    def set_rgb_image(self, image: np.ndarray) -> None:
        self.rgb_image = image
