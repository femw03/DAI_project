#!/usr/bin/env python

"""
An example of client-side bounding boxes with basic car controls.

Controls:
Welcome to CARLA for Getting Bounding Box Data.
Use WASD keys for control.
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    P            : autopilot mode
    C            : Capture Data
    l            : Loop Capture Start
    L            : Loop Capture End

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

import glob
import os
import shutil
import sys

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import argparse
import random
import time
import weakref

import carla
import cv2
from carla import ColorConverter as cc

try:
    import pygame
    from pygame.locals import (
        # K_BACKQUOTE,
        K_ESCAPE,
        K_SPACE,
        # K_TAB,
        # KMOD_SHIFT,
        K_a,
        K_c,
        K_d,
        K_l,
        K_p,
        K_s,
        K_w,
    )
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

try:
    import numpy as np
except ImportError:
    raise RuntimeError("cannot import numpy, make sure numpy package is installed")


VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90

UPPER_FOV_DEPTH = 30.0
LOWER_FOV_DEPTH = -25.0
CHANNELS_DEPTH = 64.0
RANGE_DEPTH = 200.0
POINTS_PER_SEC_DEPTH = 30000
FRAMERATE = 0.05
FRAME_PER_SEC = 20

count = 0

rgb_info = np.zeros((540, 960, 3), dtype="i")
seg_info = np.zeros((540, 960, 3), dtype="i")

# Directories to create/empty
dirs = ["custom_data/", "SegmentationImage/"]

for directory in dirs:
    if os.path.exists(directory):
        # Remove all contents in the directory
        shutil.rmtree(directory)
    # Create the (empty) directory
    os.makedirs(directory)


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.camera_segmentation = None
        self.car = None

        self.display = None
        self.image = None
        self.segmentation_image = None

        self.capture = True
        self.capture_segmentation = True
        self.capture_depth = True

        self.record = True
        self.seg_record = False
        self.rgb_record = False

        self.screen_capture = 0
        self.loop_state = False

    def camera_blueprint(self, filter):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find(filter)
        camera_bp.set_attribute("image_size_x", str(VIEW_WIDTH))
        camera_bp.set_attribute("image_size_y", str(VIEW_HEIGHT))
        camera_bp.set_attribute("fov", str(VIEW_FOV))
        camera_bp.set_attribute("sensor_tick", str(FRAMERATE))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter("vehicle.*")[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """
        seg_transform = carla.Transform(
            carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15)
        )
        self.camera_segmentation = self.world.spawn_actor(
            self.camera_blueprint("sensor.camera.semantic_segmentation"),
            seg_transform,
            attach_to=self.car,
        )
        weak_self = weakref.ref(self)
        self.camera_segmentation.listen(
            lambda image_seg: weak_self().set_segmentation(weak_self, image_seg)
        )

        # camera_transform = carla.Transform(carla.Location(x=1.5, z=2.8), carla.Rotation(pitch=-15))
        camera_transform = carla.Transform(
            carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15)
        )
        self.camera = self.world.spawn_actor(
            self.camera_blueprint("sensor.camera.rgb"),
            camera_transform,
            attach_to=self.car,
        )
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (
            2.0 * np.tan(VIEW_FOV * np.pi / 360.0)
        )
        self.camera.calibration = calibration
        self.camera_segmentation.calibration = calibration

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """
        keys = pygame.key.get_pressed()

        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1.0, min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1.0, max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        if keys[K_p]:
            car.set_autopilot(True)
        if keys[K_c]:
            self.screen_capture = self.screen_capture + 1
        else:
            self.screen_capture = 0

        if keys[K_l]:
            self.loop_state = True
        if keys[K_l] and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            self.loop_state = False
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """
        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

        if self.rgb_record:
            i = np.array(img.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            cv2.imwrite("custom_data/image" + str(self.image_count) + ".png", i3)
            print("RGB(custom)Image")

    @staticmethod
    def set_segmentation(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture_segmentation:
            self.segmentation_image = img
            self.capture_segmentation = False

        if self.seg_record:
            img.convert(cc.CityScapesPalette)
            i = np.array(img.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            cv2.imwrite("SegmentationImage/seg" + str(self.image_count) + ".png", i3)
            print("SegmentationImage")

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def game_loop(self, args):
        """
        Main program loop.
        """

        try:
            pygame.init()

            self.client = carla.Client("127.0.0.1", 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            self.setup_car()
            self.setup_camera()

            self.display = pygame.display.set_mode(
                (VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)

            self.image_count = 0
            self.time_interval = 0

            global count

            while True:
                self.world.tick()

                self.capture = True
                pygame_clock.tick_busy_loop(FRAME_PER_SEC)

                self.render(self.display)

                self.time_interval += 1
                if (self.time_interval % args.CaptureLoop) == 0 and self.loop_state:
                    self.image_count = self.image_count + 1
                    self.rgb_record = True
                    self.seg_record = True
                    count = self.image_count
                    print("-------------------------------------------------")
                    print("ImageCount - %d" % self.image_count)

                if self.screen_capture == 1:
                    self.image_count = self.image_count + 1
                    self.rgb_record = True
                    self.seg_record = True
                    count = self.image_count
                    print("-------------------------------------------------")
                    print("Captured! ImageCount - %d" % self.image_count)

                time.sleep(0.03)
                self.rgb_record = False
                self.seg_record = False
                pygame.display.flip()

                pygame.event.pump()
                if self.control(self.car):
                    return

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.camera_segmentation.destroy()
            self.car.destroy()
            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    Initializes the client-side bounding box demo.
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-l",
        "--CaptureLoop",
        metavar="N",
        default=100,
        type=int,
        help="set Capture Cycle settings, Recommand : above 100",
    )

    args = argparser.parse_args()

    print(__doc__)

    try:
        client = BasicSynchronousClient()
        client.game_loop(args)
    finally:
        print("EXIT")


if __name__ == "__main__":
    main()
