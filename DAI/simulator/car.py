import random
from typing import Callable
import weakref

import carla
import cv2
import numpy as np
import pygame
import pygame.locals as key


class Car():
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self, framerate=30, view_width = 1920//2, view_height = 1080//2, view_FOV=90):
        self.world = None
        self.camera = None
        self.camera_segmentation = None
        self.car = None
        self.depth = None

        self.display = None
        self.image = None
        self.depth_data = None
        self.segmentation_image = None

        self.capture = True
        self.capture_segmentation = True
        self.capture_depth = True

        self.record = True
        self.seg_record = False
        self.rgb_record = False
        self.depth_record = False

        self.screen_capture = 0 
        self.loop_state = False 

        self.framerate = framerate
        self.view_width = view_width
        self.view_height = view_height
        self.view_FOV = view_FOV

    def camera_blueprint(self, filter):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find(filter)
        camera_bp.set_attribute('image_size_x', str(self.view_width))
        camera_bp.set_attribute('image_size_y', str(self.view_height))
        camera_bp.set_attribute('fov', str(self.view_FOV))
        camera_bp.set_attribute('sensor_tick', str(self.framerate))
        return camera_bp

    def depth_blueprint(self, filter):
        """
        Returns depth camera blueprint
        """
        depth_bp = self.world.get_blueprint_library().find(filter)
        depth_bp.set_attribute('image_size_x', str(self.view_width))
        depth_bp.set_attribute('image_size_y', str(self.view_height))
        depth_bp.set_attribute('fov', str(self.view_FOV))
        depth_bp.set_attribute('sensor_tick', str(self.framerate))
        return depth_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self, world):
        """
        Spawns actor-vehicle to be controled.
        """
        self.world = world
        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)


    def setup_depth(self):
        """
        Spawns actor-lidar to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        depth_transform = carla.Transform(carla.Location(x=1.6, z=1.8), carla.Rotation(pitch=-15))
        self.depth = self.world.spawn_actor(self.depth_blueprint('sensor.camera.depth'), depth_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.depth.listen(lambda depth_data: weak_self().set_depth(weak_self, depth_data))


    def setup_camera(self, onImage: Callable[[np.ndarray], None]):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """
        #camera_transform = carla.Transform(carla.Location(x=1.5, z=2.8), carla.Rotation(pitch=-15))
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(self.camera_blueprint('sensor.camera.rgb'), camera_transform, attach_to=self.car)
        self.camera.listen(onImage)

        calibration = np.identity(3)
        calibration[0, 2] = self.view_width / 2.0
        calibration[1, 2] = self.view_height / 2.0
        calibration[0, 0] = calibration[1, 1] = self.view_width / (2.0 * np.tan(self.view_height * np.pi / 360.0))
        self.camera.calibration = calibration

    def control(self, keys: pygame.key, auto_control = False) -> False:
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        control = self.car.get_control()
        if auto_control:
            self.car.set_autopilot(True)
                
        if keys[key.K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[key.K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[key.K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[key.K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0

        self.car.apply_control(control)
        return False

    @staticmethod
    def set_depth(weak_self, depth_data):
        """
        Sets image coming from depth sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """
        self = weak_self()
        if self.capture_depth:
            self.depth_data = depth_data
            self.capture_depth = False
        
        
        if self.depth_record:
            # Capture the depth data when the flag is set
            self.depth_data = depth_data
            self.capture_depth = False  # Reset the capture flag
        
            # Process the depth image (in BGRA)
            depth_array = np.frombuffer(depth_data.raw_data, dtype=np.dtype("uint8"))
            depth_array = np.reshape(depth_array, (depth_data.height, depth_data.width, 4))  # BGRA format

            # Create an RGB image using the first three channels (R, G, B)
            depth_image_rgb = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype=np.uint8)  # Initialize RGB image
            depth_image_rgb[:, :, 0] = depth_array[:, :, 2]  # Extract R channel
            depth_image_rgb[:, :, 1] = depth_array[:, :, 1]  # Extract G channel
            depth_image_rgb[:, :, 2] = depth_array[:, :, 0]  # Extract B channel

            cv2.imwrite('depth_rgb/image' + str(self.image_count) + '.png', depth_image_rgb) 
            print("Depth Image RGB")

            # Step 2: Convert to grayscale logarithmic depth and save
            depth_data.convert(carla.ColorConverter.LogarithmicDepth)

            # Convert to numpy array again after applying ColorConverter
            log_depth_array = np.frombuffer(depth_data.raw_data, dtype=np.uint8)
            log_depth_array= np.reshape(log_depth_array, (depth_data.height, depth_data.width, 4))  # In BGRA format
            depth_image_grayscale = log_depth_array[:, :, 2] #the grayscale values reside in the R channel
            cv2.imwrite('depth_gray/image' + str(self.image_count) + '.png', depth_image_grayscale) 
            print("Depth Image Grayscale")
            

    def destroy(self) -> None:
        self.car.destroy()
        self.camera.destroy()
        self.depth.destroy()
