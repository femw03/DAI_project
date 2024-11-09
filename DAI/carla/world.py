from __future__ import annotations
import carla
import numpy as np
import pygame
import pygame.locals
from .car import Car
from .spawner import Spawner

class World():
    def __init__(self, framerate = 30, host='127.0.0.1', port=2000,  view_width = 1920//2, view_height = 1080//2, view_FOV=90, walkers = 50, cars = 10) -> None:
        self.framerate = 30        
        self.view_width = view_width
        self.view_height = view_height
        self.view_FOV = view_FOV
        self.host = host
        self.port = port
        self.car = Car(framerate=framerate, view_width=view_width, view_height=view_height, view_FOV=view_FOV)
        
        self.image = None
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(2.0)
        self.spawner = Spawner(self.client, vehicles=cars, walkers=walkers)
    
    def setup(self):
        pygame.init()

        self.world = self.client.get_world()

        def onImage(image_array: np.ndarray) -> None:
            self.image = convert_image(image_array.raw_data, self.view_width, self.view_height)

        self.car.setup_car(self.world)
        self.car.setup_camera(onImage=onImage)
        # self.car.setup_depth() #TODO

        self.display = pygame.display.set_mode((self.view_width, self.view_height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        self.car.set_synchronous_mode(True)
        self.spawner.start()
        self.running = True
    
    def game_loop(self):
        """
        Main program loop.
        """
        try:
            self.setup()
            framecount = 0
            while self.running:
                print(f'\rframe: {framecount}   ', end='')
                self.world.tick()

                self.capture = True
                self.clock.tick_busy_loop(self.framerate) 

                if self.image != None:
                    surface = pygame.surfarray.make_surface(self.image.swapaxes(0, 1))
                    self.display.blit(surface, (0,0))
                    
                pygame.display.flip()

                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[pygame.locals.K_ESCAPE]:
                    self.running = False
                    break
                if self.car.control(keys, auto_control=True):
                    return

        finally:
            self.car.set_synchronous_mode(False)
            self.car.destroy()
            self.spawner.stop()
            pygame.quit()
            
            
def convert_image(image_array, width, height) -> np.ndarray:
    array = np.frombuffer(image_array, dtype=np.dtype("uint8"))
    array = np.reshape(array, (height, width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array 