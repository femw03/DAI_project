import numpy as np
import gymnasium as gym

from carlaEnv import CarlaEnv

# Configuration for CarlaEnv
config = {
    "perfect": True,        # Whether the environment should assume perfect detections (ideal data)
    "max_steps": 1000,      # Maximum number of steps per episode.
}

#def custom_env_creator
# Initialize the environment
env = CarlaEnv(config)

def test_environment(env):
    print("Testing CarlaEnv environment...")
    
    obs = env.reset()

    while True:
        action = [0.0, 1.0]
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs = env.reset()

# Run the test
test_environment(env)
