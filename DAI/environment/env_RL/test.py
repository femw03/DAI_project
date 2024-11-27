import numpy as np
import gym

from carlaEnv import CarlaEnv

# Configuration for CarlaEnv
config = {}

# Initialize the environment
env = CarlaEnv(config)

def test_environment(env):
    print("Testing CarlaEnv environment...")
    
    # Test reset function
    print("Resetting the environment...")
    initial_observation = env.reset()
    print("Initial Observation:")
    print(initial_observation)
    
    # Check if the observation matches the defined observation space
    #assert env.observation_space.contains(initial_observation), "Initial observation does not match observation space!"
    
    # Test action space
    print("\nTesting action space...")
    random_action = env.action_space.sample()
    print("Random action:", random_action)
    
    # Test step function
    print("\nTaking a step in the environment...")
    observation, reward, done, info = env.step(random_action)
    print("New Observation:")
    print(observation)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
    
    # Check if the new observation matches the defined observation space
    assert env.observation_space.contains(observation), "New observation does not match observation space!"
    print("\nEnvironment tests passed!")

# Run the test
test_environment(env)
