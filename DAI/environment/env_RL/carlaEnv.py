import sys, os
sys.path.append(os.getcwd())

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from loguru import logger
import random

from DAI.simulator.extract import get_objects, get_current_max_speed, get_current_speed, get_current_affecting_light_state, has_collided
from DAI.simulator.wrappers import CarlaTrafficLightState
from DAI.interfaces import Object, ObjectType
from .carla_setup import setup_carla
import wandb

class CarlaEnv(gym.Env):
  """Custom Gym Environment for RL with variable-length object detections."""

  def __init__(self, config):
    # Configuration for the environment
    logger.info("Startup environment")
    self.collisionCounter = 0
    self.config = config
    self.perfect = config["perfect"]
    self.world_max_speed = config["world_max_speed"]
    self.world = setup_carla()
    self.max_objects = config["max_objects"]    # Maximum number of objects per observation
    self.relevant_distance = config["relevant_distance"]
    self.collisionFlag = False
    
    self.static_dim = 2                         # Static features space: speed limit, current speed
    self.task_dim = 4                           # Task-specific features space: StopFlag, distance to stop line, CrossingFlag, distance to crossing
    self.object_feature_dim = 4                 # Number of object detection features per object: type, confidence, distance, angle
    
    # Calculate the total observation space size
    total_obs_dim = self.static_dim + self.task_dim + self.max_objects * self.object_feature_dim

    # Define observation space as a single Box space
    self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)
    
    # Define actions space
    self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)       # low = full brake          high = full throttle
                                                                                          # 1 action, NOT seperate brake and throttle, because car cannot brake and throttle at same time  
    logger.info("Successfully setup environment")


  def reset(self, seed=None, **kwargs):
      """
      Reset the environment to initial state and return initial observation.
      """
      print("resetting")
      # Reset Carla world
      self.world.reset()

      # Set the random seed if provided
      if seed is not None:
          np.random.seed(seed)  # Set the seed for numpy's random number generator
            
      # Initial states
      self.collisionFlag = False
      self.static_features = np.zeros(self.static_dim, dtype=np.float32)
      self.task_features = np.zeros(self.task_dim, dtype=np.float32)  # Flatten task features
      
      # Generate random number of objects (variable length)
      num_objects = self.max_objects
      self.object_features = np.zeros((num_objects, self.object_feature_dim), dtype=np.float32)

      # Combine all features into one flattened array
      self.current_observation = np.concatenate([self.static_features, self.task_features.flatten(), self.object_features.flatten()])

      return self.current_observation, {}
  
  
  def step(self, action):
      """
      Apply an action and return the new observation, reward, and done.
      """
      #print("stepping")
      # Apply action
      action = action[0]
      #logger.info(f"Executing step with action {action}")
      self.world.set_speed(action)

      wandb.log({"action": action})
      
      # TO DO: wait for execution action in Carla ?
      
      # Compute reward
      reward = self._get_reward(action)
      
      # Check if the episode is terminated
      terminated = self._terminated()
      truncated = False
      info = {}
      
      # Update observation
      if self.perfect:
        self.current_observation = self._get_perfect_obs()      # use observation from Carla
      else:
        self.current_observation = self._get_obs()              # use observation from computer vision
      
      return self.current_observation, reward, terminated, truncated, info


  def _get_perfect_obs(self):
    object_list = get_objects(self.world)
    speed_limit = get_current_max_speed(self.world)
    current_speed = get_current_speed(self.world)
    traffic_light = get_current_affecting_light_state(self.world)
    distance_to_stop = None
    distance_to_crossing = None
    
    # Make Stop Flag
    if traffic_light == CarlaTrafficLightState.RED or traffic_light == CarlaTrafficLightState.YELLOW: 
      stop_flag = 1
    else: 
      stop_flag = 0
    
    # Remove traffic lights and traffic signs and normalize distance
    filtered_objects = []
    for obj in object_list:
      if obj.type not in [ObjectType.TRAFFIC_LIGHT, ObjectType.TRAFFIC_SIGN] and obj.distance <= self.relevant_distance+1:
        obj.distance = min(1, obj.distance / self.relevant_distance)
        if obj.type == ObjectType.PEDESTRIAN:
          obj.type = 0              # pedestrians
        else:
          obj.type = 1              # vehicles, bicycles, motorcycles
        filtered_objects.append(obj)
    # Sort by decreasing distance level
    filtered_objects.sort(key=lambda obj: obj.distance)         # because confidence = 1 (perfect information)
    # Truncate list to have the size of max_objects
    if len(filtered_objects) > self.max_objects:
      filtered_objects = filtered_objects[:self.max_objects]
    else:
      # Perform zero padding
      padding_needed = self.max_objects - len(filtered_objects) 
      padding = [Object(type=random.randint(0, 1), confidence=0.0, distance=0.0, angle=0.0) for _ in range(padding_needed)] 
      filtered_objects.extend(padding)

    #logger.info(f"Object list: {object_list}")
    #logger.info(f"Filtered objects observation: {filtered_objects}")
    #logger.info(f"Speed limit: {speed_limit}")
    #logger.info(f"Current speed: {current_speed}")
    #logger.info(f"Stop flag: {stop_flag}")

    # Static features (speed_limit, current_speed)
    static_features = np.array([speed_limit, current_speed], dtype=np.float32)
    
    # Task features
    task_features = np.array([stop_flag, 0, 0, 0], dtype=np.float32)  # Placeholder for "DistanceToStopLine", "CrossingDetected", "DistanceToCrossing"
    
    # Object features
    object_features = np.array([[obj.type, obj.confidence, obj.distance, obj.angle] for obj in filtered_objects], dtype=np.float32)
    
    # Flatten the entire observation into a single vector
    observation = np.concatenate([static_features, task_features, object_features.flatten()])

    #logger.info(f"Observation: {observation}")

    return observation

  # TO DO
  def _get_obs(self):
    """
    Generate the current observation combining static features, task-specific features,
    and padded object detection features.
    """
    # Output computer vision network: [static_features, task_features, object_features]
    # static_features = [speed_limit, current_speed]
    # task_features = {"StopFlag": stop_flag, "DistanceToStopLine":distance_to_stopline]
    # object_features = [[class, angle, distance], [class, angle, distance], [class, angle, distance], ...] 
    
    # Static features (speed_limit, current_speed)
    static_features = np.random.uniform(0, 100, size=(self.static_dim,))
    
    # Task features (StopFlag, DistanceToStopLine, CrossingDetected, DistanceToCrossing)
    task_features = np.array([
        np.random.choice([0, 1]),  # Random StopFlag
        np.random.randint(0, 101),  # Random distance to stop line
        np.random.choice([0, 1]),  # Random CrossingDetected
        np.random.randint(0, 101)   # Random distance to crossing
    ], dtype=np.float32)
    
    # Object features (randomly generated)
    num_objects = np.random.randint(1, self.max_objects + 1)
    object_features = np.random.uniform(-1, 1, size=(num_objects, self.object_feature_dim))
    
    # Padding to ensure we have max_objects number of objects
    padded_objects = np.zeros((self.max_objects, self.object_feature_dim), dtype=np.float32)
    padded_objects[:num_objects] = object_features
    
    # Flatten the entire observation into a single vector
    observation = np.concatenate([static_features, task_features, padded_objects.flatten()])

    return observation


  def _get_reward(self, action):
    """
    Calculate reward based on the action and the current environment state.
    Penalize high acceleration or braking and reward being closer to the task goal.
    """
    object_list = get_objects(self.world)
    speed_limit = get_current_max_speed(self.world)
    current_speed = get_current_speed(self.world)
    traffic_light = get_current_affecting_light_state(self.world)
    collision = has_collided(self.world)
    
    """ Make Stop Flag """
    if traffic_light == CarlaTrafficLightState.RED or traffic_light == CarlaTrafficLightState.YELLOW: 
      stop_flag = 1 
    else: 
      stop_flag = 0
      
    """ Filter objects """
    filtered_objects = []
    for obj in object_list:
      if obj.type not in [ObjectType.TRAFFIC_LIGHT, ObjectType.TRAFFIC_SIGN]:
        filtered_objects.append(obj)
    #print(object_list)
    """ Reward """
    # Initialize rewards (= 0 -> no influence)
    safe_distance_reward = 0
    slow_speed_reward = 0
    fast_speed_reward = 0
    stop_reward = 0
    pedestrian_reward = 0
    crash_reward = 0
    
    """ Collision """
    if collision:
      crash_reward = -20      # Harsh penalty for crashing in object
      self.collisionFlag = True
    
    for object in filtered_objects:
      #print("found object!!!")
      if object.type == ObjectType.PEDESTRIAN:                # Pedestrians
        #print("saw pedestrian")
        if abs(object.angle) < np.radians(45) :           # angle of attack = +-45°
          
          """Stop before pedestrian crossing"""
          """if task_features["CrossingDetected"]:
            if static_features[1] < 0.1:      # Car is considered stopped: remove stop flag
              # Stop 1m from pedestrian crossing
              if task_features["DistanceToCrossing"] > 0.9 and task_features["DistanceToCrossing"] < 1.1:
                pedestrian_reward = 1                           # Bonus for stopping close to pedestrian crossing
              elif task_features["DistanceToCrossing"] > 1.1:
                pedestrian_reward = -0.1*task_features["DistanceToCrossing"]       # Smaller penalty for stopping slightly before the stop line
              else:
                pedestrian_reward = -2                          # Penalty for overshooting the stop line
            
            else:
              pedestrian_reward = -0.1*static_features[1]       # Penalty for moving when a stop is required"""
      
      else:
        if abs(object.angle) < np.radians(20) and object.distance != 0:           # angle of attack = +-20°
          #print("car infront!!! distance: ", object.distance)
          """Safe distance"""
          if current_speed > 1:
            safe_distance = 2 * current_speed       # Approximation of 2 seconds * current_speed
          else: 
            safe_distance = 2   # to make sure that if he is stationary he won't try to get to zero meters between cars!!
          distance_margin = 0.1*safe_distance
                  
          if object.distance <= 0.5*safe_distance:
            safe_distance_reward = max(-5, -0.5 * ((0.5*safe_distance)/object.distance))          # Large penalty for being too close (unsafe)

          elif object.distance < safe_distance+distance_margin and object.distance > safe_distance-distance_margin:
            safe_distance_reward = 1                                    # Bonus for staying within 10% of safe distance
          
          elif current_speed < speed_limit:
            safe_distance_reward = max(-1, -0.1 * (object.distance/safe_distance))     # Smaller penalty for trailing too far behind
          #print("safe_distance_reward: ", safe_distance_reward)
        else:               # No vehicle in angle-of-attack
          
          """Driving too slow"""
          #print("spotted car")
          speed_margin = 0.1*speed_limit
          if speed_limit-speed_margin > current_speed and current_speed >= 0.1:
            slow_speed_reward = -0.01 * (speed_limit/current_speed)
          elif current_speed < 0.1:
            slow_speed_reward = -2


    # to deal with speedlimit if no objects have been detected        
    if len(filtered_objects) == 0:        
      """Driving too slow"""
      speed_margin = 0.1*speed_limit
      if speed_limit-speed_margin > current_speed and current_speed >= 0.1:
        slow_speed_reward = -0.01 * (speed_limit/current_speed)
      elif current_speed < 0.1:
        slow_speed_reward = -2
      
    """Following speed limit"""
    if speed_limit < current_speed:
      fast_speed_reward = -0.1 * (current_speed/speed_limit)
    
    """Progress""" 
    progress_reward = 0.1 # small positive reward for every step he didn't colide, we hope he tries to stay alive as long as possible this way
    if self.world.local_planner.done():
      progress_reward = 2  # because he finishes early, without crashing!
    # TO DO: remove stop flag when car is stopped
    """Stop line"""
    """if stop_flag:
      if current_speed < 0.1:      # Car is considered stopped: remove stop flag
        stop_flag = 0
        
        # Stop 1m from stop line
        if task_features["DistanceToStopLine"] > 0.9 and task_features["DistanceToStopLine"] < 1.1:
          stop_reward = 1                           # Bonus for stopping close to stop line
        elif task_features["DistanceToStopLine"] > 1.1:
          stop_reward = -0.1*task_features["DistanceToStopLine"]       # Smaller penalty for stopping slightly before the stop line
        else:
          stop_reward = -2                          # Penalty for overshooting the stop line
      
      else:
        stop_reward = -0.1*static_features[1]       # Penalty for moving when a stop is required
    else:
      stop_reward = 0"""
    
    #reward = safe_distance_reward + slow_speed_reward + fast_speed_reward + stop_reward + crash_reward + pedestrian_reward
    reward = slow_speed_reward + fast_speed_reward + crash_reward + safe_distance_reward
    wandb.log({"slow_speed_reward": slow_speed_reward,
               "fast_speed_reward": fast_speed_reward,
               "crash_reward": crash_reward,
               "safe_distance_reward": safe_distance_reward,
               "reward": reward,
               "speed_limit": speed_limit,
               "current_speed": current_speed})
    return reward

  def _terminated(self):
    """
    Determine if the episode is terminated based on task-specific or static conditions.
    """
    if self.world.local_planner.done():
      done = True
      print("made it to destination in time without crash!")
    else:
      done = False
    if self.collisionFlag:
      collision = True
      self.collisionCounter += 1
    else:
      collision = False
    wandb.log({"number of collisions": self.collisionCounter})
    return collision or done