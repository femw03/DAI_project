import numpy as np
import gymnasium as gym
from gymnasium import spaces
from loguru import logger

from DAI.simulator.extract import get_objects, get_current_max_speed, get_current_speed, get_current_affecting_light_state, has_collided
from DAI.simulator.wrappers import CarlaTrafficLightState
from DAI.interfaces import Object, ObjectType

class CarlaEnv(gym.Env):
  """Custom Gym Environment for RL with variable-length object detections."""

  def __init__(self, config):
    # Configuration for the environment
    self.config = config
    self.perfect = config["perfect"]
    self.world_max_speed = config["world_max_speed"]
    self.world = config["world"]
    self.max_objects = config["max_objects"]    # Maximum number of objects per observation
    self.relevant_distance = config["relevant_distance"]
    self.collisionFlag = False
    
    self.static_dim = 2                         # Static features space: speed limit, current speed
    self.task_dim = 2                           # Task-specific features space: StopFlag, distance to stop line
    self.object_feature_dim = 5                 # Number of object detection features per object: type, boundingBox, confidence, distance, angle
    
    # Define observation space
    self.static_space = spaces.Box(low=0, high=1, shape=(self.static_dim,), dtype=np.float32)     
    self.task_space = spaces.Dict({
            "StopFlag": spaces.Discrete(2),  # Boolean: 0 (no stop) or 1 (stop required)
            "DistanceToStopLine": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "CrossingDetected": spaces.Discrete(2),  # Boolean: 0 (no pedestrian crossing) or 1 (crossing detected),
            "DistanceToCrossing": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        })    
    self.object_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_objects, self.object_feature_dim), dtype=np.float32)
    
    # Combine observation spaces into a dictionary
    self.observation_space = spaces.Dict({
        "static": self.static_space,
        "task": self.task_space,
        "objects": self.object_space,
    })
    
    # Define actions space
    self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)       # low = full brake          high = full throttle
                                                                                          # 1 action, NOT seperate brake and throttle, because car cannot brake and throttle at same time
                                                                                          
  def reset(self):
      """
      Reset the environment to initial state and return initial observation.
      """
      # Reset Carla world
      self.world.reset()
            
      # Initial states
      self.collisionFlag = False
      static_features = np.zeros(self.static_dim, dtype=np.float32)
      task_features = {
            "StopFlag": 0,
            "DistanceToStopLine": 0.0,
            "CrossingDetected": 0,
            "DistanceToCrossing": 0.0,
        }
      
      # Generate random number of objects (variable length)
      num_objects = self.max_objects
      object_features = np.zeros((num_objects, self.object_feature_dim), dtype=np.float32)

      # Store observation
      self.current_observation = {
          "static": static_features,
          "task": task_features,
          "objects": object_features,
      }

      return self.current_observation, {}
  
  
  def step(self, action):
      """
      Apply an action and return the new observation, reward, and done.
      """
      # Apply action
      action = action[0]
      logger.info(f"Executing step with action {action}")
      self.world.set_speed(action)
      
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
        filtered_objects.append(obj)
    # Sort by decreasing distance level
    filtered_objects.sort(key=lambda obj: obj.distance)         # because confidence = 1 (perfect information)
    # Truncate list to have the size of max_objects
    if len(filtered_objects) > self.max_objects:
      filtered_objects = filtered_objects[:self.max_objects]
    else:
      # Perform zero padding
      padding_needed = self.max_objects - len(filtered_objects) 
      padding = [Object(type=None, boundingBox=None, confidence=0.0, distance=0.0, angle=0.0) for _ in range(padding_needed)] 
      filtered_objects.extend(padding)

    # Create the observation 
    observation = { 
        "static": np.array([speed_limit, current_speed], dtype=np.float32), 
        "task": { 
          "StopFlag": 0,                # = stop_flag
          "DistanceToStopLine": 0,      # = distance_to_stop (if distance_to_stop = None: distance_to_stop = 0) -> normalize + clipping
          "CrossingDetected": 0,        # = crossing_flag
          "DistanceToCrossing": 0,      # = distance_to_crossing (if distance_to_crossing = None: distance_to_crossing = 0)  -> normalize + clipping
          }, 
        "objects": np.array([[obj.type, obj.boundingBox, obj.confidence, obj.distance, obj.angle] for obj in filtered_objects], dtype=np.float32), 
        } 
  
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
    
    # random to test
    static_features = np.random.uniform(0, 100, size=(self.static_dim,))
    task_features = {
            "StopFlag": np.random.choice([0, 1]),             # Randomly decide if stopping is required
            "DistanceToStopLine": np.random.randint(0, 101),  # Random distance to stop line
            "CrossingDetected": np.random.choice([0, 1]),     # Randomly decide if stopping is required
            "DistanceToCrossing": np.random.randint(0, 101),  # Random distance to pedestrian crossing
        }
    num_objects = np.random.randint(1, self.max_objects + 1)
    object_features = np.random.uniform(-1, 1, size=(num_objects, self.object_feature_dim))
    padded_objects = np.zeros((self.max_objects, self.object_feature_dim), dtype=np.float32)    # Pad objects to max_objects
    padded_objects[:num_objects] = object_features

    # Create the observation
    return {
      "static": static_features,
      "task": task_features,
      "objects": padded_objects,
    }


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
    # Remove traffic lights and traffic signs
    filtered_objects = [
      obj for obj in object_list
      if obj.type not in [ObjectType.TRAFFIC_LIGHT, ObjectType.TRAFFIC_SIGN]
    ]
    
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
      crash_reward = -5      # Harsh penalty for crashing in object
      self.collisionFlag = True
    
    for object in filtered_objects:
      if object.type == ObjectType.PEDESTRIAN:                # Pedestrians
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
          
          """Safe distance"""
          safe_distance = 2 * current_speed       # Approximation of 2 seconds * current_speed
          distance_margin = 0.1*safe_distance
                  
          if object.distance <= 0.5*safe_distance:
            safe_distance_reward = max(-5, -0.5 * ((0.5*safe_distance)/object.distance))          # Large penalty for being too close (unsafe)
          
          elif object.distance < safe_distance+distance_margin and object.distance > safe_distance-distance_margin:
            safe_distance_reward = 1                                    # Bonus for staying within 10% of safe distance
          
          elif current_speed < speed_limit:
            safe_distance_reward = max(-1, -0.1 * (object.distance/safe_distance))     # Smaller penalty for trailing too far behind
        
        else:               # No vehicle in angle-of-attack
          
          """Driving too slow"""
          speed_margin = 0.1*speed_limit
          if speed_limit-speed_margin > current_speed:
            slow_speed_reward = -0.1 * (current_speed/speed_limit)
              
    """Following speed limit"""
    if speed_limit < current_speed:
      fast_speed_reward = -0.1 * (current_speed/speed_limit)
    
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
    
    reward = safe_distance_reward + slow_speed_reward + fast_speed_reward + stop_reward + crash_reward + pedestrian_reward

    return reward


  def _terminated(self):
    """
    Determine if the episode is terminated based on task-specific or static conditions.
    """
    return self.collisionFlag  