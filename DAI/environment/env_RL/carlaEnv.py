import numpy as np
import gym
from gym import spaces


class CarlaEnv(gym.Env):
  """Custom Gym Environment for RL with variable-length object detections."""

  def __init__(self, config):
    # Configuration for the environment
    self.config = config
    self.static_dim = 2                     # Static features space: speed limit, current speed
    self.task_dim = 2                       # Task-specific features space: StopFlag, distance to stop line
    self.object_feature_dim = 3             # Number of object detection features per object: class embedding, angle, distance
    self.max_objects = 30                   # Maximum number of objects per observation
    
    # Define observation space
    self.static_space = spaces.Box(low=0, high=100, shape=(self.static_dim,), dtype=np.float32)       # TO DO: map to 0-1     
    self.task_space = spaces.Dict({
            "StopFlag": spaces.Discrete(2),  # Boolean: 0 (no stop) or 1 (stop required)
            "DistanceToStopLine": spaces.Box(low=0, high=100, shape=(), dtype=np.int32),              # TO DO: map to 0-1
            "CrossingDetected": spaces.Discrete(2),  # Boolean: 0 (no pedestrian crossing) or 1 (crossing detected),
            "DistanceToCrossing": spaces.Box(low=0, high=100, shape=(), dtype=np.int32),              # TO DO: map to 0-1
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
      # Random initial states
      static_features = np.random.uniform(0, 100, size=(self.static_dim,))
      task_features = {
            "StopFlag": np.random.choice([0, 1]),
            "DistanceToStopLine": np.random.randint(0, 101),
            "CrossingDetected": np.random.choice([0, 1]),
            "DistanceToCrossing": np.random.randint(0, 101),
        }
      
      # Generate random number of objects (variable length)
      num_objects = np.random.randint(1, self.max_objects + 1)
      object_features = np.random.uniform(-1, 1, size=(num_objects, self.object_feature_dim))
      
      # Pad objects to max_objects with zeros
      padded_objects = np.zeros((self.max_objects, self.object_feature_dim), dtype=np.float32)
      padded_objects[:num_objects] = object_features

      # Store observation
      self.current_observation = {
          "static": static_features,
          "task": task_features,
          "objects": padded_objects,
      }

      return self.current_observation
  
  
  def step(self, action):
      """
      Apply an action and return the new observation, reward, and done.
      """
      # Apply action
      action = action[0]
      
      # Compute reward
      reward = self._get_reward(action)
      
      # Check if the episode is terminated
      done = self._terminated()
      
      # Update observation
      self.current_observation = self._get_obs()

      return self.current_observation, reward, done, {}


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
    static_features = self.current_observation["static"]
    task_features = self.current_observation["task"]
    object_features = self.current_observation["objects"]
    
    for object in object_features:
      if object[1] == 1:                # Pedestrians
        if abs(object[1]) < np.radians(45) :           # angle of attack = +-45°
          
          """Stop before pedestrian crossing"""
          if task_features["CrossingDetected"]:
            if static_features[1] < 0.1:      # Car is considered stopped: remove stop flag
              # Stop 1m from pedestrian crossing
              if task_features["DistanceToCrossing"] > 0.9 and task_features["DistanceToCrossing"] < 1.1:
                stop_reward = 1                           # Bonus for stopping close to stop line
              elif task_features["DistanceToCrossing"] > 1.1:
                stop_reward = -0.1*task_features["DistanceToCrossing"]       # Smaller penalty for stopping slightly before the stop line
              else:
                stop_reward = -2                          # Penalty for overshooting the stop line
            
            else:
              stop_reward = -0.1*static_features[1]       # Penalty for moving when a stop is required
      
      else:
        if abs(object[1]) < np.radians(20) :           # angle of attack = +-20°
          
          """Safe distance"""
          safe_distance = 2 * static_features[1]       # Approximation of 2 seconds * current_speed
          distance_margin = 0.1*safe_distance
                  
          if object[2] == 0.1:
            safe_distance_reward = -5                                   # Harsh penalty for crashing in object
          elif object[2] <= 0.5*safe_distance:
            safe_distance_reward = -2                                   # Large penalty for being too close (unsafe)
          elif object[2] < safe_distance+distance_margin and object[2] > safe_distance-distance_margin:
            safe_distance_reward = 1                                    # Bonus for staying within 10% of safe distance
          else:
            safe_distance_reward = -0.1 * (object[2]/safe_distance)     # Smaller penalty for trailing too far behind
        
        else:               # No vehicle in angle-of-attack
          
          """Driving too slow"""
          speed_margin = 0.1*static_features[0]
          if static_features[0]-speed_margin > static_features[1]:
            slow_speed_reward = -0.1 * (static_features[1]/static_features[0])
          else:
            slow_speed_reward = 0
              
    """Following speed limit"""
    if static_features[0] < static_features[1]:
      fast_speed_reward = -0.1 * (static_features[1]/static_features[0])
    else:
      fast_speed_reward = 0
    
    # TO DO: remove stop flag when car is stopped
    """Stop line"""
    if task_features["StopFlag"]:
      if static_features[1] < 0.1:      # Car is considered stopped: remove stop flag
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
      stop_reward = 0
        
    # Penalize large actions
    action_penalty = -np.sum(np.square(action))
    
    reward = safe_distance_reward + slow_speed_reward + fast_speed_reward + stop_reward + action_penalty

    return reward


  def _terminated(self):
    """
    Determine if the episode is terminated based on task-specific or static conditions.
    """
    pass  