import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import wandb  # Import wandb

class AdaptiveCruiseControlEnv(gym.Env):
    def __init__(self, config={}, render_mode=None):
        super(AdaptiveCruiseControlEnv, self).__init__()
        
        # Define the action space and observation space
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0, 0, 0]), 
                                            high=np.array([130, 130, 1, 110, 1, 100]), 
                                            dtype=np.float32)

        # Simulation parameters
        self.dt = 0.1
        self.max_speed = 30.0
        self.min_distance = 10.0
        self.max_distance = 100.0

        # Initialize pygame for visualization
        self.screen_width = 800
        self.screen_height = 200
        self.car_width = 50
        self.car_height = 20
        self.pixels_per_meter = 8  # Scale for visualization (8 pixels per meter)

        self.render_initialized = False
        self.clock = None
        self.screen = None
        self.episode = 0
        self.episode_reward = 0
        self.distance_to_stop = 0
        self.stop = 0
        self.car_in_front = 1  # Initially assume car is in front
        self.No_stop = False

        # Initialize wandb
        wandb.init(project="adaptive_cruise_control", config=config)
        
        # Reset environment variables
        self.reset()

        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode += 1
        self.agent_speed = np.random.uniform(0, 5)
        self.target_speed = 10.0
        self.current_target_speed = self.target_speed
        self.relative_distance = np.random.uniform(20, 50)
        self.last_action = 0.5  # Initialize to neutral action
        self.last_speed = 0  # Track last speed for acceleration calculation
        self.target_speed_update_interval = 250
        self.time_step_counter = 0
        wandb.log({"episode_reward": self.episode_reward})
        self.episode_reward = 0
        self.car_disappear_counter = 0  # Counter to manage car disappearance
        self.distance_to_stop = np.random.uniform(30, 60)  # Random distance to stop line
        self.stop = 0  # Initially no stop
        self.No_stop = False
        self.car_in_front = 1
        self.max_speed = 30

        return np.array([self.agent_speed, self.max_speed, self.car_in_front, self.relative_distance, self.stop, self.distance_to_stop], dtype=np.float32), {}

    def step(self, action):
        terminated = False
        self.last_action = action[0]  # Store the last action for rendering
        
        if action[0] <= 0.5:
            brake_intensity = (0.5 - action[0]) * 2
            self.agent_speed = max(0.0, self.agent_speed - brake_intensity * 10.0 * self.dt)
        else:
            throttle_intensity = (action[0] - 0.5) * 2
            self.agent_speed = self.agent_speed + throttle_intensity * 2.0 * self.dt

        if self.agent_speed > self.max_speed + 10:
            terminated = True

        # Update target speed gradually
        if self.time_step_counter % self.target_speed_update_interval == 0:
            self.target_speed = np.random.uniform(10, 30)
            self.target_speed_update_interval = 250

        # Gradually adjust current target speed towards target speed
        speed_diff = self.target_speed - self.current_target_speed
        speed_adjustment = np.sign(speed_diff) * min(abs(speed_diff), 0.1)  # Gradual change
        self.current_target_speed += speed_adjustment

        self.time_step_counter += 1

        # Introduce car disappearance randomly
        if self.time_step_counter % 600 == 0:  # Every 600 steps, the car might disappear
            if np.random.rand() < 0.5:  # 50% chance to disappear
                self.car_in_front = 0  # No car in front
                self.car_disappear_counter = 500  # Car disappears for 500 steps

        if self.car_disappear_counter > 0:
            self.car_disappear_counter -= 1
            if self.car_disappear_counter == 0:
                self.car_in_front = 1  # Car reappears at least 30m ahead
                self.relative_distance = 30 + np.random.uniform(0, 30)

        self.relative_distance += (self.current_target_speed - self.agent_speed) * self.dt

        # Update distance to stop line if stop flag is set
        if self.stop == 1:
            self.distance_to_stop -= self.agent_speed * self.dt

        # Introduce random stopping line at least 30m ahead
        if self.time_step_counter % 300 == 0 and np.random.rand() < 0.3:
            self.stop = 1
            self.distance_to_stop = 30 + np.random.uniform(0, 30)  # Ensure stop line is at least 30m ahead

        reward = self._get_reward()

        self.last_speed = self.agent_speed
        self.episode_reward += reward

        truncated = False  # Environment does not truncate based on distance behind
        terminated = self.No_stop or self.relative_distance < 0.5  # Terminate if stopped at the line

        # Log data to wandb
        wandb.log({
            "episode": self.episode,
            "distance": self.relative_distance,
            "agent_speed": self.agent_speed,
            "target_speed": self.current_target_speed,
            "reward": reward,
            "action": action[0],
            "car_in_front": self.car_in_front,
            "stop": self.stop,
            "distance_to_stop": self.distance_to_stop
        })

        observation = np.array([self.agent_speed, self.max_speed, self.car_in_front, self.relative_distance, self.stop, self.distance_to_stop], dtype=np.float32)
        
        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            return
        
        # Initialize pygame if not done yet
        if not self.render_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Adaptive Cruise Control Simulation")
            self.clock = pygame.time.Clock()
            self.render_initialized = True

        # Clear the screen
        self.screen.fill((255, 255, 255))

        # Calculate positions for visualization
        target_x = 50
        agent_x = target_x + self.relative_distance * self.pixels_per_meter

        # Draw target car (only if it hasn't disappeared)
        if self.current_target_speed > 0:
            pygame.draw.rect(self.screen, (0, 0, 255), (target_x, self.screen_height // 2, self.car_width, self.car_height))

        # Draw agent car
        pygame.draw.rect(self.screen, (255, 0, 0), (agent_x, self.screen_height // 2, self.car_width, self.car_height))

        # Draw stop line if stop flag is set
        if self.stop == 1:
            stop_line_x = agent_x + self.distance_to_stop * self.pixels_per_meter
            pygame.draw.line(self.screen, (0, 255, 0), (stop_line_x, self.screen_height // 2 - 10), (stop_line_x, self.screen_height // 2 + 30), 5)
            stop_text = f"Stop Line: {self.distance_to_stop:.1f} m"
        else:
            stop_text = "No Stop Line"

        # Determine action type based on last action value
        action_type = "Throttle" if self.last_action > 0.5 else "Brake"

        # Draw speed, distance, action text, and stop line text
        font = pygame.font.Font(None, 36)
        distance_text = font.render(f"Distance: {self.relative_distance:.1f} m", True, (0, 0, 0))
        agent_speed_text = font.render(f"Agent Speed: {self.agent_speed:.1f} m/s", True, (0, 0, 0))
        action_text = font.render(f"Action: {action_type} ({self.last_action:.2f})", True, (0, 0, 0))  # Display actual action value
        target_speed_text = font.render(f"Lead vehicle speed: {self.current_target_speed}", True, (0, 0, 0))
        stop_line_text = font.render(stop_text, True, (0, 0, 0))

        self.screen.blit(distance_text, (10, 10))
        self.screen.blit(agent_speed_text, (10, 50))
        self.screen.blit(action_text, (10, 90))
        self.screen.blit(target_speed_text, (10, 130))
        self.screen.blit(stop_line_text, (10, 170))

        # Update display
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.render_initialized:
            pygame.quit()
            self.render_initialized = False

    def _get_reward(self) -> float:
        """
        Calculate reward based on the action and the current environment state.
        Reward based on speed, distance, and smooth driving with constraints on safe driving.
        """
        speed_limit = self.max_speed
        current_speed = self.agent_speed
        speed_margin = speed_limit * 0.15
        
        # Constants
        safe_distance_margin = 0.25
        max_safe_distance = 100

        # Default reward
        reward = 0

        # Speed Reward Calculation
        speed_reward = 0
        max_speed = speed_limit + speed_margin
        if self.car_in_front == 0 and self.stop == 0:  # No car in front
            if current_speed == 0:
                speed_reward = 0  # No reward for being stationary
            elif 0 < current_speed <= speed_limit:
                speed_reward = min(1, current_speed / speed_limit)  # Linearly scale up to the speed limit
            elif speed_limit < current_speed < max_speed:
                speed_reward = max(0, (max_speed - current_speed) / (max_speed - speed_limit))  # Linearly ramp down
            elif current_speed >= max_speed:
                speed_reward = 0  # No reward if speed exceeds or equals max_speed

        # Safe Distance Reward Calculation
        safe_distance_reward = 0
        if current_speed > 1:
            safe_distance = 2 * (current_speed / 3.6)  # Safe distance = 2 seconds of travel in m/s
        else:
            safe_distance = 4

        lower_bound = safe_distance - safe_distance_margin
        upper_bound = safe_distance + safe_distance_margin
        if self.car_in_front == 1 and self.stop == 0:
            if lower_bound <= self.relative_distance <= upper_bound:
                safe_distance_reward = 1  # Perfect safe distance
            elif self.relative_distance < lower_bound:
                safe_distance_reward = max(0, self.relative_distance / lower_bound)  # Linearly decrease to 0
            elif self.relative_distance > upper_bound:
                safe_distance_reward = max(0, 1 - (self.relative_distance - upper_bound) / (max_safe_distance - upper_bound))

        # Smoother Driving Reward Calculation
        acceleration = (self.agent_speed - self.last_speed) / self.dt
        smoothness_penalty = min(1.0, np.abs(acceleration) / 5.0)  # Normalized penalty for high acceleration/deceleration
        smooth_driving_reward = 1.0 - smoothness_penalty

        # Stop Flag Reward Calculation 
        stop_reward = 0 
        if self.stop == 1: # If stop flag is set 
            if self.distance_to_stop > 2: # Factor in the current speed to promote smooth stopping 
                speed_factor = max(0, 1 - self.agent_speed / self.max_speed) 
                distance_factor = 1 - min(1, self.distance_to_stop / 30) # Scale down as the distance to stop decreases 
                stop_reward = 0.5 * distance_factor + 0.5 * speed_factor # Combine distance and speed factors 
            elif self.distance_to_stop <= 2 and self.agent_speed < 1: 
                stop_reward = 1 # Reward for stopping at the line 
            
            # If stop flag is ignored (speed is high near stop line), restart the environment 
            if self.agent_speed > 1 and self.distance_to_stop <= 0.1: 
                self.No_stop = True

            if self.agent_speed < 0.1:
                self.stop = 0

        reward = speed_reward + safe_distance_reward + stop_reward

        reward = reward * smooth_driving_reward
        # Determine the final reward based on conditions
        """if self.car_in_front == 0 and self.stop == 0:
            reward = speed_reward * smooth_driving_reward
        elif self.car_in_front == 1 and self.stop == 0:
            reward = safe_distance_reward * smooth_driving_reward
        elif self.stop == 1:
            reward = stop_reward * smooth_driving_reward"""

        # Ensure reward is in range [0, 1]
        reward = np.clip(reward, 0, 1)

        # Logging for debugging and analysis
        wandb.log({
            "speed_reward": speed_reward,
            "safe_distance_reward": safe_distance_reward,
            "stop_reward": stop_reward,
            "smooth_driving_reward": smooth_driving_reward,
            "reward": reward,
            "speed_limit": speed_limit,
            "current_speed": current_speed,
            "acceleration": acceleration
        })

        return reward