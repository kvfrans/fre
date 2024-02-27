###############################
#
#  Wrappers on top of gym environments
#
###############################

from typing import Dict
import gym
import numpy as np
import time

class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        self._reset_stats()
        return self.env.reset(**kwargs)

class RewardOverride(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.reward_fn = None

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)
        
        if self.env.observation_space.shape[0] == 24:
            horizontal_velocity = self.env.physics.horizontal_velocity()
            torso_upright = self.env.physics.torso_upright()
            torso_height = self.env.physics.torso_height()
            aux = np.array([horizontal_velocity, torso_upright, torso_height])
            observation_aux = np.concatenate([observation, aux])
            reward = self.reward_fn(observation_aux)
        elif self.env.observation_space.shape[0] == 17:
            horizontal_velocity = self.env.physics.speed()
            aux = np.array([horizontal_velocity])
            observation_aux = np.concatenate([observation, aux])
            reward = self.reward_fn(observation_aux)
        else:
            reward = self.reward_fn(observation)
        return observation, reward, done, info

    def reset(self, **kwargs) -> np.ndarray:
        return self.env.reset(**kwargs)

class TruncateObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, truncate_size: int):
        super().__init__(env)
        self.truncate_size = truncate_size

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation[:self.truncate_size]

class GoalWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.custom_goal = None

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if self.custom_goal is not None:
            return np.concatenate([observation, self.custom_goal])
        else:
            return observation
