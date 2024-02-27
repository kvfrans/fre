import numpy as np
import gym

# Here's a super simple bandit environment that follows the OpenAI Gym API.
# There is one continuous action. The observation is always zero.
# A reward of 1 is given if the action is either 0.5 or -0.5.

class BanditEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self._state = None
        self.width = 0.15

    def reset(self):
        self._state = np.zeros(1)
        return self._state

    def step(self, action):
        reward = 0
        if (np.abs(action[0] - 0.5) < self.width) or (np.abs(action[0] + 0.5) < self.width):
            reward = 1
        self.last_action = action
        return self._state, reward, True, {}

    def render(self, mode='human'):
        # Render the last action on a line. Also indicate where the reward is. Return this as a numpy array.
        img = np.ones((20, 100, 3), dtype=np.uint8) * 255
        # Render reward zones in green. 0-100 means actions between -1 and 1.
        center_low = 25
        center_high = 75
        width_int = int(self.width * 50)
        img[:, center_low-width_int:center_low+width_int, :] = [0, 255, 0]
        img[:, center_high-width_int:center_high+width_int, :] = [0, 255, 0]
        # Render the last action in red.
        action = self.last_action[0]
        action = int((action + 1) * 50)
        img[:, action:action+1, :] = [255, 0, 0]
        return img

 


    def close(self):
        pass