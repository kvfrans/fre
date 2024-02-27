import d4rl
import d4rl.gym_mujoco
import gym
import numpy as np
from jax import tree_util


import fre.common.envs.d4rl.d4rl_ant as d4rl_ant
from fre.common.dataset import Dataset


# Note on AntMaze. Reward = 1 at the goal, and Terminal = 1 at the goal.
# Masks = Does the episode end due to final state?
# Dones_float = Does the episode end due to time limit? OR does the episode end due to final state?
def get_dataset(env: gym.Env, env_name: str, clip_to_eps: bool = True,
                eps: float = 1e-5, dataset=None, filter_terminals=False, obs_dtype=np.float32):
    if dataset is None:
        dataset = d4rl.qlearning_dataset(env)

    if clip_to_eps:
        lim = 1 - eps
        dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

    # Mask everything that is marked as a terminal state.
    # For AntMaze, this should mask the end of each trajectory.
    masks = 1.0 - dataset['terminals']

    # In the AntMaze data, terminal is 1 when at the goal. But the episode doesn't end. 
    # This just ensures that we treat AntMaze trajectories as non-ending.
    if "antmaze" in env_name or "maze2d" in env_name:
        dataset['terminals'] = np.zeros_like(dataset['terminals'])

    # if 'antmaze' in env_name:
    #     print("Discretizing AntMaze observations.")
    #     print("Raw observations looks like", dataset['observations'].shape[1:])
    #     dataset['observations'] = d4rl_ant.discretize_obs(dataset['observations'])
    #     dataset['next_observations'] = d4rl_ant.discretize_obs(dataset['next_observations'])
    #     print("Discretized observations looks like", dataset['observations'].shape[1:])

    # Compute dones if terminal OR orbservation jumps.
    dones_float = np.zeros_like(dataset['rewards'])

    imputed_next_observations = np.roll(dataset['observations'], -1, axis=0)
    same_obs = np.all(np.isclose(imputed_next_observations, dataset['next_observations'], atol=1e-5), axis=-1)
    dones_float = 1.0 - same_obs.astype(np.float32)
    dones_float += dataset['terminals']
    dones_float[-1] = 1.0
    dones_float = np.clip(dones_float, 0.0, 1.0)

    observations = dataset['observations'].astype(obs_dtype)
    next_observations = dataset['next_observations'].astype(obs_dtype)

    return Dataset.create(
        observations=observations,
        actions=dataset['actions'].astype(np.float32),
        rewards=dataset['rewards'].astype(np.float32),
        masks=masks.astype(np.float32),
        dones_float=dones_float.astype(np.float32),
        next_observations=next_observations,
    )

def get_normalization(dataset):
    returns = []
    ret = 0
    for r, term in zip(dataset['rewards'], dataset['dones_float']):
        ret += r
        if term:
            returns.append(ret)
            ret = 0
    return (max(returns) - min(returns)) / 1000

def normalize_dataset(env_name, dataset):
    print("Normalizing", env_name)
    if 'antmaze' in env_name or 'maze2d' in env_name:
        return dataset.copy({'rewards': dataset['rewards']- 1.0})
    else:
        normalizing_factor = get_normalization(dataset)
        print(f'Normalizing factor: {normalizing_factor}')
        dataset = dataset.copy({'rewards': dataset['rewards'] / normalizing_factor})
        return dataset
    
# Flattens environment with a dictionary of observation,goal to a single concatenated observation.
class GoalReachingFlat(gym.Wrapper):
    """A wrapper that maps actions from [-1,1] to [low, hgih]."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_space['observation'].shape[0] + self.observation_space['goal'].shape[0],), dtype=np.float32)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob_flat = np.concatenate([ob['observation'], ob['goal']])
        return ob_flat, reward, done, info

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        ob_flat = np.concatenate([ob['observation'], ob['goal']])
        return ob_flat
    
def parse_trajectories(dataset):
    trajectory_ids = np.where(dataset['dones_float'] == 1)[0] + 1
    trajectory_ids = np.concatenate([[0], trajectory_ids])
    num_trajectories = trajectory_ids.shape[0] - 1
    print("There are {} trajectories. Some traj lens are {}".format(num_trajectories, [trajectory_ids[i + 1] - trajectory_ids[i] for i in range(min(5, num_trajectories))]))
    trajectories = []
    for i in range(len(trajectory_ids) - 1):
        trajectories.append(tree_util.tree_map(lambda arr: arr[trajectory_ids[i]:trajectory_ids[i + 1]], dataset._dict))
    return trajectories

class KitchenRenderWrapper(gym.Wrapper):
    def render(self, *args, **kwargs):
        from dm_control.mujoco import engine
        camera = engine.MovableCamera(self.sim, 1920, 2560)
        camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
        img = camera.render()
        return img
