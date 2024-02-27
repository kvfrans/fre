###############################
#
#   Helper that initializes environments with the proper imports.
#   Returns an environment that is:
#   - Action normalized.
#   - Video rendering works.
#   - Episode monitor attached.
#
###############################

import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import gym
import numpy as np
import functools as ft
from fre.common.train_state import NormalizeActionWrapper
from fre.common.envs.wrappers import EpisodeMonitor


# Supported envs:
env_list = [
    # From Gym
    'HalfCheetah-v2',
    'Hopper-v2',
    'Walker2d-v2',
    'Pendulum-v1',
    'CartPole-v1',
    'Acrobot_v1',
    'MountainCar-v0',
    'MountainCarContinuous-v0',
    # From DMC
    'pendulum_swingup',
    'acrobot_swingup',
    'acrobot_swingup_sparse',
    'cartpole_swingup', # has exorl dataset.
    'cartpole_swingup_sparse',
    'pointmass_easy',
    'reacher_easy',
    'reacher_hard',
    'cheetah_run',  # has exorl dataset.
    'hopper_hop',
    'walker_stand', # has exorl dataset.
    'walker_walk', # has exorl dataset.
    'walker_run', # has exorl dataset.
    'quadruped_walk', # has exorl dataset.
    'quadruped_run', # has exorl dataset.
    'humanoid_stand',
    'humanoid_run',
    'jaco_reach_top_left', # has exorl dataset.
    'jaco_reach_bottom_right', # has exorl dataset.
    # TODO: Atari games
    # Offline D4RL envs
    'antmaze-large-diverse-v2', # Start in the corner, goal is in the top corner.
    'gc-antmaze-large-diverse-v2', # Start in the corner, goal is in the top corner.
    'center-antmaze-large-diverse-v2', # Start in the center, goal is UNDEFINED (this is for RARE rewards).
    'maze2d-large-v1',
    'gc-maze2d-large-v1',
    'center-maze2d-large-v1',
    # D4RL mujoco
    'halfcheetah-expert-v2',
    'walker2d-expert-v2',
    'hopper-expert-v2',
    'kitchen-complete-v0' # broken
    'kitchen-mixed-v0' # broken
]

# Making an environment.
def make_env(env_name, **kwargs):
    if 'exorl' in env_name:
        import os
        os.environ['DISPLAY'] = ':0'
        import fre.common.envs.exorl.dmc as dmc
        _, env_name, task_name = env_name.split('_', 2)
        def make_env(env_name, task_name):
            env = dmc.make(f'{env_name}_{task_name}', obs_type='states', frame_stack=1, action_repeat=1, seed=0)
            env = dmc.DMCWrapper(env, 0)
            return env
        env = make_env(env_name, task_name)
        env.reset()
    elif '_' in env_name: # DMC Control
        import fre.common.envs.dmc as dmc2gym
        import os
        os.environ['DISPLAY'] = ':0'
        suite, task = env_name.split('_', 1)
        print(suite, task)
        if suite == 'pointmass':
            suite = 'point_mass'
        frame_skip = kwargs['frame_skip'] if 'frame_skip' in kwargs else 2
        visualize_reward = kwargs['visualize_reward'] if 'visualize_reward' in kwargs else False
        env = dmc2gym.make(
            domain_name=suite,
            task_name=task, seed=1,
            frame_skip=frame_skip,
            visualize_reward=visualize_reward)
        env = NormalizeActionWrapper(env)
    elif 'antmaze' in env_name:
        from fre.common.envs.d4rl.d4rl_ant import CenteredMaze, GoalReachingMaze, MazeWrapper
        if 'gc-antmaze' in env_name:
            env = GoalReachingMaze('antmaze-large-diverse-v2')
        elif 'center-antmaze' in env_name:
            env = CenteredMaze('antmaze-large-diverse-v2')
        else:
            env = MazeWrapper('antmaze-large-diverse-v2')
    elif 'maze2d' in env_name:
        from fre.common.envs.d4rl.d4rl_ant import CenteredMaze, GoalReachingMaze, MazeWrapper
        if 'gc-maze2d' in env_name:
            env = GoalReachingMaze('maze2d-large-v1')
        elif 'center-maze2d' in env_name:
            env = CenteredMaze('maze2d-large-v1')
        else:
            env = CenteredMaze('maze2d-large-v1', start_loc='original')
    elif 'halfcheetah-' in env_name or 'walker2d-' in env_name or 'hopper-' in env_name: # D4RL Mujoco
        import d4rl
        import d4rl.gym_mujoco
        env = gym.make(env_name)
    elif 'kitchen' in env_name: # This doesn't work yet.
        import os
        os.environ['DISPLAY'] = ':0'
        from fre.common.envs.d4rl.d4rl_utils import KitchenRenderWrapper
        env = KitchenRenderWrapper(gym.make(env_name))
    elif 'bandit' in env_name:
        from fre.common.envs.bandit.bandit import BanditEnv
        env = BanditEnv()
    else:
        env = gym.make(env_name)
    env = EpisodeMonitor(env)
    return env

# For getting offline data.
def get_dataset(env, env_name, **kwargs):
    if 'exorl' in env_name:
        from fre.common.envs.exorl.exorl_utils import get_dataset
        env_name_short = env_name.split('_', 1)[1]
        return get_dataset(env, env_name_short, **kwargs)
    elif 'ant' in env_name or 'maze2d' in env_name or 'kitchen' in env_name or 'halfcheetah' in env_name or 'walker2d' in env_name or 'hopper' in env_name:
        from fre.common.envs.d4rl.d4rl_utils import get_dataset, normalize_dataset
        dataset = get_dataset(env, env_name, **kwargs)
        dataset = normalize_dataset(env_name, dataset)
        return dataset
    elif 'cartpole' in env_name or 'cheetah' in env_name or 'jaco' in env_name or 'quadruped' in env_name or 'walker' in env_name:
        from fre.common.envs.exorl.exorl_utils import get_dataset
        return get_dataset(env, env_name, **kwargs)
    
def make_vec_env(env_name, num_envs, **kwargs):
    from gym.vector import SyncVectorEnv
    envs = [lambda : make_env(env_name, **kwargs) for _ in range(num_envs)]
    env = SyncVectorEnv(envs)
    return env