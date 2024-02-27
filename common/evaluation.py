###############################
#
#  Tools for evaluating policies in environments.
#
###############################


from typing import Dict
import jax
import gym
import numpy as np
from collections import defaultdict
import time
import wandb


def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(policy_fn, env: gym.Env, num_episodes: int, record_video : bool = False, 
             return_trajectories=False, clip_return_at_goal=False, binary_return=False, use_discrete_xy=False, clip_margin=0):
    print("Clip return at goal is", clip_return_at_goal)
    stats = defaultdict(list)
    frames = []
    trajectories = []
    for i in range(num_episodes):
        now = time.time()
        trajectory = defaultdict(list)
        ob_list = []
        ac_list = []
        observation, done = env.reset(), False
        ob_list.append(observation)
        while not done:
            if use_discrete_xy:
                import fre.common.envs.d4rl.d4rl_ant as d4rl_ant
                ob_input = d4rl_ant.discretize_obs(observation)
            else:
                ob_input = observation
            action = policy_fn(ob_input)
            action = np.array(action)
            next_observation, r, done, info = env.step(action)
            add_to(stats, flatten(info))

            if type(observation) is dict:
                obs_pure = observation['observation']
                next_obs_pure = next_observation['observation']
            else:
                obs_pure = observation
                next_obs_pure = next_observation
            transition = dict(
                observation=obs_pure,
                next_observation=next_obs_pure,
                action=action,
                reward=r,
                done=done,
                info=info,
            )
            observation = next_observation
            ob_list.append(observation)
            ac_list.append(action)
            add_to(trajectory, transition)

            if i <= 3 and record_video:
                frames.append(env.render(mode="rgb_array"))
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)
        print("Finished Episode", i, "in", time.time() - now, "seconds")

    if clip_return_at_goal and 'episode.return' in stats:
        print("Episode finished. Return is {}. Length is {}.".format(stats['episode.return'], stats['episode.length']))
        stats['episode.return'] = np.clip(np.array(stats['episode.length']) + np.array(stats['episode.return']) - clip_margin, 0, 1) # Goal is a binary indicator.
        print("Clipped return is {}.".format(stats['episode.return']))
    elif binary_return and 'episode.return' in stats:
        # Assume that the reward is either 0 or 1 at each timestep.
        print("Episode finished. Return is {}. Length is {}.".format(stats['episode.return'], stats['episode.length']))
        stats['episode.return'] = np.clip(np.array(stats['episode.return']), 0, 1)
        print("Clipped return is {}.".format(stats['episode.return']))

    if 'episode.return' in stats:
        print("Episode finished. Return is {}. Length is {}.".format(stats['episode.return'], stats['episode.length']))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if record_video:
        stacked = np.stack(frames)
        stacked = stacked.transpose(0, 3, 1, 2)
        while stacked.shape[2] > 160:
            stacked = stacked[:, :, ::2, ::2]
        stats['video'] = wandb.Video(stacked, fps=60)

    if return_trajectories:
        return stats, trajectories
    else:
        return stats