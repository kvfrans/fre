import os
from pathlib import Path
import glob
import tqdm
import numpy as np
from collections import defaultdict

# from fre.common.envs.d4rl import d4rl_utils
from fre.common.dataset import Dataset

# get path relative to 'fre' package
data_path = os.path.dirname(os.path.abspath(__file__))
data_path = Path(data_path).parents[2]
data_path = os.path.join(data_path, 'data/exorl/')
print("Path to exorl data is", data_path)

def get_dataset(env, env_name, method='rnd', dmc_dataset_size=10000000, use_task_reward=True):

    # dmc_dataset_size /= 10
    # print("WARNING: Only using 10 percent of exorl data.")

    domain_name, task_name = env_name.split('_', 1)

    path = os.path.join(data_path, domain_name, method)
    if not os.path.exists(path):
        print("Downloading exorl data.")
        os.makedirs(path)
        url = "https://dl.fbaipublicfiles.com/exorl/" + domain_name + "/" + method + ".zip"
        print("Downloading from", url)
        os.system("wget " + url + " -P " + path)
        os.system("unzip " + path + "/" + method + ".zip -d " + path)

    # process data into Dataset object.
    path = os.path.join(data_path, domain_name, method, 'buffer')
    npzs = sorted(glob.glob(f'{path}/*.npz'))
    dataset_npy = os.path.join(data_path, domain_name, method, task_name + '.npy')
    if os.path.exists(dataset_npy):
        dataset = np.load(dataset_npy, allow_pickle=True).item()
    else:
        print("Calculating exorl rewards.")
        dataset = defaultdict(list)
        num_steps = 0
        for i, npz in tqdm.tqdm(enumerate(npzs)):
            traj_data = dict(np.load(npz))
            dataset['observations'].append(traj_data['observation'][:-1, :])
            dataset['next_observations'].append(traj_data['observation'][1:, :])
            dataset['actions'].append(traj_data['action'][1:, :])
            dataset['physics'].append(traj_data['physics'][1:, :])  # Note that this corresponds to next_observations (i.e., r(s, a, s') = r(s') -- following the original DMC rewards)

            if use_task_reward:
                # TODO: make this faster and sanity check it
                rewards = []
                reward_spec = env.reward_spec()
                states = traj_data['physics']
                for j in range(states.shape[0]):
                    with env.physics.reset_context():
                        env.physics.set_state(states[j])
                    reward = env.task.get_reward(env.physics)
                    reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
                    rewards.append(reward)
                traj_data['reward'] = np.array(rewards, dtype=reward_spec.dtype)
                dataset['rewards'].append(traj_data['reward'][1:])
            else:
                dataset['rewards'].append(traj_data['reward'][1:, 0])

            terminals = np.full((len(traj_data['observation']) - 1,), False)
            dataset['terminals'].append(terminals)
            num_steps += len(traj_data['observation']) - 1
            if num_steps >= dmc_dataset_size:
                break
        print("Loaded {} steps".format(num_steps))
        for k, v in dataset.items():
            dataset[k] = np.concatenate(v, axis=0)
        np.save(dataset_npy, dataset)

    

    # Processing
    masks = 1.0 - dataset['terminals']
    dones_float = dataset['terminals']

    return Dataset.create(
        observations=dataset['observations'],
        actions=dataset['actions'],
        rewards=dataset['rewards'],
        masks=masks,
        dones_float=dones_float,
        next_observations=dataset['next_observations'],
    )