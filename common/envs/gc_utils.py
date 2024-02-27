from fre.common.dataset import Dataset
from flax.core.frozen_dict import FrozenDict
from flax.core import freeze
import dataclasses
import numpy as np
import jax
import ml_collections

@dataclasses.dataclass
class GCDataset:
    dataset: Dataset
    p_randomgoal: float
    p_trajgoal: float
    p_currgoal: float
    geom_sample: int
    discount: float
    terminal_key: str = 'dones_float'
    reward_scale: float = 1.0
    reward_shift: float = -1.0
    mask_terminal: int = 1

    @staticmethod
    def get_default_config():
        return ml_collections.ConfigDict({
            'p_randomgoal': 0.3,
            'p_trajgoal': 0.5,
            'p_currgoal': 0.2,
            'geom_sample': 1,
            'discount': 0.99,
            'reward_scale': 1.0,
            'reward_shift': -1.0,
            'mask_terminal': 1,
        })

    def __post_init__(self):
        self.terminal_locs, = np.nonzero(self.dataset[self.terminal_key] > 0)
        assert np.isclose(self.p_randomgoal + self.p_trajgoal + self.p_currgoal, 1.0)

    def sample_goals(self, indx, p_randomgoal=None, p_trajgoal=None, p_currgoal=None):
        if p_randomgoal is None:
            p_randomgoal = self.p_randomgoal
        if p_trajgoal is None:
            p_trajgoal = self.p_trajgoal
        if p_currgoal is None:
            p_currgoal = self.p_currgoal

        batch_size = len(indx)
        # Random goals
        goal_indx = np.random.randint(self.dataset.size, size=batch_size)
        
        # Goals from the same trajectory
        final_state_indx = self.terminal_locs[np.searchsorted(self.terminal_locs, indx)]

        distance = np.random.rand(batch_size)
        if self.geom_sample:
            us = np.random.rand(batch_size)
            middle_goal_indx = np.minimum(indx + np.ceil(np.log(1 - us) / np.log(self.discount)).astype(int), final_state_indx)
        else:
            middle_goal_indx = np.round((np.minimum(indx + 1, final_state_indx) * distance + final_state_indx * (1 - distance))).astype(int)

        goal_indx = np.where(np.random.rand(batch_size) < p_trajgoal / (1.0 - p_currgoal), middle_goal_indx, goal_indx)
        
        # Goals at the current state
        goal_indx = np.where(np.random.rand(batch_size) < p_currgoal, indx, goal_indx)

        return goal_indx

    def sample(self, batch_size: int, indx=None):
        if indx is None:
            indx = np.random.randint(self.dataset.size-1, size=batch_size)
        
        batch = self.dataset.sample(batch_size, indx)
        goal_indx = self.sample_goals(indx)

        success = (indx == goal_indx)
        batch['rewards'] = success.astype(float) * self.reward_scale + self.reward_shift
        batch['goals'] = jax.tree_map(lambda arr: arr[goal_indx], self.dataset['observations'])

        if self.mask_terminal:
            batch['masks'] = 1.0 - success.astype(float)
        else:
            batch['masks'] = np.ones(batch_size)

        return batch
    
    def sample_traj_random(self, batch_size, num_traj_states, num_random_states, num_random_states_decode):
        indx = np.random.randint(self.dataset.size-1, size=batch_size)
        batch = self.dataset.sample(batch_size, indx)
        indx_expand = np.repeat(indx, num_traj_states-1) # (batch_size * num_traj_states)
        traj_indx = self.sample_goals(indx_expand, p_randomgoal=0.0, p_trajgoal=1.0, p_currgoal=0.0)
        traj_indx = traj_indx.reshape(batch_size, num_traj_states-1) # (batch_size, num_traj_states)
        batch['traj_states'] = jax.tree_map(lambda arr: arr[traj_indx], self.dataset['observations'])
        batch['traj_states'] = np.concatenate([batch['observations'][:,None,:], batch['traj_states']], axis=1)

        rand_indx = np.random.randint(self.dataset.size-1, size=batch_size * num_random_states)
        rand_indx = rand_indx.reshape(batch_size, num_random_states)
        batch['random_states'] = jax.tree_map(lambda arr: arr[rand_indx], self.dataset['observations'])

        rand_indx_decode = np.random.randint(self.dataset.size-1, size=batch_size * num_random_states_decode)
        rand_indx_decode = rand_indx_decode.reshape(batch_size, num_random_states_decode)
        batch['random_states_decode'] = jax.tree_map(lambda arr: arr[rand_indx_decode], self.dataset['observations'])
        return batch

def flatten_obgoal(obgoal):
    return np.concatenate([obgoal['observation'], obgoal['goal']], axis=-1)