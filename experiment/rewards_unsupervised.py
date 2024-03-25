import numpy as np
import tqdm
import opensimplex
import jax
import jax.numpy as jnp
from functools import partial

class RewardFunction():
    # Given a batch of trajectory states and random states, generate a reward function.
    # Return the labelled state-reward pairs. (batch_size, num_pairs, obs_dim + 1)
    def generate_params_and_pairs(self, traj_states, random_states):
        raise NotImplementedError
    
    # Given a batch of states and a batch of parameters, compute the reward.
    def compute_reward(self, states, params):
        raise NotImplementedError
    
class GoalReachingRewardFunction(RewardFunction):
    def __init__(self):
        self.p_current = 0.2
        self.p_trajectory = 0.5
        self.p_random = 0.3
    
    # TODO: If this is slow, we can try and JIT it.
    # Select a random goal from the provided states.
    def generate_params_and_pairs(self, traj_states, random_states, random_states_decode):
        all_states = np.concatenate([traj_states, random_states], axis=1)
        batch_size = all_states.shape[0]
        p_trajectory_normalized = self.p_trajectory / traj_states.shape[1]
        p_random_normalized = self.p_random / random_states.shape[1]
        probabilities = [self.p_current] + [p_trajectory_normalized] * (traj_states.shape[1]-1) \
            + [p_random_normalized] * random_states.shape[1]
        probabilities = np.array(probabilities) / np.sum(probabilities)
        selected_goal_idx = np.random.choice(len(probabilities), size=(batch_size,), p=probabilities)
        selected_goal = all_states[np.arange(batch_size), selected_goal_idx]

        params = selected_goal # (batch_size, obs_dim)
        encode_pairs = np.concatenate([traj_states, random_states], axis=1)
        encode_rewards = self.compute_reward(encode_pairs, params[:, None, :])[:, :, None]
        encode_pairs = np.concatenate([encode_pairs, encode_rewards], axis=-1)

        decode_pairs = random_states_decode
        decode_pairs[:, 0] = params # Decode the goal state too.
        decode_rewards = self.compute_reward(decode_pairs, params[:, None, :])[:, :, None]
        decode_pairs = np.concatenate([random_states_decode, decode_rewards], axis=-1)

        rewards = encode_pairs[:, 0, -1] # (batch_size,)
        masks = -rewards # If (rew=-1, mask=1), else (rew=0, mask=0)

        return params, encode_pairs, decode_pairs, rewards, masks
    
    def compute_reward(self, states, params, delta=False):
        assert len(states.shape) == len(params.shape), states.shape # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)
        if states.shape[-1] == 29: # AntMaze
            if delta:
                dists = np.linalg.norm(states - params, axis=-1)
                is_goal = (dists < 0.1)
            else:
                dists = np.linalg.norm(states[..., :2] - params[..., :2], axis=-1)
                is_goal = (dists < 2)
            return -1 + is_goal.astype(float) # (batch_size,)
        elif states.shape[-1] == 18: # Cheetah
            std = np.array([[0.4407440506721877, 10.070289916801876, 0.5172332956856273, 0.5601041145815341, 0.518947027289748, 0.3204431592542281, 0.5501848643154092, 0.3856393812067661, 1.9882502334402663, 1.6377168569884073, 4.308505013609855, 12.144181770553105, 13.537567521831702, 16.88983033626308, 7.715009572436841, 14.345667964212357, 10.6904255152284, 100]])
            assert len(states.shape) == len(params.shape), states.shape # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)
            # if len(states.shape) == 3:
            #     breakpoint()
            dists_per_dim = states - params
            dists_per_dim = dists_per_dim / std
            dists = np.linalg.norm(dists_per_dim, axis=-1) / states.shape[-1]
            is_goal = (dists < 0.08)
            # print(dists_per_dim)
            # print(dists, is_goal)
            return -1 + is_goal.astype(float) # (batch_size,)
        elif states.shape[-1] == 27: # Walker
            std = np.array([[0.7212967364054736, 0.6775020895964047, 0.7638155887842976, 0.6395721376821286, 0.6849394775886244, 0.7078581708129903, 0.7113168519036742, 0.6753408522523937, 0.6818095329625652, 0.7133958718133511, 0.65227578338642, 0.757622576816855, 0.7311826446274479, 0.6745824928740024, 0.36822491550384456, 2.1134839667805805, 1.813353841099317, 10.594648894374815, 17.41041469033713, 17.836743227082106, 22.399097178637533, 16.1492222730888, 15.693574546557201, 18.539929326905067, 100, 100, 100]])
            assert len(states.shape) == len(params.shape), states.shape # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)
            dists_per_dim = states - params
            dists_per_dim = dists_per_dim / std
            dists = np.linalg.norm(dists_per_dim, axis=-1) / states.shape[-1]
            is_goal = (dists < 0.2)
            return -1 + is_goal.astype(float) # e6yfwsc ebnev (batch_size,)
        elif states.shape[-1] == 30: # Kitchen
            dists_per_dim = states - params
            dists_per_dim = dists_per_dim
            dists = np.linalg.norm(dists_per_dim, axis=-1) / states.shape[-1]
            is_goal = (dists < 1e-6)
            return -1 + is_goal.astype(float)
        else:
            raise NotImplementedError

    def make_encoder_pairs_testing(self, params, random_states):
        assert len(params.shape) == 2, params.shape # (batch_size, 2)
        assert len(random_states.shape) == 3, random_states.shape # (batch_size, num_pairs, obs_dim)

        if random_states.shape[-1] == 29: # AntMaze
            random_states[:, 0, :2] = params[:, :2] # Make sure to include the goal.
        else:
            random_states[:, 0] = params
        reward_pair_rews = self.compute_reward(random_states, params[:, None, :])[..., None]
        reward_pairs = np.concatenate([random_states, reward_pair_rews], axis=-1)
        return reward_pairs # (batch_size, reward_pairs, obs_dim + 1)
    
class LinearRewardFunction(RewardFunction):
    def __init__(self):
        pass
    
    # Randomly generate a linear weighting over state features.
    def generate_params_and_pairs(self, traj_states, random_states, random_states_decode):
        assert len(traj_states.shape) == 3, traj_states.shape # (batch_size, traj_len, obs_dim)
        batch_size = traj_states.shape[0]
        state_len = traj_states.shape[-1]

        params = np.random.uniform(-1, 1, size=(batch_size, state_len)) # Uniform weighting.
        random_mask = np.random.uniform(size=(batch_size,state_len)) < 0.9
        if state_len == 29:
            random_mask[:, :2] = True # Zero out the XY position for antmaze.
        random_mask_positive = np.random.randint(2, state_len, size=(batch_size))
        random_mask[np.arange(batch_size), random_mask_positive] = False # Force at least one positive weight.
        params[random_mask] = 0 # Zero out some of the weights.
        # if state_len == 29:
        #     params = params / np.linalg.norm(params, axis=-1, keepdims=True) # Normalize XY

        # Remove auxilliary features during training.
        if state_len == 27:
            params[:, -3:] = 0
        if state_len == 18:
            params[:, -1:] = 0

        clip_bit = np.random.uniform(size=(batch_size,)) < 0.5
        params = np.concatenate([params, clip_bit[:, None]], axis=-1) # (batch_size, obs_dim + 1)

        encode_pairs = np.concatenate([traj_states, random_states], axis=1)
        encode_rewards = self.compute_reward(encode_pairs, params[:, None, :])[:, :, None]
        encode_pairs = np.concatenate([encode_pairs, encode_rewards], axis=-1)

        decode_pairs = random_states_decode
        decode_rewards = self.compute_reward(decode_pairs, params[:, None, :])[:, :, None]
        decode_pairs = np.concatenate([random_states_decode, decode_rewards], axis=-1)

        rewards = encode_pairs[:, 0, -1] # (batch_size,)
        masks = np.ones_like(rewards) # (batch_size,)

        return params, encode_pairs, decode_pairs, rewards, masks
    
    def compute_reward(self, states, params):
        params_raw = params[..., :-1]
        assert len(states.shape) == len(params_raw.shape), states.shape # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)
        r = np.sum(states * params_raw, axis=-1) # (batch_size,)
        r = np.where(params[..., -1] > 0, np.clip(r, 0, 1), np.clip(r, -1, 1))
        return r
    
    def make_encoder_pairs_testing(self, params, random_states):
        assert len(params.shape) == 2, params.shape # (batch_size, 2)
        assert len(random_states.shape) == 3, random_states.shape # (batch_size, num_pairs, obs_dim)

        reward_pair_rews = self.compute_reward(random_states, params[:, None, :])[..., None]
        reward_pairs = np.concatenate([random_states, reward_pair_rews], axis=-1)
        return reward_pairs # (batch_size, reward_pairs, obs_dim + 1)
    
class RandomRewardFunction(RewardFunction):
    def __init__(self, num_simplex, obs_len=29):
        # Pre-compute parameter matrices.
        print("Generating parameter matrices...")
        self.simplex_size = num_simplex
        np_random = np.random.RandomState(0)
        self.param_w1 = np_random.normal(size=(self.simplex_size, obs_len, 32)) * np.sqrt(1/32)
        self.param_b1 = np_random.normal(size=(self.simplex_size, 1, 32)) * np.sqrt(16)
        self.param_w2 = np_random.normal(size=(self.simplex_size, 32, 1)) * np.sqrt(1/16)

        # Remove auxilliary features during training.
        if obs_len == 27:
            self.param_w1[:, -3:] = 0
        if obs_len == 18:
            self.param_w1[:, -1:] = 0
    
    # Random neural network.
    def generate_params_and_pairs(self, traj_states, random_states, random_states_decode):
        batch_size = traj_states.shape[0]
        params = np.random.randint(self.simplex_size, size=(batch_size, 1)) # (batch_size, 1)

        encode_pairs = np.concatenate([traj_states, random_states], axis=1)
        encode_rewards = self.compute_reward(encode_pairs, params[:, None, :])[:, :, None]
        encode_pairs = np.concatenate([encode_pairs, encode_rewards], axis=-1)

        decode_pairs = random_states_decode
        decode_rewards = self.compute_reward(decode_pairs, params[:, None, :])[:, :, None]
        decode_pairs = np.concatenate([random_states_decode, decode_rewards], axis=-1)

        rewards = encode_pairs[:, 0, -1] # (batch_size,)
        masks = np.ones_like(rewards) # (batch_size,)

        return params, encode_pairs, decode_pairs, rewards, masks
    
    def compute_reward(self, states, params):
        assert len(states.shape) == len(params.shape), states.shape # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)

        param_id = params[..., 0].astype(int)
        param1_w = self.param_w1[param_id]
        param1_b = self.param_b1[param_id]
        param2_w = self.param_w2[param_id]

        obs = states
        x = np.expand_dims(obs, -2) # [batch, (pairs), 1, features_in]
        x = np.matmul(x, param1_w) # [batch, (pairs), 1, features_out]
        x = x + param1_b
        x = np.tanh(x)
        x = np.matmul(x, param2_w) # [batch, (pairs), 1, 1]
        x = x.squeeze(-1).squeeze(-1) # [batch, (pairs)]
        x = np.clip(x, -1, 1)
        return x
    
    def make_encoder_pairs_testing(self, params, random_states):
        assert len(params.shape) == 2, params.shape # (batch_size, 2)
        assert len(random_states.shape) == 3, random_states.shape # (batch_size, num_pairs, obs_dim)
        batch_size = random_states.shape[0]

        reward_pair_rews = self.compute_reward(random_states, params[:, None, :])[..., None]
        reward_pairs = np.concatenate([random_states, reward_pair_rews], axis=-1)
        return reward_pairs # (batch_size, reward_pairs, obs_dim + 1)
