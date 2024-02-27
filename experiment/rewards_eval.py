import numpy as np
import tqdm
import opensimplex
import jax
import jax.numpy as jnp
from functools import partial

from fre.experiment.rewards_unsupervised import RewardFunction


class VelocityRewardFunction(RewardFunction):
    def __init__(self):
        pass
    
    # Select an XY velocity from a future state in the trajectory.
    def generate_params_and_pairs(self, traj_states, random_states, random_states_decode):
        batch_size = traj_states.shape[0]
        selected_traj_state_idx = np.random.randint(traj_states.shape[1], size=(batch_size,))
        selected_traj_state = traj_states[np.arange(batch_size), selected_traj_state_idx] # (batch_size, obs_dim)
        params = selected_traj_state[:, 15:17] # (batch_size, 2)
        params[:batch_size//4] = np.random.uniform(-1, 1, size=(batch_size//4, 2)) # Randomize 25% of the time.
        params = params / np.linalg.norm(params, axis=-1, keepdims=True) # Normalize XY

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
        xy_vels = states[..., 15:17] * 0.33820298
        return np.sum(xy_vels * params, axis=-1) # (batch_size,)
    
    def make_encoder_pairs_testing(self, params, random_states):
        assert len(params.shape) == 2, params.shape # (batch_size, 2)
        assert len(random_states.shape) == 3, random_states.shape # (batch_size, num_pairs, obs_dim)

        reward_pair_rews = self.compute_reward(random_states, params[:, None, :])[..., None]
        reward_pairs = np.concatenate([random_states, reward_pair_rews], axis=-1)
        return reward_pairs # (batch_size, reward_pairs, obs_dim + 1)

class TestRewMatrix(RewardFunction):
    def __init__(self):
        self.pos = np.zeros((36, 24))
        self.xvel = np.zeros((36, 24))
        self.yvel = np.zeros((36, 24))

    def generate_params_and_pairs(self, traj_states, random_states, random_states_decode):
        batch_size = traj_states.shape[0]
        params = np.zeros((batch_size, 1)) # (batch_size, 1)

        encode_pairs = np.concatenate([traj_states, random_states], axis=1)
        encode_rewards = self.compute_reward(encode_pairs, params[:, None, :])[:, :, None]
        encode_pairs = np.concatenate([encode_pairs, encode_rewards], axis=-1)

        decode_pairs = random_states_decode
        decode_rewards = self.compute_reward(decode_pairs, params[:, None, :])[:, :, None]
        decode_pairs = np.concatenate([random_states_decode, decode_rewards], axis=-1)

        rewards = encode_pairs[:, 0, -1] # (batch_size,)
        masks = np.ones_like(rewards) # (batch_size,)

        return params, encode_pairs, decode_pairs, rewards, masks
    
    def compute_reward(self, s, params):
        rews = np.zeros_like(s[..., 0]) # (batch, examples)
        # XY Vel Reward
        xy_vels = s[..., 15:17] * 0.33820298
        
        x = s[..., 0].astype(int).clip(0, 35)
        y = s[..., 1].astype(int).clip(0, 23)
        simplex = self.pos[x, y]
        simplex_xvel = self.xvel[x, y]
        simplex_yvel = self.yvel[x, y]
        rews = (simplex > 0.3).astype(float) * 0.5
        rews += xy_vels[...,0] * simplex_xvel + xy_vels[...,1] * simplex_yvel

        return rews
    
    def make_encoder_pairs_testing(self, params, random_states):
        assert len(params.shape) == 2, params.shape # (batch_size, 2)
        assert len(random_states.shape) == 3, random_states.shape # (batch_size, num_pairs, obs_dim)
        batch_size = random_states.shape[0]

        # TODO: Be smarter about the states to use here.

        reward_pair_rews = self.compute_reward(random_states, params[:, None, :])[..., None]
        reward_pairs = np.concatenate([random_states, reward_pair_rews], axis=-1)
        return reward_pairs # (batch_size, reward_pairs, obs_dim + 1)
    
class SimplexRewardFunction(RewardFunction):
    def __init__(self, num_simplex):
        self.simplex_size = num_simplex
        self.simplex_seeds_pos = np.zeros((self.simplex_size, 36, 24))
        self.simplex_seeds_xvel = np.zeros((self.simplex_size, 36, 24))
        self.simplex_seeds_yvel = np.zeros((self.simplex_size, 36, 24))
        self.simplex_best_xy = np.zeros((self.simplex_size, 10, 2))
        print("Generating simplex seeds")
        xi = np.arange(36)
        yi = np.arange(24)
        for r in tqdm.tqdm(range(self.simplex_size)):
            opensimplex.seed(r)
            self.simplex_seeds_pos[r] = opensimplex.noise2array(x=xi/20.0, y=yi/20.0).T
            opensimplex.seed(r + self.simplex_size)
            self.simplex_seeds_xvel[r] = opensimplex.noise2array(x=xi/20.0, y=yi/20.0).T
            opensimplex.seed(r + self.simplex_size * 2)
            self.simplex_seeds_yvel[r] = opensimplex.noise2array(x=xi/20.0, y=yi/20.0).T

            best_topn = np.argpartition(self.simplex_seeds_pos[r].flatten(), -10)[-10:] # (10,)
            best_xy = np.array(np.unravel_index(best_topn, self.simplex_seeds_pos[r].shape)).T # (10, 2)
            self.simplex_best_xy[r] = best_xy
        self.simplex_seeds_xvel[np.abs(self.simplex_seeds_xvel) < 0.5] = 0
        self.simplex_seeds_yvel[np.abs(self.simplex_seeds_yvel) < 0.5] = 0
    
    # Select an XY velocity from a future state in the trajectory.
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
        
        simplex_id = params[..., 0].astype(int)
        x = states[..., 0].astype(int).clip(0, 35)
        y = states[..., 1].astype(int).clip(0, 23)
        simplex = self.simplex_seeds_pos[simplex_id, x, y]
        simplex_xvel = self.simplex_seeds_xvel[simplex_id, x, y]
        simplex_yvel = self.simplex_seeds_yvel[simplex_id, x, y]
        rews = -1 + (simplex > 0.3).astype(float) * 0.5
        xy_vels = states[..., 15:17] * 0.33820298
        rews += xy_vels[...,0] * simplex_xvel + xy_vels[...,1] * simplex_yvel
        return rews # (batch_size,)
    
    def make_encoder_pairs_testing(self, params, random_states):
        assert len(params.shape) == 2, params.shape # (batch_size, 2)
        assert len(random_states.shape) == 3, random_states.shape # (batch_size, num_pairs, obs_dim)
        batch_size = random_states.shape[0]

        # For simplex rewards, make sure to include the top 4 best points.
        simplex_id = params[..., 0].astype(int)
        random_best_4 = np.random.randint(0, 10, size=(batch_size, 4))
        random_states[:, :4, :2] = self.simplex_best_xy[simplex_id[:, None], random_best_4, :]

        reward_pair_rews = self.compute_reward(random_states, params[:, None, :])[..., None]
        reward_pairs = np.concatenate([random_states, reward_pair_rews], axis=-1)
        return reward_pairs # (batch_size, reward_pairs, obs_dim + 1)

class TestRewMatrixEdges(TestRewMatrix):
    def __init__(self):
        super().__init__()
        self.pos[:3, :] = 1
        self.pos[-3:, :] = 1
        self.pos[:, :3] = 1
        self.pos[:, -3:] = 1

class TestRewLoop(TestRewMatrix):
    def __init__(self):
        super().__init__()
        self.pos[22:33, 14:18] = 1
        self.xvel[22:33, 14:18] = -1

        self.pos[21:, 0:3] = 1
        self.xvel[21:, 0:3] = 1

        self.pos[33:, 3:18] = 1
        self.yvel[33:, 3:18] = 1

        self.pos[18:21, 0:7] = 1
        self.yvel[18:21, 0:7] = -1

class TestRewPath(TestRewMatrix):
    def __init__(self):
        super().__init__()
        self.pos[3:21, 7:10] = 1
        self.xvel[3:21, 7:10] = -1

        self.pos[0:3, 3:10] = 1
        self.yvel[0:3, 3:10] = -1

        self.pos[0:18, 0:3] = 1
        self.xvel[0:18, 0:3] = 1

class TestRewLoop2(TestRewMatrix):
    def __init__(self):
        super().__init__()
        self.pos[22:33, 14:18] = 1
        self.pos[21:, 0:3] = 1
        self.pos[33:, 3:18] = 1
        self.pos[18:21, 0:7] = 1

class TestRewPath2(TestRewMatrix):
    def __init__(self):
        super().__init__()
        self.pos[3:21, 7:10] = 1
        self.pos[0:3, 3:10] = 1
        self.pos[0:18, 0:3] = 1


# =================== For DMC

class VelocityRewardFunctionWalker(RewardFunction):
    def __init__(self):
        pass
    
    def generate_params_and_pairs(self, traj_states, random_states, random_states_decode):
        batch_size = traj_states.shape[0]
        params = np.random.uniform(low=0, high=8, size=(batch_size, 1)) # (batch_size, 1)

        encode_pairs = np.concatenate([traj_states, random_states], axis=1)
        encode_rewards = self.compute_reward(encode_pairs, params[:, None, :])[:, :, None]
        encode_pairs = np.concatenate([encode_pairs, encode_rewards], axis=-1)

        decode_pairs = random_states_decode
        decode_rewards = self.compute_reward(decode_pairs, params[:, None, :])[:, :, None]
        decode_pairs = np.concatenate([random_states_decode, decode_rewards], axis=-1)

        rewards = encode_pairs[:, 0, -1] # (batch_size,)
        masks = np.ones_like(rewards) # (batch_size,)

        return params, encode_pairs, decode_pairs, rewards, masks
    
    def _sigmoids(self, x, value_at_1, sigmoid):
        if sigmoid == 'gaussian':
            scale = np.sqrt(-2 * np.log(value_at_1))
            return np.exp(-0.5 * (x*scale)**2)

        elif sigmoid == 'linear':
            scale = 1-value_at_1
            scaled_x = x*scale
            return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)
    
    def tolerance(self, x, lower, upper, margin=0.0, sigmoid='gaussian', value_at_margin=0.1):
        in_bounds = np.logical_and(lower <= x, x <= upper)
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, self._sigmoids(d, value_at_margin, sigmoid))
        return value
    
    def compute_reward(self, states, params):
        assert len(states.shape) == len(params.shape), states.shape # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)

        _STAND_HEIGHT = 1.2
        horizontal_velocity = states[..., 24:25]
        torso_upright = states[..., 25:26]
        torso_height = states[..., 26:27]
        standing = self.tolerance(torso_height, lower=_STAND_HEIGHT, upper=float('inf'), margin=_STAND_HEIGHT/2)
        upright = (1 + torso_upright) / 2
        stand_reward = (3*standing + upright) / 4
        move_reward = self.tolerance(horizontal_velocity,
                                        lower=params,
                                        upper=float('inf'),
                                        margin=params/2,
                                        value_at_margin=0.5,
                                        sigmoid='linear')
        # move_reward[params == 0] = stand_reward[params == 0]
        rew = stand_reward * (5*move_reward + 1) / 6
        return rew[..., 0]
    
    def make_encoder_pairs_testing(self, params, random_states):
        assert len(params.shape) == 2, params.shape # (batch_size, 2)
        assert len(random_states.shape) == 3, random_states.shape # (batch_size, num_pairs, obs_dim)

        reward_pair_rews = self.compute_reward(random_states, params[:, None, :])[..., None]
        reward_pairs = np.concatenate([random_states, reward_pair_rews], axis=-1)
        return reward_pairs # (batch_size, reward_pairs, obs_dim + 1)
    
class VelocityRewardFunctionCheetah(RewardFunction):
    def __init__(self):
        pass
    
    def generate_params_and_pairs(self, traj_states, random_states, random_states_decode):
        batch_size = traj_states.shape[0]
        params = np.random.uniform(low=-10, high=10, size=(batch_size, 1)) # (batch_size, 1)

        encode_pairs = np.concatenate([traj_states, random_states], axis=1)
        encode_rewards = self.compute_reward(encode_pairs, params[:, None, :])[:, :, None]
        encode_pairs = np.concatenate([encode_pairs, encode_rewards], axis=-1)

        decode_pairs = random_states_decode
        decode_rewards = self.compute_reward(decode_pairs, params[:, None, :])[:, :, None]
        decode_pairs = np.concatenate([random_states_decode, decode_rewards], axis=-1)

        rewards = encode_pairs[:, 0, -1] # (batch_size,)
        masks = np.ones_like(rewards) # (batch_size,)

        return params, encode_pairs, decode_pairs, rewards, masks
    
    def _sigmoids(self, x, value_at_1, sigmoid):
        if sigmoid == 'linear':
            scale = 1-value_at_1
            scaled_x = x*scale
            return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)
        else:
            raise NotImplementedError
    
    def tolerance(self, x, lower, upper, margin=0.0, sigmoid='linear', value_at_margin=0):
        in_bounds = np.logical_and(lower <= x, x <= upper)
        d = np.where(x < lower, lower - x, x - upper) / margin
        value = np.where(in_bounds, 1.0, self._sigmoids(d, value_at_margin, sigmoid))
        return value
    
    def compute_reward(self, states, params):
        assert len(states.shape) == len(params.shape), states.shape # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)

        horizontal_velocity = states[..., 17:18]
        sign_of_param = np.sign(params)
        horizontal_velocity = horizontal_velocity * sign_of_param
        rew = self.tolerance(horizontal_velocity,
                             lower=np.abs(params),
                             upper=float('inf'),
                             margin=np.abs(params),
                             value_at_margin=0,
                             sigmoid='linear')
        return rew[..., 0]
    
    def make_encoder_pairs_testing(self, params, random_states):
        assert len(params.shape) == 2, params.shape # (batch_size, 2)
        assert len(random_states.shape) == 3, random_states.shape # (batch_size, num_pairs, obs_dim)

        reward_pair_rews = self.compute_reward(random_states, params[:, None, :])[..., None]
        reward_pairs = np.concatenate([random_states, reward_pair_rews], axis=-1)
        return reward_pairs # (batch_size, reward_pairs, obs_dim + 1)

# =================== For Kitchen

class SingleTaskRewardFunction(RewardFunction):
    def __init__(self):
        self.obs_element_indices = {
            "bottom left burner": np.array([11, 12]),
            "top left burner": np.array([15, 16]),
            "light switch": np.array([17, 18]),
            "slide cabinet": np.array([19]),
            "hinge cabinet": np.array([20, 21]),
            "microwave": np.array([22]),
            "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
        }
        self.obs_element_goals = {
            "bottom left burner": np.array([-0.88, -0.01]),
            "top left burner": np.array([-0.92, -0.01]),
            "light switch": np.array([-0.69, -0.05]),
            "slide cabinet": np.array([0.37]),
            "hinge cabinet": np.array([0.0, 1.45]),
            "microwave": np.array([-0.75]),
            "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
        }
        self.dist_thresh = 0.3
        self.num_tasks = len(self.obs_element_indices)

    def generate_params_and_pairs(self, traj_states, random_states, random_states_decode):
        batch_size = traj_states.shape[0]
        params = np.random.randint(self.num_tasks, size=(batch_size, 1))  # (batch_size, 1)
        params = np.eye(self.num_tasks)[params[:, 0]]  # (batch_size, num_tasks)

        encode_pairs = np.concatenate([traj_states, random_states], axis=1)
        encode_rewards = self.compute_reward(encode_pairs, params[:, None, :])[:, :, None]
        encode_pairs = np.concatenate([encode_pairs, encode_rewards], axis=-1)

        decode_pairs = random_states_decode
        decode_rewards = self.compute_reward(decode_pairs, params[:, None, :])[:, :, None]
        decode_pairs = np.concatenate([random_states_decode, decode_rewards], axis=-1)

        rewards = encode_pairs[:, 0, -1]  # (batch_size,)
        masks = np.ones_like(rewards)  # (batch_size,)

        return params, encode_pairs, decode_pairs, rewards, masks

    def compute_reward(self, states, params):
        assert len(states.shape) == len(params.shape), states.shape  # (batch_size, obs_dim) OR (batch_size, num_pairs, obs_dim)
        task_rewards = []
        for task, target_indices in self.obs_element_indices.items():
            task_dists = np.linalg.norm(states[..., target_indices] - self.obs_element_goals[task], axis=-1)
            task_completes = (task_dists < self.dist_thresh).astype(float)
            task_rewards.append(task_completes)
        task_rewards = np.stack(task_rewards, axis=-1)

        return np.sum(task_rewards * params, axis=-1)

    def make_encoder_pairs_testing(self, params, random_states):
        assert len(params.shape) == 2, params.shape  # (batch_size, 2)
        assert len(random_states.shape) == 3, random_states.shape  # (batch_size, num_pairs, obs_dim)

        reward_pair_rews = self.compute_reward(random_states, params[:, None, :])[..., None]
        reward_pairs = np.concatenate([random_states, reward_pair_rews], axis=-1)
        return reward_pairs  # (batch_size, reward_pairs, obs_dim + 1)

