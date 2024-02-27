
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
import functools
import ml_collections
from ml_collections import config_flags
from absl import app, flags
import os
import pickle
import tqdm
from flax.training import checkpoints
import matplotlib.pyplot as plt
import wandb

from fre.common.typing import *
from fre.common.networks.transformer import Transformer
import fre.common.networks.transformer as transformer
from fre.common.dataset import Dataset
from fre.common.typing import *
from fre.common.train_state import TrainState, target_update
from fre.common.networks.basic import Policy, ValueCritic, Critic, ensemblize
from fre.common.wandb import setup_wandb, default_wandb_config, get_flag_dict
from fre.common.envs.gc_utils import GCDataset
from fre.common.envs.env_helper import make_env, get_dataset
from fre.common.evaluation import evaluate
from fre.common.envs.wrappers import EpisodeMonitor, RewardOverride, TruncateObservation
from fre.common.utils import supply_rng
from fre.experiment.rewards_unsupervised import *
from fre.experiment.rewards_eval import *

import flax

###############################
#  Configs
###############################

FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'antmaze-large-diverse-v2', 'Environment name.')
flags.DEFINE_integer('dmc_dataset_size', 5000000, 'ExORL dataset size.')
flags.DEFINE_string('name', 'default', '')

flags.DEFINE_string('save_dir', None, 'Logging dir (if not None, save params).')
flags.DEFINE_string('load_dir', None, 'Logging dir (if not None, load params).')

flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('eval_episodes', 20,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 200000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 200000, 'Eval interval.')
flags.DEFINE_integer('video_interval', 10050000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 512, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')

flags.DEFINE_integer('reward_pairs_encode', 32, 'Number of reward pairs to use for encoding.')
flags.DEFINE_integer('reward_pairs_decode', 8, 'Number of reward pairs to use for decoding.')
flags.DEFINE_integer('reward_pairs_encode_test', 32, 'Number of reward pairs to use for encoding (for testing).')

flags.DEFINE_float('rew_ratio_goal', 0.3333, 'Ratio of reward functions that are goal.')
flags.DEFINE_float('rew_ratio_linear', 0.3333, 'Ratio of reward functions that are linear.')
flags.DEFINE_float('rew_ratio_mlp', 0.3333, 'Ratio of reward functions that are random mlp.')

# Env-Specific Settings
flags.DEFINE_string('start_loc', 'center2', 'Starting location of the ant')
flags.DEFINE_integer('use_discrete_xy', 1, 'Use discrete XY encoding for antmaze?')
flags.DEFINE_integer('dmc_use_oracle', 0, 'Use true rewards during training?')

agent_config = ml_collections.ConfigDict({
    'lr': 1e-4,
    'reward_pairs_emb_dim': 128,
    'hidden_dims': (512, 512, 512),
    'discount': 0.99,
    'expectile': 0.8,
    'temperature': 3.0, # 0 for behavior cloning.
    'tau': 0.001,
    'opt_decay_schedule': 'none',
    'warmup_steps': 150000,
    "num_discrete_embeddings": 32,
    'kl_weight': 0.01,
    'actor_loss_type': 'awr', # awr or ddpg.
    'bc_coefficient': 0.0,
})

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'fre_fre',
    'name': 'fre_{env_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('agent', agent_config, lock_config=False)
config_flags.DEFINE_config_dict('gcdataset', GCDataset.get_default_config(), lock_config=False)
config_flags.DEFINE_config_dict('transformer', transformer.get_default_config(), lock_config=False)


###############################
#  Agent. Contains the neural networks, training logic, and sampling.
###############################

def expectile_loss(adv, diff, expectile=0.7):
    weight = jnp.where(adv >= 0, expectile, (1 - expectile))
    return weight * (diff**2)

class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x):
        return jnp.expand_dims(x, axis=-1)


class FRENetwork(nn.Module):
    transformer_params: dict
    hidden_dims: Sequence[int]
    action_dim: int
    reward_pairs_emb_dim : int
    num_discrete_embeddings: int

    def setup(self):
        self.encoder_transformer = Transformer(**self.transformer_params)
        self.encoder_mean = nn.Dense(self.reward_pairs_emb_dim)
        self.encoder_log_std = nn.Dense(self.reward_pairs_emb_dim)

        self.reward_embed = nn.Embed(self.num_discrete_embeddings, self.reward_pairs_emb_dim // 2)
        self.embed_reward_pairs = nn.Dense(self.reward_pairs_emb_dim // 2)

        self.value = ValueCritic(self.hidden_dims)
        self.critic = ensemblize(Critic, num_qs=2)(self.hidden_dims)
        self.actor = Policy(self.hidden_dims, action_dim=self.action_dim,
            log_std_min=-5.0, state_dependent_std=False, tanh_squash_distribution=False)
        
        self.reward_predict = ValueCritic(self.hidden_dims)
        
    def __call__(self, x): # (batch_size, timesteps, emb_dim)
        raise None
    
    def get_transformer_encoding(self, reward_state_pairs):
        reward_states = reward_state_pairs[:, :, :-1]
        reward_values = reward_state_pairs[:, :, -1]
        reward_values_idx = jnp.floor((reward_values / 2.0 + 0.5) * self.num_discrete_embeddings).astype(jnp.int32)
        reward_values_idx = jnp.clip(reward_values_idx, 0, self.num_discrete_embeddings - 1)

        reward_state_emb = self.embed_reward_pairs(reward_states)
        reward_state_val = self.reward_embed(reward_values_idx)
        reward_state_pairs = jnp.concatenate([reward_state_emb, reward_state_val], axis=-1)

        w_pre = self.encoder_transformer(reward_state_pairs, train=True) # [batch, reward_pairs, emb_dim]
        w_pair_mean = w_pre.mean(axis=1)
        w_mean = self.encoder_mean(w_pair_mean)
        w_log_std = self.encoder_log_std(w_pair_mean)

        return w_mean, w_log_std # (batch_size, emb_dim)

    def get_value(self, w, obs):
        w_and_obs = jnp.concatenate([w, obs], axis=-1)
        return self.value(w_and_obs)

    def get_critic(self, w, obs, actions):
        w_and_obs = jnp.concatenate([w, obs], axis=-1)
        return self.critic(w_and_obs, actions)

    def get_actor(self, w, obs, temperature=1.0):
        w_and_obs = jnp.concatenate([w, obs], axis=-1)
        return self.actor(w_and_obs, temperature)
    
    def get_reward_pred(self, w, reward_pairs): # Reward Pairs: [batch, reward_pairs, obs_dim + 1]
        z_expand = jnp.expand_dims(w, axis=1) # [batch, 1, emb_dim]
        z_expand = jnp.repeat(z_expand, repeats=reward_pairs.shape[1], axis=1) 
        reward_states = reward_pairs[:, :, :-1]
        w_and_obs = jnp.concatenate([z_expand, reward_states], axis=-1)
        reward_pred = self.reward_predict(w_and_obs)
        return reward_pred # [batch, reward_pairs]
    
    def get_all(self, reward_state_pairs, obs, actions):
        w_mean, w_log_std = self.get_transformer_encoding(reward_state_pairs)
        w = w_mean
        w_and_obs = jnp.concatenate([w, obs], axis=-1)
        ret = self.value(w_and_obs), self.get_actor(w, obs), self.get_reward_pred(w, reward_state_pairs), self.critic(w_and_obs, actions)
        return ret

    
class FREAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    fre: TrainState
    target_fre: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @functools.partial(jax.jit, static_argnames=('train_encoder', 'train_actor', 'train_critic'))
    def update(agent, batch: Batch, train_encoder=True, train_actor=True, train_critic=True, apply_updates=True) -> InfoDict:
        new_rng, w_key = jax.random.split(agent.rng, 2)
        reward_state_pairs = batch['reward_pairs_encode']
        reward_pairs_decode = batch['reward_pairs_decode']

        def full_loss_fn(params):
            if train_encoder:
                w_mean, w_log_std = agent.fre.do('get_transformer_encoding')(reward_state_pairs, params=params)
            else:
                w_mean, w_log_std = agent.fre.do('get_transformer_encoding')(reward_state_pairs)
            w_no_grad = jax.lax.stop_gradient(w_mean)

            if train_encoder:
                # Reward Pred Loss
                w = w_mean + jax.random.normal(agent.rng, w_mean.shape) * jnp.exp(w_log_std)
                reward_pred = agent.fre.do('get_reward_pred')(w, reward_pairs_decode, params=params)
                reward_truths = reward_pairs_decode[:, :, -1]
                reward_pred_loss = ((reward_pred - reward_truths)**2).mean()
                kl_loss = -0.5 * (1 + w_log_std - w_mean**2 - jnp.exp(w_log_std)).mean()
                reward_kl_loss = reward_pred_loss + kl_loss * agent.config['kl_weight']
                reward_pred_info = {
                    'reward_pred_loss': reward_pred_loss,
                    'reward_pred': reward_pred.mean(),
                    'kl_loss': kl_loss,
                }
            else:
                reward_kl_loss = 0.0
                reward_pred_info = {}

            if train_critic:
                # Implicit Q-Learning
                # Value Loss: Update V towards expectile of min(q1, q2).
                w_target_mean = w_no_grad
                w_mean = w_no_grad
                q1, q2 = agent.target_fre.do("get_critic")(w_target_mean, batch['observations'], batch['actions'])
                q = jnp.minimum(q1, q2)
                v = agent.fre.do("get_value")(w_mean, batch['observations'], params=params)
                adv = q - v
                v_loss = expectile_loss(adv, q - v, agent.config['expectile'])
                v_loss = (v_loss).mean()

                # Critic Loss. Update Q = r
                next_v = jax.lax.stop_gradient(agent.fre.do("get_value")(w_mean, batch['next_observations']))
                q = batch['rewards'] + agent.config['discount'] * batch['masks'] * next_v

                q1, q2 = agent.fre.do("get_critic")(w_mean, batch['observations'], batch['actions'], params=params)
                q_loss = (q1 - q) ** 2 + (q2 - q) ** 2
                q_loss = (q_loss).mean()

                value_loss = v_loss + q_loss
                value_info = {
                    # 'value_loss': value_loss,
                    'v_loss': v_loss,
                    'q_loss': q_loss,
                    'v': v.mean(),
                    'q': q.mean(),
                }
            else:
                value_loss = 0.0
                value_info = {}
            
            if train_actor:
                # Actor Loss
                actor_w = w_mean
                if agent.config['actor_loss_type'] == 'awr':
                    v = agent.fre.do("get_value")(w_no_grad, batch['observations'])
                    q1, q2 = agent.fre.do("get_critic")(w_no_grad, batch['observations'], batch['actions'])
                    q = jnp.minimum(q1, q2)
                    adv = q - v

                    actions = batch['actions']
                    exp_a = jnp.exp(adv * agent.config['temperature'])
                    exp_a = jnp.minimum(exp_a, 100.0)
                    dist = agent.fre.do('get_actor')(actor_w, batch['observations'], params=params)
                    log_probs = dist.log_prob(actions)
                    assert exp_a.shape == log_probs.shape
                    print("Log probs shape", log_probs.shape)
                    actor_loss = -(exp_a * log_probs).mean()
                elif agent.config['actor_loss_type'] == 'ddpg':
                    dist = agent.fre.do("get_actor")(actor_w, batch['observations'], params=params)
                    normalized_actions = jnp.tanh(dist.loc)
                    q1, q2 = agent.fre.do("get_critic")(w_no_grad, batch['observations'], normalized_actions)
                    q = (q1 + q2) / 2

                    q_loss = -q.mean()

                    log_probs = dist.log_prob(batch['actions'])
                    bc_loss = -((agent.config['bc_coefficient'] * log_probs)).mean()

                    actor_loss = ((q_loss + bc_loss)).mean()

                std = dist.stddev().mean()
                mse_error = jnp.square(dist.loc - batch['actions']).mean()
                actor_info = {
                    'actor_loss': actor_loss,
                    'std': std,
                    'adv': adv.mean(),
                    'mse_error': mse_error,
                }
            else:
                actor_loss = 0.0
                actor_info = {}
        
            return value_loss + actor_loss + reward_kl_loss, {**value_info, **actor_info, **reward_pred_info}
        
        new_fre, info = agent.fre.apply_loss_fn(loss_fn=full_loss_fn, has_aux=True)
        new_target_fre = target_update(agent.fre, agent.target_fre, agent.config['target_update_rate'])

        return agent.replace(fre=new_fre, target_fre=new_target_fre, rng=new_rng), {
            **info
        }

    @jax.jit
    def sample_actions(agent,
                       observations: np.ndarray, # [obs_dim]
                       reward_pairs: np.ndarray, # [1, reward_pairs, obs_dim + 1]
                       *,
                       seed: PRNGKey,
                       temperature: float = 1.0) -> jnp.ndarray:
        if type(observations) is dict:
            observations = jnp.concatenate([observations['observation'], observations['goal']], axis=-1)
        observations = jnp.expand_dims(observations, axis=0)
        print("Reward pairs shape", reward_pairs.shape)
        w_mean, w_log_std = agent.fre.do('get_transformer_encoding')(reward_pairs)
        print("W shape", w_mean.shape)
        actions = agent.fre.do('get_actor')(w_mean, observations, temperature=temperature).sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions[0]
    
    def get_reward_pred(agent, observations: np.ndarray, reward_pairs: np.ndarray):
        # append a dummy reward to the observations.
        decode_pairs = jnp.concatenate([observations, np.ones((observations.shape[0], 1))], axis=-1)[None]
        w_mean, w_log_std = agent.fre.do('get_transformer_encoding')(reward_pairs)
        return agent.fre.do('get_reward_pred')(w_mean, decode_pairs)
    
    def get_value_pred(agent, observations: np.ndarray, reward_pairs: np.ndarray):
        w_mean, w_log_std = agent.fre.do('get_transformer_encoding')(reward_pairs) # [batch, emb_dim]
        w_expand = jnp.repeat(w_mean, repeats=observations.shape[0], axis=0)
        v = agent.fre.do('get_value')(w_expand, observations)
        return v

def create_learner(
                seed: int,
                batch: Batch,
                transformer_params: dict,
                lr: float,
                reward_pairs_emb_dim: int,
                num_discrete_embeddings: int,
                kl_weight: float,
                hidden_dims: Sequence[int],
                discount: float,
                tau: float,
                expectile: float,
                temperature: float,
                max_steps: Optional[int],
                opt_decay_schedule: str,
                actor_loss_type: str,
                bc_coefficient: float,
            **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = batch['actions'].shape[-1]
        transformer_params['causal'] = False
        transformer_params['emb_dim'] = reward_pairs_emb_dim
        transformer_params['num_heads'] = 2
        transformer_params['num_layers'] = 2
        fre_def = FRENetwork(transformer_params, hidden_dims, action_dim, reward_pairs_emb_dim=reward_pairs_emb_dim, num_discrete_embeddings=num_discrete_embeddings)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-lr, max_steps)
            tx = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            tx = optax.adam(learning_rate=lr)

        params = fre_def.init(actor_key, batch['reward_pairs_encode'], batch['observations'], batch['actions'], method='get_all')['params']
        fre = TrainState.create(fre_def, params, tx=tx)
        target_fre = TrainState.create(fre_def, params)

        config = flax.core.FrozenDict(dict(
            discount=discount, temperature=temperature, expectile=expectile, target_update_rate=tau, reward_pairs_emb_dim=reward_pairs_emb_dim, kl_weight=kl_weight, actor_loss_type=actor_loss_type, bc_coefficient=bc_coefficient, num_discrete_embeddings=num_discrete_embeddings
        ))

        return FREAgent(rng, fre=fre, target_fre=target_fre, config=config)

###############################
#  Run Script. Loads data, logs to wandb, and runs the training loop.
###############################

def main(_):
    # Create wandb logger
    setup_wandb(FLAGS.agent.to_dict(), **FLAGS.wandb)
    assert 'ant' in FLAGS.env_name or 'dmc' in FLAGS.env_name or 'kitchen' in FLAGS.env_name

    agent = None

    if FLAGS.save_dir is not None:
        os.makedirs(FLAGS.save_dir, exist_ok=True)
        print(f'Saving config to {FLAGS.save_dir}/config.pkl')
        with open(os.path.join(FLAGS.save_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(get_flag_dict(), f)

    if 'ant' in FLAGS.env_name:
        import fre.common.envs.d4rl.d4rl_ant as d4rl_ant
        env = d4rl_ant.CenteredMaze(FLAGS.env_name)
        dataset = get_dataset(env, FLAGS.env_name)
        dataset = dataset.copy({'masks': np.ones_like(dataset['masks'])})
        dataset_gc = GCDataset(dataset, **FLAGS.gcdataset.to_dict())
        example_batch = dataset.sample(1)
        eval_env = EpisodeMonitor(RewardOverride(d4rl_ant.CenteredMaze(FLAGS.env_name)))
        ## =============== Reward Functions for Testing =============== ##

        base_ob = example_batch['observations'][0]
        def goal_at(x,y):
            goal = base_ob.copy()
            goal[:2] = [x,y]
            return goal
        reward_fn_ratios = [FLAGS.rew_ratio_goal, FLAGS.rew_ratio_linear, FLAGS.rew_ratio_mlp]
        GoalReachingRewards = GoalReachingRewardFunction()
        VelocityRewards = VelocityRewardFunction()
        LinearRewards = LinearRewardFunction()
        SimplexRewards = SimplexRewardFunction(num_simplex=10)
        RandomRewards = RandomRewardFunction(num_simplex=10000)
        reward_fns = [GoalReachingRewards, LinearRewards, RandomRewards]
        
        linear_states = dataset.sample(5)['observations'][:, None, :]
        linear_params = LinearRewards.generate_params_and_pairs(linear_states, linear_states, linear_states)[0] # (5, params_dim)
        print("Linear Params: ", linear_params)
        test_rewards = [
            (GoalReachingRewards, 'goal_bottom', goal_at(28, 0)),
            (GoalReachingRewards, 'goal_left', goal_at(0, 15)),
            (GoalReachingRewards, 'goal_top', goal_at(35, 24)),
            (GoalReachingRewards, 'goal_center', goal_at(12, 24)),
            (GoalReachingRewards, 'goal_right', goal_at(33, 16)),
            (VelocityRewards, 'vel_left', np.array([-1, 0])),
            (VelocityRewards, 'vel_up', np.array([0, 1])),
            (VelocityRewards, 'vel_down', np.array([0, -1])),
            (VelocityRewards, 'vel_right', np.array([1, 0])),
            (SimplexRewards, 'simplex_1', np.array([1])),
            (SimplexRewards, 'simplex_2', np.array([2])),
            (SimplexRewards, 'simplex_3', np.array([3])),
            (SimplexRewards, 'simplex_4', np.array([4])),
            (SimplexRewards, 'simplex_5', np.array([5])),
            (TestRewPath(), 'path_center', np.array([0])),
            (TestRewLoop(), 'path_loop', np.array([0])),
            (TestRewMatrixEdges(), 'path_edges', np.array([0])),
        ]

        slices = []
        slices.append(0)
        for j in range(len(reward_fns)):
            slices.append(int(FLAGS.batch_size * reward_fn_ratios[j]))
        slices[-1] = FLAGS.batch_size - sum(slices[:-1])
        print("Number of samples for each reward func: ", slices)
        slices = np.cumsum(slices)
        print("Cumsum of samples for each reward func: ", slices)
    elif 'dmc' in FLAGS.env_name:
        _, env_name, task_name = FLAGS.env_name.split('_')
        env = make_env(f'{env_name}_{task_name}')
        env.reset()

        # Load dataset.
        import pathlib
        file_path = str(pathlib.Path().resolve().parents[0])
        path = file_path + f'/fre/data/exorl/{env_name}/rnd'
        dataset_npy = os.path.join(path, task_name + '.npy')
        dataset = np.load(dataset_npy, allow_pickle=True).item()
        dataset['dones_float'] = np.zeros_like(dataset['rewards'])
        dataset['dones_float'][::1000] = 1.0 # Each exorl trajectory is length 1000.
        dataset['dones_float'][-1] = 1.0 # Last state is terminal.

        # For evaluating the velocity rewareds, we need an augmented observation that uses the physics state.
        if 'walker' in FLAGS.env_name:
            aux = np.load(file_path+'/fre/data/aux_walker.npy', allow_pickle=True)
        elif 'cheetah' in FLAGS.env_name:
            aux = np.load(file_path+'/fre/data/aux_cheetah.npy', allow_pickle=True)
        dataset['observations'] = np.concatenate([dataset['observations'], aux], axis=1)
        aux_shifted = np.concatenate([aux[1::], aux[-1:]], axis=0)
        dataset['next_observations'] = np.concatenate([dataset['next_observations'], aux_shifted], axis=1)

        dataset = Dataset(dataset)
        dataset_gc = GCDataset(dataset, **FLAGS.gcdataset.to_dict())
        eval_env = EpisodeMonitor(RewardOverride(make_env(f'{env_name}_{task_name}')))
        eval_env.reset()

        def goal_at(seed):
            return dataset.sample(1, indx=seed*777)['observations']

        VelocityRewardsWalker = VelocityRewardFunctionWalker()
        VelocityRewardsCheetah = VelocityRewardFunctionCheetah()
        GoalReachingRewards = GoalReachingRewardFunction()
        LinearRewards = LinearRewardFunction()
        if 'walker' in FLAGS.env_name:
            reward_fns = [VelocityRewardsWalker]
            RandomRewards = RandomRewardFunction(num_simplex=10000, obs_len=27)
            test_rewards = [
                (VelocityRewardsWalker, 'vel0.1', np.array([0.1])),
                (VelocityRewardsWalker, 'vel1', np.array([1])),
                (VelocityRewardsWalker, 'vel4', np.array([4])),
                (VelocityRewardsWalker, 'vel8', np.array([8])),
                (GoalReachingRewards, 'goal_1', goal_at(1)),
                (GoalReachingRewards, 'goal_2', goal_at(2)),
                (GoalReachingRewards, 'goal_3', goal_at(3)),
                (GoalReachingRewards, 'goal_4', goal_at(4)),
                (GoalReachingRewards, 'goal_5', goal_at(5)),
            ]
        elif 'cheetah' in FLAGS.env_name:
            reward_fns = [VelocityRewardsCheetah]
            RandomRewards = RandomRewardFunction(num_simplex=10000, obs_len=18)
            test_rewards = [
                (VelocityRewardsCheetah, 'vel10Back', np.array([-10])),
                (VelocityRewardsCheetah, 'vel2Back', np.array([-2])),
                (VelocityRewardsCheetah, 'vel2', np.array([2])),
                (VelocityRewardsCheetah, 'vel10', np.array([10])),
                (GoalReachingRewards, 'goal_1', goal_at(1)),
                (GoalReachingRewards, 'goal_2', goal_at(2)),
                (GoalReachingRewards, 'goal_3', goal_at(3)),
                (GoalReachingRewards, 'goal_4', goal_at(4)),
                (GoalReachingRewards, 'goal_5', goal_at(5)),
            ]
        if FLAGS.dmc_use_oracle:
            pass
        else:
            reward_fns = [GoalReachingRewards, LinearRewards, RandomRewards]

        reward_fn_ratios = [FLAGS.rew_ratio_goal, FLAGS.rew_ratio_linear, FLAGS.rew_ratio_mlp]
        slices = []
        slices.append(0)
        for j in range(len(reward_fns)):
            slices.append(int(FLAGS.batch_size * reward_fn_ratios[j]))
        slices[-1] = FLAGS.batch_size - sum(slices[:-1])
        print("Number of samples for each reward func: ", slices)
        slices = np.cumsum(slices)
        print("Cumsum of samples for each reward func: ", slices)

    elif 'kitchen' in FLAGS.env_name:
        # HACK: Monkey patching to make it compatible with Python 3.10.
        import collections
        if not hasattr(collections, 'Mapping'):
            collections.Mapping = collections.abc.Mapping

        def make_kitchen_env(env_name):
            env = make_env(env_name)
            # Only use the first 30 dimensions (because the other half corresponds to the goal).
            env = TruncateObservation(env, truncate_size=30)
            return env
        dataset = get_dataset(make_kitchen_env(FLAGS.env_name), FLAGS.env_name, filter_terminals=True)
        dataset = dataset.copy({'observations': dataset['observations'][:, :30], 'next_observations': dataset['next_observations'][:, :30]})
        dataset = dataset.copy({'masks': np.ones_like(dataset['masks'])})
        dataset_gc = GCDataset(dataset, **FLAGS.gcdataset.to_dict())
        eval_env = EpisodeMonitor(RewardOverride(make_kitchen_env(FLAGS.env_name)))

        SingleTaskRewards = SingleTaskRewardFunction()
        GoalReachingRewards = GoalReachingRewardFunction()
        LinearRewards = LinearRewardFunction()
        RandomRewards = RandomRewardFunction(num_simplex=10000, obs_len=30)
        reward_fns = [GoalReachingRewards, LinearRewards, RandomRewards]

        reward_fn_ratios = [FLAGS.rew_ratio_goal, FLAGS.rew_ratio_linear, FLAGS.rew_ratio_mlp]
        slices = []
        slices.append(0)
        for j in range(len(reward_fns)):
            slices.append(int(FLAGS.batch_size * reward_fn_ratios[j]))
        slices[-1] = FLAGS.batch_size - sum(slices[:-1])
        print("Number of samples for each reward func: ", slices)
        slices = np.cumsum(slices)
        print("Cumsum of samples for each reward func: ", slices)
        test_rewards = [
            (SingleTaskRewards, 'binary_bottom_left_burner', np.array([1, 0, 0, 0, 0, 0, 0])),
            (SingleTaskRewards, 'binary_top_left_burner', np.array([0, 1, 0, 0, 0, 0, 0])),
            (SingleTaskRewards, 'binary_light_switch', np.array([0, 0, 1, 0, 0, 0, 0])),
            (SingleTaskRewards, 'binary_slide_cabinet', np.array([0, 0, 0, 1, 0, 0, 0])),
            (SingleTaskRewards, 'binary_hinge_cabinet', np.array([0, 0, 0, 0, 1, 0, 0])),
            (SingleTaskRewards, 'binary_microwave', np.array([0, 0, 0, 0, 0, 1, 0])),
            (SingleTaskRewards, 'binary_kettle', np.array([0, 0, 0, 0, 0, 0, 1])),
        ]
    else:
        raise NotImplementedError
    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):

        # Sample a batch of trajectories.
        # Get future states from the trajectory, and random states.
        num_traj_states = min(8, FLAGS.reward_pairs_encode-1)

        num_random_states = FLAGS.reward_pairs_encode - num_traj_states
        num_random_states_decode = FLAGS.reward_pairs_decode
        batch = dataset_gc.sample_traj_random(FLAGS.batch_size, num_traj_states, num_random_states, num_random_states_decode)
        # The first index of traj_states contains the CURRENT state.
        assert batch['traj_states'].shape == (FLAGS.batch_size, num_traj_states, dataset['observations'].shape[-1])
        assert batch['random_states'].shape == (FLAGS.batch_size, num_random_states, dataset['observations'].shape[-1])
        assert batch['random_states_decode'].shape == (FLAGS.batch_size, num_random_states_decode, dataset['observations'].shape[-1])

        encode_pairs = np.zeros((FLAGS.batch_size, FLAGS.reward_pairs_encode, dataset['observations'].shape[-1] + 1))
        decode_pairs = np.zeros((FLAGS.batch_size, FLAGS.reward_pairs_decode, dataset['observations'].shape[-1] + 1))
        rewards = np.zeros((FLAGS.batch_size))
        masks = np.zeros((FLAGS.batch_size))        
        for j in range(len(reward_fns)):
            batch_traj_states = batch['traj_states'][slices[j]:slices[j+1], :, :]
            batch_random_states = batch['random_states'][slices[j]:slices[j+1], :, :]
            batch_random_states_decode = batch['random_states_decode'][slices[j]:slices[j+1], :, :]
            params_slice, encode_pairs_slice, decode_pairs_slice, rewards_slice, masks_slice = reward_fns[j].generate_params_and_pairs(batch_traj_states, batch_random_states, batch_random_states_decode) 
            encode_pairs[slices[j]:slices[j+1], :, :] = encode_pairs_slice
            decode_pairs[slices[j]:slices[j+1], :, :] = decode_pairs_slice
            rewards[slices[j]:slices[j+1]] = rewards_slice
            masks[slices[j]:slices[j+1]] = masks_slice
        
        assert len(encode_pairs.shape) == 3 # (batch_size, reward_pairs_encode, obs_dim + 1)
        assert len(decode_pairs.shape) == 3 # (batch_size, reward_pairs_decode, obs_dim + 1)

        batch['rewards'] = rewards
        batch['masks'] = masks
        batch['reward_pairs_encode'] = encode_pairs
        batch['reward_pairs_decode'] = decode_pairs

        if FLAGS.use_discrete_xy and 'ant' in FLAGS.env_name:
            batch['observations'] = d4rl_ant.discretize_obs(batch['observations'])
            batch['next_observations'] = d4rl_ant.discretize_obs(batch['next_observations'])

        # Don't train agents using the auxilliary physics states.
        if 'walker' in FLAGS.env_name:
            batch['observations'] = batch['observations'][:, :24]
            batch['next_observations'] = batch['next_observations'][:, :24]
        elif 'cheetah' in FLAGS.env_name:
            batch['observations'] = batch['observations'][:, :17]
            batch['next_observations'] = batch['next_observations'][:, :17]

        if agent is None:
            agent = create_learner(FLAGS.seed,
                batch,
                transformer_params=FLAGS.transformer.to_dict(),
                max_steps=FLAGS.max_steps,
                **FLAGS.agent)
            if FLAGS.load_dir is not None:
                agent = checkpoints.restore_checkpoint(FLAGS.load_dir, agent)
    
        agent, update_info = agent.update(batch, 
                                        train_encoder=(i <= FLAGS.agent['warmup_steps']),
                                        train_actor=(i > FLAGS.agent['warmup_steps']),
                                        train_critic=(i > FLAGS.agent['warmup_steps'])
        )

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            
            # Debug logs for wandb.
            if i == FLAGS.log_interval:
                fig, ax = plt.subplots(figsize=(3.0, 3.0))
                reward_pair_rewards = batch['reward_pairs_encode'][:, :, -1].flatten()
                ax.hist(reward_pair_rewards, bins=32)
                ax.set_title("Reward Values (For Reward Pairs)")
                wandb.log({"Reward Values (For Reward Pairs)": wandb.Image(fig)}, step=i)
                fig.clf()
                plt.cla()
                plt.clf()
                plt.close('all')

            wandb.log(train_metrics, step=i)

        # Evaluate the unsupervised rewards on the evaluation rewards.
        if i % 10000 == 0 and i < FLAGS.agent['warmup_steps']:
            eval_rew_metrics = {}
            for k, test_reward in enumerate(test_rewards):
                test_reward_generator, test_reward_label, test_reward_params = test_reward
                random_states_encode = dataset.sample(FLAGS.reward_pairs_encode_test)['observations']
                random_states_decode = dataset.sample(FLAGS.reward_pairs_encode_test)['observations']
                test_reward_pairs = test_reward_generator.make_encoder_pairs_testing(test_reward_params[None], \
                                                                                        random_states_encode[None])
                test_reward_pairs_decode = test_reward_generator.make_encoder_pairs_testing(test_reward_params[None], \
                                                                                        random_states_decode[None])
                assert test_reward_pairs.shape == (1, FLAGS.reward_pairs_encode_test, dataset['observations'].shape[-1] + 1)
                true_decode_rewards = test_reward_pairs_decode[0, :, -1] # (reward_pairs_encode_test, )
                decode_predictions = agent.get_reward_pred(random_states_decode, test_reward_pairs)[0] # (reward_pairs_encode_test, )
                assert true_decode_rewards.shape == decode_predictions.shape
                loss = jnp.mean((true_decode_rewards - decode_predictions)**2)
                eval_rew_metrics[f'rew_pred/{test_reward_label}'] = loss
            if 'ant' in FLAGS.env_name:
                # Merge separate metrics into simpler metrics.
                total_goals = eval_rew_metrics['rew_pred/goal_bottom'] + eval_rew_metrics['rew_pred/goal_center'] + eval_rew_metrics['rew_pred/goal_top'] + eval_rew_metrics['rew_pred/goal_left'] + eval_rew_metrics['rew_pred/goal_right']
                total_velocity = eval_rew_metrics['rew_pred/vel_left'] + eval_rew_metrics['rew_pred/vel_up'] + eval_rew_metrics['rew_pred/vel_down'] + eval_rew_metrics['rew_pred/vel_right']
                total_simplex = eval_rew_metrics['rew_pred/simplex_1'] + eval_rew_metrics['rew_pred/simplex_2'] + eval_rew_metrics['rew_pred/simplex_3'] + eval_rew_metrics['rew_pred/simplex_4'] + eval_rew_metrics['rew_pred/simplex_5']
                total_path = eval_rew_metrics['rew_pred/path_center'] + eval_rew_metrics['rew_pred/path_loop'] + eval_rew_metrics['rew_pred/path_edges']
                eval_rew_metrics['rew_pred_total/total_goals'] = total_goals
                eval_rew_metrics['rew_pred_total/total_velocity'] = total_velocity
                eval_rew_metrics['rew_pred_total/total_simplex'] = total_simplex
                eval_rew_metrics['rew_pred_total/total_path'] = total_path
                wandb.log(eval_rew_metrics, step=i)
            elif 'dmc' in FLAGS.env_name:
                # Merge separate metrics into simpler metrics.
                total_goals = eval_rew_metrics['rew_pred/goal_1'] + eval_rew_metrics['rew_pred/goal_2'] + eval_rew_metrics['rew_pred/goal_3'] + eval_rew_metrics['rew_pred/goal_4'] + eval_rew_metrics['rew_pred/goal_5']
                if 'cheetah' in FLAGS.env_name:
                    total_vel = eval_rew_metrics['rew_pred/vel10Back'] + eval_rew_metrics['rew_pred/vel2Back'] + eval_rew_metrics['rew_pred/vel2'] + eval_rew_metrics['rew_pred/vel10']
                elif 'walker' in FLAGS.env_name:
                    total_vel = eval_rew_metrics['rew_pred/vel0.1'] + eval_rew_metrics['rew_pred/vel1'] + eval_rew_metrics['rew_pred/vel4'] + eval_rew_metrics['rew_pred/vel8']
                eval_rew_metrics['rew_pred_total/total_goals'] = total_goals
                eval_rew_metrics['rew_pred_total/total_vel'] = total_vel
                wandb.log(eval_rew_metrics, step=i)

        # Evaluate on test tasks. These are training tasks AND test tasks.
        if i % FLAGS.eval_interval == 0 or (i == 10000 and FLAGS.eval_interval < 10006000):
            print("Performing Eval Loop")
            record_video = i % FLAGS.video_interval == 0
            eval_metrics = {}

            for k, test_reward in enumerate(test_rewards):
                test_reward_generator, test_reward_label, test_reward_params = test_reward
                print("Eval on reward function", test_reward_label)

                # Update eval env to record the right reward.
                def override_reward(s):
                    r = test_reward_generator.compute_reward(s[None,:], test_reward_params[None, :])
                    return r[0]
                eval_env.env.reward_fn = override_reward
                random_states_encode = dataset.sample(FLAGS.reward_pairs_encode_test)['observations']
                test_reward_pairs = test_reward_generator.make_encoder_pairs_testing(test_reward_params[None], \
                                                                                        random_states_encode[None])
                assert test_reward_pairs.shape == (1, FLAGS.reward_pairs_encode_test, dataset['observations'].shape[-1] + 1)

                # Run policy.
                policy_fn = functools.partial(supply_rng(agent.sample_actions), temperature=0.0, reward_pairs=test_reward_pairs)
                if 'dmc' in FLAGS.env_name:
                    eval_info, trajs = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes, record_video=record_video, return_trajectories=True, clip_return_at_goal=('goal' in test_reward_label), use_discrete_xy=False, clip_margin=100)
                elif 'antmaze' in FLAGS.env_name:
                    eval_info, trajs = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes, record_video=record_video, return_trajectories=True, clip_return_at_goal=('goal' in test_reward_label), use_discrete_xy=FLAGS.use_discrete_xy)
                elif 'kitchen' in FLAGS.env_name:
                    eval_info, trajs = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes, record_video=record_video, return_trajectories=True, clip_return_at_goal=('goal' in test_reward_label), use_discrete_xy=False, binary_return=('binary' in test_reward_label))
                else:
                    raise NotImplementedError

                eval_metrics[f'evaluation/{test_reward_label}.return'] = eval_info['episode.return']
                if record_video:
                    wandb.log({f'{test_reward_label}.video': eval_info['video']}, step=i)

                # Antmaze Specific Logging
                if 'antmaze' in FLAGS.env_name and 'large' in FLAGS.env_name and FLAGS.env_name.startswith('antmaze'):
                    import fre.experiment.ant_helper as ant_helper
                    # Make an image of the trajectories.
                    traj_image = d4rl_ant.trajectory_image(eval_env, trajs)
                    # eval_metrics[f'trajectories/{test_reward_label}'] = wandb.Image(traj_image)

                    # Make image of reward function predictions.
                    test_reward_expand = np.tile(test_reward_params[None, :], (280, 1)) # (280, 3)
                    ground_truth_rew = lambda s_grid : test_reward_generator.compute_reward(s_grid, test_reward_expand)
                    true_rew_img = ant_helper.value_image(eval_env, dataset, ground_truth_rew, None)
                    # eval_metrics[f'draw_true/{test_reward_label}'] = wandb.Image(true_rew_img)

                    mask = []
                    for pair in test_reward_pairs[0]:
                        mask.append(pair[:2])
                    mask_rew_img = ant_helper.value_image(eval_env, dataset, ground_truth_rew, mask)
                    # eval_metrics[f'draw_mask/{test_reward_label}'] = wandb.Image(mask_rew_img)

                    pred_rew = lambda s_grid : agent.get_reward_pred(s_grid, test_reward_pairs)
                    pred_rew_img = ant_helper.value_image(eval_env, dataset, pred_rew, None)
                    # eval_metrics[f'draw_pred/{test_reward_label}'] = wandb.Image(pred_rew_img)

                    def pred_value(s_grid):
                        if FLAGS.use_discrete_xy and 'ant' in FLAGS.env_name:
                            s_grid = d4rl_ant.discretize_obs(s_grid)
                        return agent.get_value_pred(s_grid, test_reward_pairs)
                    pred_value_img = ant_helper.value_image(eval_env, dataset, pred_value, None, clip=False)
                    # eval_metrics[f'draw_value1/{test_reward_label}'] = wandb.Image(pred_value_img1)


                    full_img = np.concatenate([
                        np.concatenate([true_rew_img, mask_rew_img], axis=0), 
                        np.concatenate([pred_rew_img, traj_image], axis=0),
                        np.concatenate([pred_value_img, pred_value_img], axis=0)
                    ], axis=1)
                    print("Min/Max of full_img is", np.min(full_img), np.max(full_img))
                    # if any nans, breakpoint.
                    if np.isnan(full_img).any():
                        breakpoint()
                    eval_metrics[f'draw/{test_reward_label}'] = wandb.Image(full_img)

            if 'ant' in FLAGS.env_name:
                # Merge separate metrics into simpler metrics.
                total_goals = eval_metrics['evaluation/goal_bottom.return'] + eval_metrics['evaluation/goal_center.return'] + eval_metrics['evaluation/goal_top.return'] + eval_metrics['evaluation/goal_left.return'] + eval_metrics['evaluation/goal_right.return']
                total_velocity = eval_metrics['evaluation/vel_left.return'] + eval_metrics['evaluation/vel_up.return'] + eval_metrics['evaluation/vel_down.return'] + eval_metrics['evaluation/vel_right.return']
                total_simplex = eval_metrics['evaluation/simplex_1.return'] + eval_metrics['evaluation/simplex_2.return'] + eval_metrics['evaluation/simplex_3.return'] + eval_metrics['evaluation/simplex_4.return'] + eval_metrics['evaluation/simplex_5.return']
                total_path = eval_metrics['evaluation/path_center.return'] + eval_metrics['evaluation/path_loop.return'] + eval_metrics['evaluation/path_edges.return']
                eval_metrics['evaluation_total/total_goals'] = total_goals
                eval_metrics['evaluation_total/total_velocity'] = total_velocity
                eval_metrics['evaluation_total/total_simplex'] = total_simplex
                eval_metrics['evaluation_total/total_path'] = total_path
                print(eval_metrics)
            elif 'dmc' in FLAGS.env_name:
                total_goals = eval_metrics['evaluation/goal_1.return'] + eval_metrics['evaluation/goal_2.return'] + eval_metrics['evaluation/goal_3.return'] + eval_metrics['evaluation/goal_4.return'] + eval_metrics['evaluation/goal_5.return']
                eval_metrics['evaluation/total_goals'] = total_goals
            elif 'kitchen' in FLAGS.env_name:
                total_test = 0.
                for test_reward in test_rewards:
                    total_test += eval_metrics[f'evaluation/{test_reward[1]}.return']
                eval_metrics['evaluation/total_test'] = total_test

            wandb.log(eval_metrics, step=i)

        if i % FLAGS.save_interval == 0 and FLAGS.save_dir is not None:
            checkpoints.save_checkpoint(FLAGS.save_dir, agent, i)

if __name__ == '__main__':
    app.run(main)