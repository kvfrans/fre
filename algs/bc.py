import os
from absl import app, flags
from functools import partial
import numpy as np
import tqdm
import jax
import jax.numpy as jnp
import optax
import flax
import wandb
from ml_collections import config_flags
import pickle
from flax.training import checkpoints
import ml_collections

import fre.common.envs.d4rl.d4rl_utils as d4rl_utils
from fre.common.envs.gc_utils import GCDataset
from fre.common.envs.env_helper import make_env
from fre.common.wandb import setup_wandb, default_wandb_config, get_flag_dict
from fre.common.evaluation import evaluate
from fre.common.utils import supply_rng
from fre.common.typing import *
from fre.common.train_state import TrainState, target_update
from fre.common.networks.basic import Policy, ValueCritic, Critic, ensemblize


###############################
#  Configs
###############################


FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'gc-antmaze-large-diverse-v2', 'Environment name.')
flags.DEFINE_string('name', 'default', '')

flags.DEFINE_string('save_dir', None, 'Logging dir (if not None, save params).')

flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('eval_episodes', 20,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 250000, 'Eval interval.')
flags.DEFINE_integer('video_interval', 50000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('goal_conditioned', 0, 'Whether to use goal conditioned relabelling or not.')

# These variables are passed to the BCAgent class.
agent_config = ml_collections.ConfigDict({
    'actor_lr': 3e-4,
    'hidden_dims': (512, 512, 512),
    'opt_decay_schedule': 'none',
    'use_tanh': 0,
    'state_dependent_std': 0,
    'use_layer_norm': 1,
})

wandb_config = default_wandb_config()
wandb_config.update({
    'project': 'mujoco_rlalgs',
    'name': 'bc_{env_name}',
})


config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('agent', agent_config, lock_config=False)
config_flags.DEFINE_config_dict('gcdataset', GCDataset.get_default_config(), lock_config=False)

###############################
#  Agent. Contains the neural networks, training logic, and sampling.
###############################

class BCAgent(flax.struct.PyTreeNode):
    rng: PRNGKey
    actor: TrainState
    config: dict = flax.struct.field(pytree_node=False)

    @jax.jit
    def update(agent, batch: Batch) -> InfoDict:
        observations = batch['observations']
        actions = batch['actions']

        def actor_loss_fn(actor_params):
            dist = agent.actor(observations, params=actor_params)
            log_probs = dist.log_prob(actions)
            actor_loss = -(log_probs).mean()

            mse_error = jnp.square(dist.loc - actions).mean()

            return actor_loss, {
                'actor_loss': actor_loss,
                'action_std': dist.stddev().mean(),
                'mse_error': mse_error,
            }
    
        new_actor, actor_info = agent.actor.apply_loss_fn(loss_fn=actor_loss_fn, has_aux=True)

        return agent.replace(actor=new_actor), {
            **actor_info
        }

    @jax.jit
    def sample_actions(agent,
                       observations: np.ndarray,
                       *,
                       seed: PRNGKey,
                       temperature: float = 1.0) -> jnp.ndarray:
        if type(observations) is dict:
            observations = jnp.concatenate([observations['observation'], observations['goal']], axis=-1)
        actions = agent.actor(observations, temperature=temperature).sample(seed=seed)
        actions = jnp.clip(actions, -1, 1)
        return actions

def create_agent(
                seed: int,
                observations: jnp.ndarray,
                actions: jnp.ndarray,
                actor_lr: float,
                use_tanh: bool,
                state_dependent_std: bool,
                use_layer_norm: bool,
                hidden_dims: Sequence[int],
                opt_decay_schedule: str,
                max_steps: Optional[int] = None,
            **kwargs):

        print('Extra kwargs:', kwargs)

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = Policy(hidden_dims, action_dim=action_dim, 
            log_std_min=-5.0, state_dependent_std=state_dependent_std, tanh_squash_distribution=use_tanh, mlp_kwargs=dict(use_layer_norm=use_layer_norm))

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            actor_tx = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            actor_tx = optax.adam(learning_rate=actor_lr)

        actor_params = actor_def.init(actor_key, observations)['params']
        actor = TrainState.create(actor_def, actor_params, tx=actor_tx)

        config = flax.core.FrozenDict(dict(
            actor_lr=actor_lr,
        ))

        return BCAgent(rng, actor=actor, config=config)

###############################
#  Run Script. Loads data, logs to wandb, and runs the training loop.
###############################

def main(_):
    if FLAGS.goal_conditioned:
        assert 'gc' in FLAGS.env_name
    else:
        assert 'gc' not in FLAGS.env_name

    # Create wandb logger
    setup_wandb(FLAGS.agent.to_dict(), **FLAGS.wandb)
    
    env = make_env(FLAGS.env_name)
    eval_env = make_env(FLAGS.env_name)

    dataset = d4rl_utils.get_dataset(env, FLAGS.env_name)
    dataset = d4rl_utils.normalize_dataset(FLAGS.env_name, dataset)
    if FLAGS.goal_conditioned:
        dataset = GCDataset(dataset, **FLAGS.gcdataset.to_dict())
        example_batch = dataset.sample(1)
        example_obs = np.concatenate([example_batch['observations'], example_batch['goals']], axis=-1)
        debug_batch = dataset.sample(100)
        print("Masks Look Like", debug_batch['masks'])
        print("Rewards Look Like", debug_batch['rewards'])
    else:
        example_obs = dataset.sample(1)['observations']

    agent = create_agent(FLAGS.seed,
                    example_obs,
                    example_batch['actions'],
                    max_steps=FLAGS.max_steps,
                    **FLAGS.agent)
    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):

        batch = dataset.sample(FLAGS.batch_size)
        if FLAGS.goal_conditioned:
            batch['observations'] = np.concatenate([batch['observations'], batch['goals']], axis=-1)
            batch['next_observations'] = np.concatenate([batch['next_observations'], batch['goals']], axis=-1)

        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            wandb.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            record_video = i % FLAGS.video_interval == 0
            policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
            eval_info, trajs = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes, record_video=record_video, return_trajectories=True)
            eval_metrics = {}
            for k in ['episode.return', 'episode.length']:
                eval_metrics[f'evaluation/{k}'] = eval_info[k]
                print(f'evaluation/{k}: {eval_info[k]}')
            eval_metrics['evaluation/episode.return.normalized'] = eval_env.get_normalized_score(eval_info['episode.return'])
            print(f'evaluation/episode.return.normalized: {eval_metrics["evaluation/episode.return.normalized"]}')
            if record_video:
                wandb.log({'video': eval_info['video']}, step=i)

            # Antmaze Specific Logging
            if 'antmaze-large' in FLAGS.env_name or 'maze2d-large' in FLAGS.env_name:
                import fre.common.envs.d4rl.d4rl_ant as d4rl_ant
                # Make an image of the trajectories.
                traj_image = d4rl_ant.trajectory_image(eval_env, trajs)
                eval_metrics['trajectories'] = wandb.Image(traj_image)

            wandb.log(eval_metrics, step=i)

if __name__ == '__main__':
    app.run(main)