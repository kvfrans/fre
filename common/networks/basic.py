###############################
#
#  Common Flax Networks.
#
###############################

from fre.common.typing import *

import flax.linen as nn
import jax.numpy as jnp

import distrax
import flax.linen as nn
import jax.numpy as jnp
from dataclasses import field

###############################
#
#  Common Networks
#
###############################

def mish(x):
    return x * jnp.tanh(nn.softplus(x))

def default_init(scale: Optional[float] = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = mish
    activate_final: int = False
    use_layer_norm: bool = True
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    def setup(self):
        self.layers = [
            nn.Dense(size, kernel_init=self.kernel_init) for size in self.hidden_dims
        ]
        if self.use_layer_norm:
            self.layer_norms = [nn.LayerNorm() for _ in self.hidden_dims]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i + 1 < len(self.layers) and self.use_layer_norm:
                x = self.layer_norms[i](x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = self.activations(x)
        return x

###############################
#
#  Common RL Networks
#
###############################


# DQN-style critic.
class DiscreteCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_actions: int
    mlp_kwargs: Dict[str, Any] = field(default_factory=dict)

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return MLP((*self.hidden_dims, self.n_actions), **self.mlp_kwargs)(
            observations
        )

# Q(s,a) critic.
class Critic(nn.Module):
    hidden_dims: Sequence[int]
    mlp_kwargs: Dict[str, Any] = field(default_factory=dict)

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1), **self.mlp_kwargs)(inputs)
        return jnp.squeeze(critic, -1)

# V(s) critic.
class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    mlp_kwargs: Dict[str, Any] = field(default_factory=dict)

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1), **self.mlp_kwargs)(observations)
        return jnp.squeeze(critic, -1)

# pi(a|s). Returns a distrax distribution.
class Policy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    mlp_kwargs: Dict[str, Any] = field(default_factory=dict)

    is_discrete: bool = False
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2
    mean_min: Optional[float] = -5
    mean_max: Optional[float] = 5
    tanh_squash_distribution: bool = False
    state_dependent_std: bool = True
    final_fc_init_scale: float = 1e-2

    @nn.compact
    def __call__(
        self, observations: jnp.ndarray, temperature: float = 1.0
    ) -> distrax.Distribution:
        outputs = MLP(
            self.hidden_dims,
            activate_final=True,
            **self.mlp_kwargs
        )(observations)

        if self.is_discrete:
            logits = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )(outputs)
            distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))
        else:
            means = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )(outputs)
            if self.state_dependent_std:
                log_stds = nn.Dense(
                    self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
                )(outputs)
            else:
                log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

            log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
            means = jnp.clip(means, self.mean_min, self.mean_max)

            distribution = distrax.MultivariateNormalDiag(
                loc=means, scale_diag=jnp.exp(log_stds) * temperature
            )
            if self.tanh_squash_distribution:
                distribution = TransformedWithMode(
                    distribution, distrax.Block(distrax.Tanh(), ndims=1)
                )
        return distribution
    
###############################
#
#   Helper Things
#
###############################


class TransformedWithMode(distrax.Transformed):
    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())
    
def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """
    Useful for making ensembles of Q functions (e.g. double Q in SAC).

    Usage:

        critic_def = ensemblize(Critic, 2)(hidden_dims=hidden_dims)

    """
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs
    )