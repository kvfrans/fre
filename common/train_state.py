###############################
#
#  Structures for managing training of flax networks.
#
###############################

from fre.common.typing import *
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import tree_util
import optax
import functools

import gym

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


def shard_batch(batch):
    d = jax.local_device_count()

    def reshape(x):
        assert (
            x.shape[0] % d == 0
        ), f"Batch size needs to be divisible by # devices, got {x.shape[0]} and {d}"
        return x.reshape((d, x.shape[0] // d, *x.shape[1:]))

    return tree_util.tree_map(reshape, batch)


def target_update(
    model: "TrainState", target_model: "TrainState", tau: float
) -> "TrainState":
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), model.params, target_model.params
    )
    return target_model.replace(params=new_target_params)


class TrainState(flax.struct.PyTreeNode):
    """
    Core abstraction of a model in this repository.

    Creation:
    ```
        model_def = nn.Dense(12) # or any other flax.linen Module
        params = model_def.init(jax.random.PRNGKey(0), jnp.ones((1, 4)))['params']
        model = TrainState.create(model_def, params, tx=None) # Optionally, pass in an optax optimizer
    ```

    Usage:
    ```
        y = model(jnp.ones((1, 4))) # By default, uses the `__call__` method of the model_def and params stored in TrainState
        y = model(jnp.ones((1, 4)), params=params) # You can pass in params (useful for gradient computation)
        y = model(jnp.ones((1, 4)), method=method) # You can apply a different method as well
    ```

    More complete example:
    ```
        def loss(params):
            y_pred = model(x, params=params)
            return jnp.mean((y - y_pred) ** 2)

        grads = jax.grad(loss)(model.params)
        new_model = model.apply_gradients(grads=grads) # Alternatively, new_model = model.apply_loss_fn(loss_fn=loss)
    ```
    """

    step: int
    apply_fn: Callable[..., Any] = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Params
    tx: Optional[optax.GradientTransformation] = nonpytree_field()
    opt_state: Optional[optax.OptState] = None

    @classmethod
    def create(
        cls,
        model_def: nn.Module,
        params: Params,
        tx: Optional[optax.GradientTransformation] = None,
        **kwargs,
    ) -> "TrainState":
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(
        self,
        *args,
        params=None,
        extra_variables: dict = None,
        method: ModuleMethod = None,
        **kwargs,
    ):
        """
        Internally calls model_def.apply_fn with the following logic:

        Arguments:
            params: If not None, use these params instead of the ones stored in the model.
            extra_variables: Additional variables to pass into apply_fn
            method: If None, use the `__call__` method of the model_def. If a string, uses
                the method of the model_def with that name (e.g. 'encode' -> model_def.encode).
                If a function, uses that function.

        """
        if params is None:
            params = self.params

        variables = {"params": params}

        if extra_variables is not None:
            variables = {**variables, **extra_variables}

        if isinstance(method, str):
            method = getattr(self.model_def, method)

        return self.apply_fn(variables, *args, method=method, **kwargs)
    
    def do(self, method):
        return functools.partial(self, method=method)

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
            grads: Gradients that have the same pytree structure as `.params`.
            **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
            An updated instance of `self` with `step` incremented by one, `params`
            and `opt_state` updated by applying `grads`, and additional attributes
            replaced as specified by `kwargs`.
        """
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, *, loss_fn, pmap_axis=None, has_aux=False):
        """
        Takes a gradient step towards minimizing `loss_fn`. Internally, this calls
        `jax.grad` followed by `TrainState.apply_gradients`. If pmap_axis is provided,
        additionally it averages gradients (and info) across devices before performing update.
        """
        if has_aux:
            grads, info = jax.grad(loss_fn, has_aux=has_aux)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
                info = jax.lax.pmean(info, axis_name=pmap_axis)

            return self.apply_gradients(grads=grads), info

        else:
            grads = jax.grad(loss_fn, has_aux=has_aux)(self.params)
            if pmap_axis is not None:
                grads = jax.lax.pmean(grads, axis_name=pmap_axis)
            return self.apply_gradients(grads=grads)

class NormalizeActionWrapper(gym.Wrapper):
    """A wrapper that maps actions from [-1,1] to [low, hgih]."""
    def __init__(self, env):
        super().__init__(env)
        self.active = type(env.action_space) == gym.spaces.Box
        if self.active:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_scale = (self.action_high - self.action_low) * 0.5
            self.action_mid = (self.action_high + self.action_low) * 0.5
            print("Normalizing Action Space from [{}, {}] to [-1, 1]".format(self.action_low[0], self.action_high[0]))
    def step(self, action):
        if self.active:
            action = np.clip(action, -1, 1)
            action = action * self.action_scale
            action = action + self.action_mid
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
