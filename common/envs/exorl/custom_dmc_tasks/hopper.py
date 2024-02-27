"""Hopper domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import typing as tp
from typing import Any, Tuple

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import io as resources

_CONTROL_TIMESTEP: float
_DEFAULT_TIME_LIMIT: int
_HOP_SPEED: int
_SPIN_SPEED: int
_STAND_HEIGHT: float

SUITE = containers.TaggedTasks()

_CONTROL_TIMESTEP = .02  # (Seconds)

# Default duration of an episode, in seconds.
_DEFAULT_TIME_LIMIT = 20

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.6

# Hopping speed above which hop reward is 1.
_HOP_SPEED = 2
_SPIN_SPEED = 5


def make(task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward: bool = False):
    task_kwargs = task_kwargs or {}
    if environment_kwargs is not None:
        task_kwargs = task_kwargs.copy()
        task_kwargs['environment_kwargs'] = environment_kwargs
    env = SUITE[task](**task_kwargs)
    env.task.visualize_reward = visualize_reward
    return env


def get_model_and_assets() -> Tuple[Any, Any]:
    """Returns a tuple containing the model XML string and a dict of assets."""
    root_dir = os.path.dirname(os.path.dirname(__file__))
    xml = resources.GetResource(
        os.path.join(root_dir, 'custom_dmc_tasks', 'hopper.xml'))
    return xml, common.ASSETS


@SUITE.add('benchmarking')
def hop_backward(time_limit: int = _DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns a Hopper that strives to hop forward."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Hopper(hopping=True, forward=False, flip=False, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


@SUITE.add('benchmarking')
def flip(time_limit: int = _DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns a Hopper that strives to hop forward."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Hopper(hopping=True, forward=True, flip=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


@SUITE.add('benchmarking')
def flip_backward(time_limit: int = _DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns a Hopper that strives to hop forward."""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Hopper(hopping=True, forward=False, flip=True, random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(physics,
                               task,
                               time_limit=time_limit,
                               control_timestep=_CONTROL_TIMESTEP,
                               **environment_kwargs)


class Physics(mujoco.Physics):
    """Physics simulation with additional features for the Hopper domain."""

    def height(self) -> Any:
        """Returns height of torso with respect to foot."""
        return (self.named.data.xipos['torso', 'z'] -
                self.named.data.xipos['foot', 'z'])

    def speed(self) -> Any:
        """Returns horizontal speed of the Hopper."""
        return self.named.data.sensordata['torso_subtreelinvel'][0]

    def touch(self) -> Any:
        """Returns the signals from two foot touch sensors."""
        return np.log1p(self.named.data.sensordata[['touch_toe',
                                                    'touch_heel']])

    def angmomentum(self) -> Any:
        """Returns the angular momentum of torso of the Cheetah about Y axis."""
        return self.named.data.subtree_angmom['torso'][1]


class Hopper(base.Task):
    """A Hopper's `Task` to train a standing and a jumping Hopper."""

    def __init__(self, hopping, forward=True, flip=False, random=None) -> None:
        """Initialize an instance of `Hopper`.

    Args:
      hopping: Boolean, if True the task is to hop forwards, otherwise it is to
        balance upright.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
        self._hopping = hopping
        self._forward = 1 if forward else -1
        self._flip = flip
        self._timeout_progress = 0
        super(Hopper, self).__init__(random=random)

    def initialize_episode(self, physics) -> None:
        """Sets the state of the environment at the start of each episode."""
        randomizers.randomize_limited_and_rotational_joints(
            physics, self.random)
        self._timeout_progress = 0
        super(Hopper, self).initialize_episode(physics)

    def get_observation(self, physics) -> tp.Dict[str, Any]:
        """Returns an observation of positions, velocities and touch sensors."""
        obs = collections.OrderedDict()
        # Ignores horizontal position to maintain translational invariance:
        obs['position'] = physics.data.qpos[1:].copy()
        obs['velocity'] = physics.velocity()
        obs['touch'] = physics.touch()
        return obs

    def get_reward(self, physics) -> Any:
        """Returns a reward applicable to the performed task."""
        standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
        assert self._hopping
        if self._flip:
            hopping = rewards.tolerance(self._forward * physics.angmomentum(),
                                        bounds=(_SPIN_SPEED, float('inf')),
                                        margin=_SPIN_SPEED,
                                        value_at_margin=0,
                                        sigmoid='linear')
        else:
            hopping = rewards.tolerance(self._forward * physics.speed(),
                                        bounds=(_HOP_SPEED, float('inf')),
                                        margin=_HOP_SPEED / 2,
                                        value_at_margin=0.5,
                                        sigmoid='linear')
        return standing * hopping
