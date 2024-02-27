import unittest
import dataclasses
from collections import OrderedDict, deque
import typing as tp
from typing import Any

from gym import core, spaces
from dm_env import Environment
from dm_env import StepType, specs
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels
import fre.common.envs.exorl.custom_dmc_tasks as cdmc
import numpy as np


S = tp.TypeVar("S", bound="TimeStep")
Env = tp.Union["EnvWrapper", Environment]


@dataclasses.dataclass
class TimeStep:
    step_type: StepType
    reward: float
    discount: float
    observation: np.ndarray
    physics: np.ndarray = dataclasses.field(default=np.ndarray([]), init=False)

    def first(self) -> bool:
        return self.step_type == StepType.FIRST  # type: ignore

    def mid(self) -> bool:
        return self.step_type == StepType.MID  # type: ignore

    def last(self) -> bool:
        return self.step_type == StepType.LAST  # type: ignore

    def __getitem__(self, attr: str) -> tp.Any:
        return getattr(self, attr)

    def _replace(self: S, **kwargs: tp.Any) -> S:
        for name, val in kwargs.items():
            setattr(self, name, val)
        return self


@dataclasses.dataclass
class ExtendedTimeStep(TimeStep):
    action: tp.Any


class EnvWrapper:
    def __init__(self, env: Env) -> None:
        self._env = env

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if not isinstance(time_step, TimeStep):
            # dm_env time step is a named tuple
            time_step = TimeStep(**time_step._asdict())
        if self.physics is not None:
            return time_step._replace(physics=self.physics.get_state())
        else:
            return time_step

    def reset(self) -> TimeStep:
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action: np.ndarray) -> TimeStep:
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def observation_spec(self) -> tp.Any:
        assert isinstance(self, EnvWrapper)
        return self._env.observation_spec()

    def action_spec(self) -> specs.Array:
        return self._env.action_spec()

    def render(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        return self._env.render(*args, **kwargs)  # type: ignore

    @property
    def base_env(self) -> tp.Any:
        env = self._env
        if isinstance(env, EnvWrapper):
            return self.base_env
        return env

    @property
    def physics(self) -> tp.Any:
        if hasattr(self._env, "physics"):
            return self._env.physics

    def __getattr__(self, name):
        return getattr(self._env, name)


class FlattenJacoObservationWrapper(EnvWrapper):
    def __init__(self, env: Env) -> None:
        super().__init__(env)
        self._obs_spec = OrderedDict()
        wrapped_obs_spec = env.observation_spec().copy()
        if 'front_close' in wrapped_obs_spec:
            spec = wrapped_obs_spec['front_close']
            # drop batch dim
            self._obs_spec['pixels'] = specs.BoundedArray(shape=spec.shape[1:],
                                                          dtype=spec.dtype,
                                                          minimum=spec.minimum,
                                                          maximum=spec.maximum,
                                                          name='pixels')
            wrapped_obs_spec.pop('front_close')

        for spec in wrapped_obs_spec.values():
            assert spec.dtype == np.float64
            assert type(spec) == specs.Array
        dim = np.sum(
            np.fromiter((int(np.prod(spec.shape))  # type: ignore
                         for spec in wrapped_obs_spec.values()), np.int32))

        self._obs_spec['observations'] = specs.Array(shape=(dim,),
                                                     dtype=np.float32,
                                                     name='observations')

    def observation_spec(self) -> tp.Any:
        return self._obs_spec

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        super()._augment_time_step(time_step=time_step, action=action)
        obs = OrderedDict()

        # TODO: this is badly typed since observation is a dict in this case
        if 'front_close' in time_step.observation:
            pixels = time_step.observation['front_close']
            time_step.observation.pop('front_close')  # type: ignore
            pixels = np.squeeze(pixels)
            obs['pixels'] = pixels

        features = []
        for feature in time_step.observation.values():  # type: ignore
            features.append(feature.ravel())
        obs['observations'] = np.concatenate(features, axis=0)
        return time_step._replace(observation=obs)


class ActionRepeatWrapper(EnvWrapper):
    def __init__(self, env: tp.Any, num_repeats: int) -> None:
        super().__init__(env)
        self._num_repeats = num_repeats

    def step(self, action: np.ndarray) -> TimeStep:
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)


class FrameStackWrapper(EnvWrapper):
    def __init__(self, env: Env, num_frames: int, pixels_key: str = 'pixels') -> None:
        super().__init__(env)
        self._num_frames = num_frames
        self._frames: tp.Deque[np.ndarray] = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation')

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        super()._augment_time_step(time_step=time_step, action=action)
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step: TimeStep) -> np.ndarray:
        pixels_ = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels_.shape) == 4:
            pixels_ = pixels_[0]
        return pixels_.transpose(2, 0, 1).copy()

    def reset(self) -> TimeStep:
        time_step = self._env.reset()
        pixels_ = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels_)
        return self._augment_time_step(time_step)

    def step(self, action: np.ndarray) -> TimeStep:
        time_step = self._env.step(action)
        pixels_ = self._extract_pixels(time_step)
        self._frames.append(pixels_)
        return self._augment_time_step(time_step)


class ActionDTypeWrapper(EnvWrapper):
    def __init__(self, env: Env, dtype) -> None:
        super().__init__(env)
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def action_spec(self) -> specs.BoundedArray:
        return self._action_spec

    def step(self, action) -> Any:
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)


class ObservationDTypeWrapper(EnvWrapper):
    def __init__(self, env: Env, dtype) -> None:
        super().__init__(env)
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()['observations']
        self._obs_spec = specs.Array(wrapped_obs_spec.shape, dtype,
                                     'observation')

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        obs = time_step.observation['observations'].astype(self._dtype)
        return time_step._replace(observation=obs)

    def observation_spec(self) -> Any:
        return self._obs_spec


class ExtendedTimeStepWrapper(EnvWrapper):

    def _augment_time_step(self, time_step: TimeStep, action: tp.Optional[np.ndarray] = None) -> TimeStep:
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        ts = ExtendedTimeStep(observation=time_step.observation,
                              step_type=time_step.step_type,
                              action=action,
                              reward=time_step.reward or 0.0,
                              discount=time_step.discount or 1.0)
        return super()._augment_time_step(time_step=ts, action=action)


def _make_jaco(obs_type, domain, task, frame_stack, action_repeat, seed) -> FlattenJacoObservationWrapper:
    env = cdmc.make_jaco(task, obs_type, seed)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env)
    return env


def _make_dmc(obs_type, domain, task, frame_stack, action_repeat, seed):
    visualize_reward = False
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(domain,
                         task,
                         task_kwargs=dict(random=seed),
                         environment_kwargs=dict(flat_observation=True),
                         visualize_reward=visualize_reward)
    else:
        env = cdmc.make(domain,
                        task,
                        task_kwargs=dict(random=seed),
                        environment_kwargs=dict(flat_observation=True),
                        visualize_reward=visualize_reward)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    if obs_type == 'pixels':
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env,
                             pixels_only=True,
                             render_kwargs=render_kwargs)
    return env


def make(
    name: str, obs_type='states', frame_stack=1, action_repeat=1,
    seed=1,
) -> EnvWrapper:
    assert obs_type in ['states', 'pixels']
    if name.startswith('point_mass_maze'):
        domain = 'point_mass_maze'
        _, _, _, task = name.split('_', 3)
    else:
        domain, task = name.split('_', 1)
    domain = dict(cup='ball_in_cup').get(domain, domain)

    make_fn = _make_jaco if domain == 'jaco' else _make_dmc
    env = make_fn(obs_type, domain, task, frame_stack, action_repeat, seed)

    if obs_type == 'pixels':
        env = FrameStackWrapper(env, frame_stack)
    else:
        env = ObservationDTypeWrapper(env, np.float32)

    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = ExtendedTimeStepWrapper(env)
    return env


def extract_physics(env: Env) -> tp.Dict[str, float]:
    """Extract some physics available in the env"""
    output = {}
    names = ["torso_height", "torso_upright", "horizontal_velocity", "torso_velocity"]
    for name in names:
        if not hasattr(env.physics, name):
            continue
        val: tp.Union[float, np.ndarray] = getattr(env.physics, name)()
        if isinstance(val, (int, float)) or not val.ndim:
            output[name] = float(val)
        else:
            for k, v in enumerate(val):
                output[f"{name}#{k}"] = float(v)
    return output


class FloatStats:
    """Handle for keeping track of the statistics of a float variable"""

    def __init__(self) -> None:
        self.min = np.inf
        self.max = -np.inf
        self.mean = 0.0
        self._count = 0

    def add(self, value: float) -> "FloatStats":
        self.min = min(value, self.min)
        self.max = max(value, self.max)
        self._count += 1
        self.mean = (self._count - 1) / self._count * self.mean + 1 / self._count * value
        return self

    def items(self) -> tp.Iterator[tp.Tuple[str, float]]:
        for name, val in self.__dict__.items():
            if not name.startswith("_"):
                yield name, val


class PhysicsAggregator:
    """Aggregate stats on the physics of an environment"""

    def __init__(self) -> None:
        self.stats: tp.Dict[str, FloatStats] = {}

    def add(self, env: Env) -> "PhysicsAggregator":
        phy = extract_physics(env)
        for key, val in phy.items():
            self.stats.setdefault(key, FloatStats()).add(val)
        return self

    def dump(self) -> tp.Iterator[tp.Tuple[str, float]]:
        """Exports all statistics and reset the statistics"""
        for key, stats in self.stats.items():
            for stat, val in stats.items():
                yield (f'{key}/{stat}', val)
        self.stats.clear()


def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)


def _flatten_obs(obs):
    obs_pieces = []
    v = obs
    flat = np.array([v]) if np.isscalar(v) else v.ravel()
    obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper(core.Env):
    def __init__(
            self,
            env,
            seed,
            from_pixels=False,
            height=84,
            width=84,
            camera_id=0,
            frame_skip=1,
            channels_first=True,
    ):
        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._camera_id = camera_id
        self._frame_skip = frame_skip
        self._channels_first = channels_first

        self._env = env

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = [3, height, width] if channels_first else [height, width, 3]
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box([self._env.observation_spec()], np.float32)

        self._state_space = _spec_to_box([self._env.observation_spec()], np.float32)

        self.current_state = None

        self.seed(seed)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self._from_pixels:
            obs = self.render(
                height=self._height,
                width=self._width,
                camera_id=self._camera_id
            )
            if self._channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def reward_range(self):
        return 0, self._frame_skip

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {'internal_state': self._env.physics.get_state().copy()}

        for _ in range(self._frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra['discount'] = time_step.discount
        return obs, reward, done, extra

    def reset(self):
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        return obs

    def render(self, mode='rgb_array', height=None, width=None, camera_id=0):
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height = height or self._height
        width = width or self._width
        camera_id = camera_id or self._camera_id
        return self._env.physics.render(
            height=height, width=width, camera_id=camera_id
        )