import matplotlib
matplotlib.use('Agg')
from matplotlib import patches

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from functools import partial
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import os.path as osp

import gym
import d4rl
import numpy as np
import functools as ft
import math
import matplotlib.gridspec as gridspec

from fre.common.envs.gc_utils import GCDataset

class MazeWrapper(gym.Wrapper):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env.render(mode='rgb_array', width=200, height=200)
        self.env_name = env_name
        self.inner_env = get_inner_env(self.env)
        if 'antmaze' in env_name:
            if 'medium' in env_name:
                self.env.viewer.cam.lookat[0] = 10
                self.env.viewer.cam.lookat[1] = 10
                self.env.viewer.cam.distance = 40
                self.env.viewer.cam.elevation = -90
            elif 'umaze' in env_name:
                self.env.viewer.cam.lookat[0] = 4
                self.env.viewer.cam.lookat[1] = 4
                self.env.viewer.cam.distance = 30
                self.env.viewer.cam.elevation = -90
            elif 'large' in env_name:
                self.env.viewer.cam.lookat[0] = 18
                self.env.viewer.cam.lookat[1] = 13
                self.env.viewer.cam.distance = 55
                self.env.viewer.cam.elevation = -90
            self.inner_env.goal_sampler = ft.partial(valid_goal_sampler, self.inner_env)
        elif 'maze2d' in env_name:
            if 'open' in env_name:
                pass
            elif 'large' in env_name:
                self.env.viewer.cam.lookat[0] = 5
                self.env.viewer.cam.lookat[1] = 6.5
                self.env.viewer.cam.distance = 15
                self.env.viewer.cam.elevation = -90
                self.env.viewer.cam.azimuth = 180
            self.draw_ant_maze = get_inner_env(gym.make('antmaze-large-diverse-v2'))
        self.action_space = self.env.action_space

    def render(self, *args, **kwargs):
        img = self.env.render(*args, **kwargs)
        if 'maze2d' in self.env_name:
            img = img[::-1]
        return img
    
    # ======== BELOW is helper stuff for drawing and visualizing ======== #

    def get_starting_boundary(self):
        if 'antmaze' in self.env_name:
            self = self.inner_env
        else:
            self = self.draw_ant_maze
        torso_x, torso_y = self._init_torso_x, self._init_torso_y
        S =  self._maze_size_scaling
        return (0 - S / 2 + S - torso_x, 0 - S/2 + S - torso_y), (len(self._maze_map[0]) * S - torso_x - S/2 - S, len(self._maze_map) * S - torso_y - S/2 - S)

    def XY(self, n=20, m=30):
        bl, tr = self.get_starting_boundary()
        X = np.linspace(bl[0] + 0.04 * (tr[0] - bl[0]) , tr[0] - 0.04 * (tr[0] - bl[0]), m)
        Y = np.linspace(bl[1] + 0.04 * (tr[1] - bl[1]) , tr[1] - 0.04 * (tr[1] - bl[1]), n)
        
        X,Y = np.meshgrid(X,Y)
        states = np.array([X.flatten(), Y.flatten()]).T
        return states

    def four_goals(self):
        self = self.inner_env

        valid_cells = []
        goal_cells = []

        for i in range(len(self._maze_map)):
            for j in range(len(self._maze_map[0])):
                if self._maze_map[i][j] in [0, 'r', 'g']:
                    valid_cells.append(self._rowcol_to_xy((i, j), add_random_noise=False))
        
        goals = []
        goals.append(max(valid_cells, key=lambda x: -x[0]-x[1]))
        goals.append(max(valid_cells, key=lambda x: x[0]-x[1]))
        goals.append(max(valid_cells, key=lambda x: x[0]+x[1]))
        goals.append(max(valid_cells, key=lambda x: -x[0] + x[1]))
        return goals
    
    def draw(self, ax=None, scale=1.0):
        if not ax: ax = plt.gca()
        if 'antmaze' in self.env_name:
            self = self.inner_env
        else:
            self = self.draw_ant_maze
        torso_x, torso_y = self._init_torso_x, self._init_torso_y
        S =  self._maze_size_scaling
        if scale < 1.0:
            S *= 0.965
            torso_x -= 0.7
            torso_y -= 0.95
        for i in range(len(self._maze_map)):
            for j in range(len(self._maze_map[0])):
                struct = self._maze_map[i][j]
                if struct == 1:
                    rect = patches.Rectangle((j *S - torso_x - S/ 2,
                                            i * S- torso_y - S/ 2),
                                            S,
                                            S, linewidth=1, edgecolor='none', facecolor='grey', alpha=1.0)

                    ax.add_patch(rect)
        ax.set_xlim(0 - S /2 + 0.6 * S - torso_x, len(self._maze_map[0]) * S - torso_x - S/2 - S * 0.6)
        ax.set_ylim(0 - S/2 + 0.6 * S - torso_y, len(self._maze_map) * S - torso_y - S/2 - S * 0.6)
        ax.axis('off')

class CenteredMaze(MazeWrapper):
    start_loc: str = "center"

    def __init__(self, env_name, start_loc="center"):
        super().__init__(env_name)
        self.start_loc = start_loc
        self.t = 0

    def step(self, action):
        next_obs, r, done, info = self.env.step(action)
        if 'antmaze' in self.env_name:
            info['x'], info['y'] = self.get_xy()
        self.t += 1
        done = self.t >= 2000
        return next_obs, r, done, info

    def reset(self, **kwargs):
        self.t = 0
        obs = self.env.reset(**kwargs)
        if 'maze2d' in self.env_name:
            if self.start_loc == 'center' or self.start_loc == 'center2':
                obs = self.env.reset_to_location([4, 5.8])
            elif self.start_loc == 'original':
                obs = self.env.reset_to_location([0.9, 0.9])
            else:
                raise NotImplementedError
        elif 'antmaze' in self.env_name:
            if self.start_loc == 'center' or self.start_loc == 'center2':
                self.env.set_xy([20, 15])
                obs[:2] = [20, 15]
            elif self.start_loc == 'original':
                pass
            else:
                raise NotImplementedError
        return obs

class GoalReachingMaze(MazeWrapper):
    def __init__(self, env_name):
        super().__init__(env_name)
        self.observation_space = gym.spaces.Dict({
            'observation': self.env.observation_space,
            'goal': self.env.observation_space,
        })

    def step(self, action):
        next_obs, r, done, info = self.env.step(action)
        
        if 'antmaze' in self.env_name:
            achieved = self.get_xy()
            desired = self.target_goal
        elif 'maze2d' in self.env_name:
            achieved = next_obs[:2]
            desired = self.env.get_target()
        distance = np.linalg.norm(achieved - desired)
        info['x'], info['y'] = achieved
        info['achieved_goal'] = np.array(achieved)
        info['desired_goal'] = np.copy(desired)
        info['success'] = float(distance < 0.5)
        done = 'TimeLimit.truncated' in info or info['success']
        
        return self.get_obs(next_obs), r, done, info
        
    def get_obs(self, obs):
        if 'antmaze' in self.env_name:
            desired = self.target_goal
        elif 'maze2d' in self.env_name:
            desired = self.env.get_target()
        target_goal = obs.copy()
        target_goal[:2] = desired
        if 'antmaze' in self.env_name:
            obs = discretize_obs(obs)
            target_goal = discretize_obs(target_goal)
        return dict(observation=obs, goal=target_goal)
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if 'maze2d' in self.env_name:
            obs = self.env.reset_to_location([0.9, 0.9])
        return self.get_obs(obs)
    
    def get_normalized_score(self, score):
        return score
    
# ===================================
# HELPER FUNCTIONS FOR OB DISCRETIZATION
# ===================================

def discretize_obs(ob, num_bins=32, disc_type='tanh', disc_temperature=1.0):
    min_ob = np.array([0, 0])
    max_ob = np.array([35, 35])
    disc_dims = 2
    bins = np.linspace(min_ob, max_ob, num_bins).T # [num_bins,] values from min_ob to max_ob
    bin_size = (max_ob - min_ob) / num_bins
    if disc_type == 'twohot':
        raise NotImplementedError
    elif disc_type == 'tanh':
        orig_ob = ob
        ob = np.expand_dims(ob, -1)
        # Convert each discretized dimension into num_bins dimensions. Value of each dimension is tanh of the distance from the bin center.
        bin_diff = ob[..., :disc_dims, :] - bins[:disc_dims]
        bin_diff_normalized = bin_diff / np.expand_dims(bin_size[:disc_dims], -1) * disc_temperature
        bin_tanh = np.tanh(bin_diff_normalized).reshape(*orig_ob.shape[:-1], -1)
        disc_ob = np.concatenate([bin_tanh, orig_ob[..., disc_dims:]], axis=-1)
        return disc_ob
    else:
        raise NotImplementedError

# ===================================
# HELPER FUNCTIONS FOR VISUALIZATION
# ===================================
    
def get_canvas_image(canvas):
    canvas.draw() 
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(canvas.get_width_height()[::-1] + (3,))
    return out_image

def valid_goal_sampler(self, np_random):
    valid_cells = []
    goal_cells = []
    # print('Hello')

    for i in range(len(self._maze_map)):
      for j in range(len(self._maze_map[0])):
        if self._maze_map[i][j] in [0, 'r', 'g']:
          valid_cells.append((i, j))

    # If there is a 'goal' designated, use that. Otherwise, any valid cell can
    # be a goal.
    sample_choices = valid_cells
    cell = sample_choices[np_random.choice(len(sample_choices))]
    xy = self._rowcol_to_xy(cell, add_random_noise=True)

    random_x = np.random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling
    random_y = np.random.uniform(low=0, high=0.5) * 0.25 * self._maze_size_scaling

    xy = (max(xy[0] + random_x, 0), max(xy[1] + random_y, 0))

    return xy


def get_inner_env(env):
    if hasattr(env, '_maze_size_scaling'):
        return env
    elif hasattr(env, 'env'):
        return get_inner_env(env.env)
    elif hasattr(env, 'wrapped_env'):
        return get_inner_env(env.wrapped_env)
    return env


# ===================================
# PLOT VALUE FUNCTION
# ===================================

def value_image(env, dataset, value_fn):
    """
    Visualize the value function.
    Args:
        env: The environment.
        value_fn: a function with signature value_fn([# states, state_dim]) -> [#states, 1]
    Returns:
        A numpy array of the image.
    """
    fig, axs = plt.subplots(2, 2, tight_layout=True)
    axs_flat = axs.flatten()
    canvas = FigureCanvas(fig)
    if type(dataset) is GCDataset:
        dataset = dataset.dataset
    if 'antmaze' in env.env_name:
        goals = env.four_goals()
        goal_states = dataset['observations'][0]
        goal_states = goal_states[-29:] # Remove discretized observations.
        goal_states = np.tile(goal_states, (len(goals), 1))
        goal_states[:, :2] = goals
        goal_states = discretize_obs(goal_states)
    elif 'maze2d' in env.env_name:
        goals = np.array([[0.8, 0.8], [1, 9.7], [6.8, 9], [6.8, 1]])
        goal_states = dataset['observations'][0]
        goal_states = np.tile(goal_states, (len(goals), 1))
        goal_states[:, :2] = goals
    for i in range(4):
        plot_value(goal_states[i], env, dataset, value_fn, axs_flat[i])
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def plot_value(goal_observation, env, dataset, value_fn, ax):
    N = 14
    M = 20
    ob_xy = env.XY(n=N, m=M)

    goal_observation = np.tile(goal_observation, (ob_xy.shape[0], 1)) # (N*M, 29)

    base_observation = np.copy(dataset['observations'][0])
    xy_observations = np.tile(base_observation, (ob_xy.shape[0], 1)) # (N*M, 29)
    if 'antmaze' in env.env_name:
        xy_observations = xy_observations[:, -29:] # Remove discretized observations.
        xy_observations[:, :2] = ob_xy # Set to XY.
        xy_observations = discretize_obs(xy_observations) # Discretize again.
        assert xy_observations.shape[1] == 91
    elif 'maze2d' in env.env_name:
        ob_xy_scaled = ob_xy / 3.5
        ob_xy_scaled = ob_xy_scaled[:, [1, 0]]
        xy_observations[:, :2] = ob_xy_scaled
        assert xy_observations.shape[1] == 4 # (x, y, vx, vy)
    values = value_fn(xy_observations, goal_observation) # (N*M, 1)

    x, y = ob_xy[:, 0], ob_xy[:, 1]
    x = x.reshape(N, M)
    y = y.reshape(N, M) * 0.975 + 0.7
    values = values.reshape(N, M)
    mesh = ax.pcolormesh(x, y, values, cmap='viridis')

    env.draw(ax, scale=0.95)


# ===================================
# PLOT TRAJECTORIES
# ===================================
    
# Makes an image of the trajectory the Ant follows.
def trajectory_image(env, trajectories, **kwargs):
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)

    plot_trajectories(env, trajectories, fig, plt.gca(), **kwargs)

    plt.tight_layout()
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

# Helper that plots the XY coordinates as scatter plots.
def plot_trajectories(env, trajectories, fig, ax, color_list=None):
    if color_list is None:
        from itertools import cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        color_list = cycle(color_cycle)

    for color, trajectory in zip(color_list, trajectories):
        obs = np.array(trajectory['observation'])

        # convert back to xy?
        if 'ant' in env.env_name:
            all_x = []
            all_y = []
            for info in trajectory['info']:
                all_x.append(info['x'])
                all_y.append(info['y'])
            all_x = np.array(all_x)
            all_y = np.array(all_y)
        elif 'maze2d' in env.env_name:
            all_x = obs[:, 1] * 4 - 3.2
            all_y = obs[:, 0] * 4 - 3.2
        ax.scatter(all_x, all_y, s=5, c=color, alpha=0.2)
        ax.scatter(all_x[-1], all_y[-1], s=50, c=color, marker='*', alpha=1, edgecolors='black')

    env.draw(ax)