import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def get_canvas_image(canvas):
    canvas.draw() 
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(canvas.get_width_height()[::-1] + (3,))
    return out_image

def value_image(env, dataset, value_fn, mask, clip=False):
    """
    Visualize the value function.
    Args:
        env: The environment.
        value_fn: a function with signature value_fn([# states, state_dim]) -> [#states, 1]
    Returns:
        A numpy array of the image.
    """
    fig = plt.figure(tight_layout=True)
    canvas = FigureCanvas(fig)
    plot_value(env, dataset, value_fn, mask, fig, plt.gca(), clip=clip)
    image = get_canvas_image(canvas)
    plt.close(fig)
    return image

def plot_value(env, dataset, value_fn, mask, fig, ax, title=None, clip=True):
    N = 14
    M = 20
    ob_xy = env.XY(n=N, m=M)

    base_observation = np.copy(dataset['observations'][0])
    base_observations = np.tile(base_observation, (5, ob_xy.shape[0], 1))
    base_observations[:, :, :2] = ob_xy
    base_observations[:, :, 15:17] = 0.0
    base_observations[0, :, 15] = 1.0
    base_observations[1, :, 16] = 1.0
    base_observations[2, :, 15] = -1.0
    base_observations[3, :, 16] = -1.0
    print("Base observations, ", base_observations.shape)


    values = []
    for i in range(5):
        values.append(value_fn(base_observations[i]))
    values = np.stack(values, axis=0)
    print("Values", values.shape)

    x, y = ob_xy[:, 0], ob_xy[:, 1]
    x = x.reshape(N, M)
    y = y.reshape(N, M) * 0.975 + 0.7
    values = values.reshape(5, N, M)
    values[-1, 10, 0] = np.min(values[-1]) + 0.3 # Hack to make the scaling not show small errors.
    print("Clip:", clip)
    if clip:
        mesh = ax.pcolormesh(x, y, values[-1], cmap='viridis', vmin=-0.1, vmax=1.0)
    else:
        mesh = ax.pcolormesh(x, y, values[-1], cmap='viridis')

    v = (values[1] - values[3]) / 2
    u = (values[0] - values[2]) / 2
    uv_dist = np.sqrt(u**2 + v**2) + 1e-6
    # Normalize u,v
    un = u / uv_dist
    vn = v / uv_dist
    un[uv_dist < 0.1] = 0
    vn[uv_dist < 0.1] = 0
    
    plt.quiver(x, y, un, vn, color='r', pivot='mid', scale=0.75, scale_units='xy')

    if mask is not None and type(mask) == np.ndarray:
        # mask = NxM array of things to unmask.
        from matplotlib.colors import LinearSegmentedColormap
        colors = [(0,0,0,c) for c in np.linspace(0,1,100)]
        cmapred = LinearSegmentedColormap.from_list('mycmap', colors, N=5)
        mask_mesh_ax = ax.pcolormesh(x, y, mask, cmap=cmapred)
    elif mask is not None and type(mask) is list:
        maskmesh = np.ones((N, M))
        for xy in mask:
            for xi in range(N):
                for yi in range(M):
                    if np.linalg.norm(np.array(xy) - np.array([x[xi, yi], y[xi, yi]])) < 1.4:
                        # print(xy, x[xi, yi], y[xi, yi])
                        maskmesh[xi,yi] = 0
        from matplotlib.colors import LinearSegmentedColormap
        colors = [(0,0,0,c) for c in np.linspace(0,1,100)]
        cmapred = LinearSegmentedColormap.from_list('mycmap', colors, N=5)
        mask_mesh_ax = ax.pcolormesh(x, y, maskmesh, cmap=cmapred)

    env.draw(ax, scale=0.95)



    # env.draw(ax, scale=1.0)

    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes('right', size='5%', pad=0.05)
    # fig.colorbar(mesh, cax=cax, orientation='vertical')

    if title:
        ax.set_title(title)