import numpy as np
from typing import Callable, Union
import matplotlib.pyplot as plt
from matplotlib import animation


# FOR ANIMATIONS AND CHAOS BOOK ANALYSIS STUFF


class System:
    def __init__(self, dx_dt: Callable[[np.array, float, dict], np.array], params: dict,
                 dim: int):
        self.dx_dt = dx_dt
        self.params = params
        self.t = 0
        self.dim = dim

    def evolve(self, X: np.array, h, h_tol):
        # return X + self.d_dt(X, self.t, self.params) * dt
        # Get the k values
        k1 = h * self.dx_dt(X, self.t, self.params)
        k2 = h * self.dx_dt(X + 1.0/4.0 * k1, self.t + 1.0/4.0 * h, self.params)
        k3 = h * self.dx_dt(X + 3.0/32.0 * k1 + 9.0/32.0 * k2, self.t + 3.0/8.0 * h, self.params)
        k4 = h * self.dx_dt(X + 1932.0/2197.0 * k1 - 7200.0/2197.0 * k2 + 7296.0/2197.0 * k3, self.t + 12.0/13.0 * h, self.params)
        k5 = h * self.dx_dt(X + 439.0/216.0 * k1 - 8 * k2 + 3680.0/513.0 * k3 - 845.0/4104.0 * k4, self.t + h, self.params)
        k6 = h * self.dx_dt(X - 8.0/27.0 * k1 + 2 * k2 - 3544.0/2565.0 * k3 + 1859.0/4104.0 * k4 - 11.0/40.0 * k5, self.t + 1.0/2.0 * h, self.params)

        # Get 4th and 5th order solutions
        x_k1 = X + 25.0/216.0 * k1 + 1408.0/2565.0 * k3 + 2197.0/4101.0 * k4 - 1.0/5.0 * k5
        z_k1 = X + 16.0/135.0 * k1 + 6656.0/12825.0 * k3 + 28561.0/56430.0 * k4 - 9.0/50.0 * k5 + 2.0/55.0 * k6

        # Get timestep from these solutions
        max_norm = max(abs(z_k1 - x_k1).flatten())
        s = np.pow(0.5, 0.25) * np.pow(h_tol / max_norm, 0.25)
        dt = s * h

        # Approximate step
        self.t += dt
        res = X + self.dx_dt(X, self.t, self.params) * dt
        return res




class DiscreteMap:
    def __init__(self, sys_map: Callable[[np.array, dict], np.array], params: dict,
                 dim: int):
        self.sys_map = sys_map
        self.params = params
        self.dim = dim

    def evolve(self, X: np.array):
        return self.sys_map(X, self.params)


def euler(sys: Union[System, DiscreteMap], X_0: np.array, dt=0.01, max_t=100, n=1000) -> Union[tuple, np.array]:
    dim = sys.dim
    X_0 = np.array(X_0, dtype=float)
    states = [[X_0[i] for i in range(dim)]]
    is_discrete = False
    if isinstance(sys, DiscreteMap):
        is_discrete = True
    if is_discrete:
        for i in range(n):
            cur_X = sys.evolve(states[-1])
            states.append(cur_X)
        return np.array(states)
    else:
        t = [0]
        while t[-1] + dt < max_t:
            cur_X = sys.evolve(states[-1], dt)
            states.append(cur_X)
            t.append(t[-1] + dt)

        return np.array(states), np.array(t)


def find_loc_maxima1D(x: np.array, t=None):
    maxima = []
    maxima_t = []
    for i in range(1, len(x) - 1):
        if x[i-1] < x[i] > x[i+1]:
            maxima.append(x[i])
            if t is not None:
                maxima_t.append([t[i]])
    if t is not None:
        return np.array(maxima), np.array(maxima_t)
    else:
        return np.array(maxima)


def lorenz_map(X: Union[list, np.array]):
    X_max = find_loc_maxima1D(np.array(X))
    return np.array([[X_max[i-1], X_max[i]] for i in range(1, len(X_max))]).T


def analyze_system(sys: Union[System, DiscreteMap], X_0: np.array, dt=0.01, max_t=100, n=1000, lorenz_dim=-1, model_name=None):
    fig = plt.figure()
    dim = sys.dim

    linewidth = 0.5

    if lorenz_dim == -1:
        if dim == 1 or dim == 2:
            axis_pres = fig.add_subplot(1, 1, 1)
        elif dim == 3:
            axis_pres = fig.add_subplot(1, 1, 1, projection='3d')
            axis_proj = fig.add_subplot(2, 2, 4)
    elif 0 <= lorenz_dim < dim:
        if dim == 1 or dim == 2:
            axis_pres = fig.add_subplot(2, 2, 1)
        elif dim == 3:
            axis_pres = fig.add_subplot(2, 2, 1, projection='3d')
            axis_proj = fig.add_subplot(2, 2, 4)
        ax_ndim = fig.add_subplot(2, 2, 2)
        ax_lorenz = fig.add_subplot(2, 2, 3)
    else:
        raise Exception('Invalid dimension!')

    is_discrete = False
    if isinstance(sys, DiscreteMap):
        is_discrete = True
        states = euler(sys, X_0, n=n).T
    else:
        states, t = euler(sys, X_0, dt=dt, max_t=max_t)
        states = states.T

    if dim == 1:
        if is_discrete:
            axis_pres.plot(states[0], linewidth=linewidth)
            axis_pres.set_xlabel('n')
        else:
            axis_pres.plot(t, states[0], linewidth=linewidth)
            axis_pres.set_xlabel('t')
        axis_pres.set_ylabel('x')
    elif dim == 2:
        axis_pres.plot(states[0], states[1], linewidth=linewidth)
        axis_pres.set_xlabel('x')
        axis_pres.set_ylabel('y')
    elif dim == 3:
        axis_pres.plot(states[0], states[1], states[2], linewidth=linewidth)
        axis_pres.set_xlabel('x')
        axis_pres.set_ylabel('y')
        axis_pres.set_zlabel('z')
        axis_proj.plot(states[0], states[1], linewidth=linewidth)
        axis_proj.set_xlabel('x')
        axis_proj.set_ylabel('y')

    if model_name is not None:
        axis_pres.set_title(model_name)

    if lorenz_dim != -1:
        dim_label = f'dim({lorenz_dim})'
        if is_discrete:
            ax_ndim.plot(states[lorenz_dim], label='$\mathregular{' + dim_label + '}$')
            ax_ndim.set_xlabel('n')
        else:
            ax_ndim.plot(t, states[lorenz_dim], label='$\mathregular{' + dim_label + '}$')
            ax_ndim.set_xlabel('t')
        maxima, t_max = find_loc_maxima1D(states[lorenz_dim], t)
        ax_ndim.plot(t_max, maxima, label='$\mathregular{' + dim_label + '^{max}}$')
        ax_ndim.set_ylabel('$\mathregular{' + dim_label + '}$')
        ax_ndim.legend()

        lorenz_dim_arr = lorenz_map(states[lorenz_dim])
        ax_lorenz.set_title('Lorenz Map')
        ax_lorenz.scatter(lorenz_dim_arr[0], lorenz_dim_arr[1], label='Lorenz Map', s=5)
        ax_lorenz.set_xlabel('$\mathregular{' + dim_label + '^{max}_n}$')
        ax_lorenz.set_ylabel('$\mathregular{' + dim_label + '^{max}_{n+1}}$')
        ax_lorenz.legend()

    plt.tight_layout()
    plt.show()
    return


anim_X = []
past_X = []


def figure_update(t, sys: System, ax, h, h_tol, projection, dims):
    global anim_X, past_X
    ax.cla()
    new_X = sys.evolve(anim_X, h, h_tol)
    past_X.append(new_X)
    past_arr = np.array(past_X).T
    xlims = dims[0]
    ylims = dims[1]
    if len(dims) > 2:
        zlims = dims[2]
    if projection is None:
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if len(dims) > 2:
            ax.set_zlim(zlims[0], zlims[1])
            ax.set_zlabel('z')
            ax.scatter(new_X[0], new_X[1], new_X[2], s=10)
        else:
            ax.scatter(new_X[0], new_X[1], s=10)
        for p in past_arr:    
            if len(dims) > 2:
                ax.plot(p[0][-1000:], p[1][-1000:], p[2][-1000:], c='cyan')
            else:
                ax.plot(p[0][-2000:], p[1][-2000:], c='cyan', linewidth=0.75)
    elif projection == 'xz':
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(zlims[0], zlims[1])
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.scatter(new_X[0], new_X[2], s=10)
        for p in past_arr:
            ax.plot(p[0][-1000:], p[2][-1000:], c='cyan', linewidth=1.0)
    elif projection == 'xy':
        ax.set_xlim(xlims[0], xlims[1])
        ax.set_ylim(ylims[0], ylims[1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(new_X[0], new_X[1], s=10)
        for p in past_arr:
            ax.plot(p[0][-1000:], p[1][-1000:], c='cyan', linewidth=1.0)
    elif projection == 'yz':
        ax.set_xlim(ylims[0], ylims[1])
        ax.set_ylim(zlims[0], zlims[1])
        ax.set_xlabel('y')
        ax.set_ylabel('z')
        ax.scatter(new_X[0], new_X[2], s=10)
        for p in past_arr:
            ax.plot(p[1][-1000:], p[2][-1000:], c='cyan', linewidth=1.0)
    anim_X = new_X



def show_spread(sys: Union[System, DiscreteMap], X_0: np.array, h=0.1, h_tol=1.0e-8, max_t=100, delta=0.5, dims=[(-30, 30), (-30, 30), (-50, 50)], dup_dim=2, projection=None):
    fig = plt.figure()
    if projection is None and len(dims) == 3:
        ax = fig.add_subplot(1, 1, 1, projection='3d')
    else:
        ax = fig.add_subplot(1, 1, 1)
    global anim_X
    anim_X = []
    for i in range(dup_dim):
        for j in range(dup_dim):
            if len(X_0[0]) > 2:
                for k in range(dup_dim):
                    for X in X_0:
                        anim_X.append([X[0] + np.random.uniform(-delta, delta), X[1] + np.random.uniform(-delta, delta), X[2] + np.random.uniform(-delta, delta)])
            else:
                for X in X_0:
                        anim_X.append([X[0] + np.random.uniform(-delta, delta), X[1] + np.random.uniform(-delta, delta)])
    
    anim_X = np.array(anim_X, dtype=float).T
    anim = animation.FuncAnimation(fig, figure_update, fargs=(sys, ax, h, h_tol, projection, dims),
                                   interval=1, blit=False)
    plt.show()
    plt.close()
    return


def n_dim_approx_LCE(sys: Union[System, DiscreteMap], X_0: np.array, delta: float, lce_dim: int, n=1000, dt=0.01, max_t=100):
    print(lce_dim)
    print(sys.dim)
    if not (0 <= lce_dim < sys.dim):
        raise Exception('Invalid LCE dimension')

    is_discrete = False

    if isinstance(sys, DiscreteMap):
        is_discrete = True

    cur_X = np.array(X_0, dtype=float)
    if not is_discrete:
        sum = 0
        dX = np.copy(cur_X)
        dX[lce_dim] += delta
        iter = 0
        t = 0
        while t < max_t:
            cur_X = sys.evolve(cur_X, dt)
            dX = sys.evolve(dX, dt)
            diff = dX - cur_X
            dist = np.sqrt(np.dot(diff, diff))
            sum += np.log(np.abs(dist / delta))
            iter += 1
            t += dt
        return sum / iter



