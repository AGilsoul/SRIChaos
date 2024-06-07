import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable
from n_dim_systems import *


def Log_Map(X: np.array, params: dict) -> np.array:
    r = params['r']
    x = X[0]
    return np.array([r * x * (1 - x)])


def run_Log():
    params = {'r': 3.2}
    sys = DiscreteMap(Log_Map, params, [0.2])
    analyze_system(sys, n=10)
    print(n_dim_approx_LCE(sys, 0.0001, 0, 1000))


def LotVolt_Sys(X: np.array, t: float, params: dict) -> np.array:
    x, y = X[0], X[1]
    a, b, c, d = params['a'], params['b'], params['c'], params['d']
    return np.array([(a * x) - (b * x * y), (c * x * y) - (d * y)])


def run_LotkaVolterra():
    params = {'a': 1,
              'b': 0.25,
              'c': 0.2,
              'd': 0.6}
    dt = 0.01
    max_t = 300
    sys = System(LotVolt_Sys, params, [10, 6])
    analyze_system(sys, dt, max_t)


def Rossler_Sys(X: np.array, t: float, params: dict) -> np.array:
    x, y, z = X[0], X[1], X[2]
    a, b, c = params['a'], params['b'], params['c']

    dx = -y - z
    dy = x + a * y
    dz = b + z * (x - c)

    return np.array([dx, dy, dz])


def run_Rossler():
    params = {'a': 0.1,
              'b': 0.1,
              'c': 10.0}

    dt = 0.01
    max_t = 300
    init_conditions = [5, 5, 1]
    sys = System(Rossler_Sys, params, 3)

    show_spread(sys, init_conditions)
    # analyze_system(sys, init_conditions, dt, max_t, lorenz_dim=2, model_name='Rossler Equations')
    # print(f'LCE: {n_dim_approx_LCE(sys, init_conditions, 1e-9, 2, dt=0.01, max_t=300)}')
    return


def Lorenz_Sys(X: np.array, t: float, params: dict) -> np.array:
    x, y, z = X[0], X[1], X[2]
    s, r, b = params['sigma'], params['rho'], params['beta']

    dx = s * (y - x)
    dy = x * (r - z) - y
    dz = x * y - b * z

    return np.array([dx, dy, dz])


def run_Lorenz():
    params = {'sigma': 10,
              'rho': 28,
              'beta': 8/3}

    dt = 0.01
    max_t = 100
    init_conditions = [1, 1, 1]
    sys = System(Lorenz_Sys, params, init_conditions)

    # analyze_system(sys, dt, max_t, lorenz_dim=2, model_name='Lorenz System')
    print(f'LCE: {n_dim_approx_LCE(sys, 1e-9, 2, dt=0.01, max_t=20)}')


def Duffing_Sys(X: np.array, t: float, params: dict) -> np.array:
    x, y = X[0], X[1]
    a, b, d, g, w = params['alpha'], params['beta'], params['delta'], params['gamma'], params['omega']

    dx = y
    dy = g * np.cos(w * t * np.pi / 180) - d * y - a * x - b * x**3

    return np.array([dx, dy])


def run_Duffing():
    params = {'alpha': -1,
              'beta': 0.25,
              'delta': 0.2,
              'gamma': 2.5,
              'omega': 2}

    dt = 0.01
    max_t = 1000
    init_conditions = [1.4, 1]
    sys = System(Duffing_Sys, params, init_conditions)

    analyze_system(sys, dt, max_t, True, 0, 'Duffing System')


run_Rossler()
# run_LotkaVolterra()
# run_Lorenz()
# run_Duffing()
# run_Log()
