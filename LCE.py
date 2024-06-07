import numpy as np
from helper_code.maps import logistic_map


def LCE(x_0, r, n):
    x_vals = logistic_map(r, x_0, n)
    sum = 0
    for i in range(1, n+1):
        sum += np.log(np.abs(r * (1 - 2 * x_vals[i])))
    return sum / n


def approx_LCE(x_0, func, n, delta):
    sum = 0
    cur_x = x_0
    for i in range(1, n+1):
        next_val = func(cur_x)
        approx_deriv = (func(cur_x + delta) - next_val) / delta
        sum += np.log(np.abs(approx_deriv))
        cur_x = next_val

    return sum / n


def log_func(x):
    return logistic_map(r, x, 1)


r = 3.2
x_0 = 0.2
n = 1000
print(f'Actual LCE: {LCE(x_0, r, n)}')
print(f'Approx LCE: {approx_LCE(x_0, log_func, n, 0.0001)}')
