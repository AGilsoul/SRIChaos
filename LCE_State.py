# STATE CLASS, WITH POSITION, VELOCITY, AND ACCELERATION
# USED BY N_BODY_PROBLEM.PY
# NOT USED FOR

import numpy as np


class State:
    # set initial state of all particles in system
    # m is a list of particle masses
    # r is a list of lists of particle position components, etc.
    def __init__(self, m, r, v, a):
        self.m = m
        self.r = r
        self.r_mag = np.dot(r[0], r[0])**0.5
        self.v = v
        self.v_mag = np.dot(v[0], v[0])**0.5
        self.a = a
        self.a_mag = np.dot(a[0], a[0])**0.5

    # given a certain quantity, add a delta value to that quantity
    def delta_quantity(self, quantity: str, value):
        if quantity == 'm':
            self.m += value
            return
        if quantity == 'r':
            self.r += np.ones(self.r.shape) * value
            self.r_mag = np.dot(self.r[0], self.r[0])**0.5
            return
        if quantity == 'v':
            self.v += np.ones(self.v.shape) * value
            self.v_mag = np.dot(self.v[0], self.v[0])**0.5
            return
        if quantity == 'a':
            self.a += np.ones(self.a.shape) * value
            self.a_mag = np.dot(self.a[0], self.a[0])**0.5
            return
        raise Exception('Invalid Input')

    # overloads [] operator to allow us to use something like System['m'] to get all the mass values of each particle
    def __getitem__(self, item):
        if item == 'm':
            return self.m
        if item == 'm_mag':
            return self.m
        if item == 'r':
            return self.r
        if item == 'r_mag':
            return self.r_mag
        if item == 'v':
            return self.v
        if item == 'v_mag':
            return self.v_mag
        if item == 'a':
            return self.a
        if item == 'a_mag':
            return self.a_mag
        raise Exception('Invalid Input')

    # for printing
    def __str__(self):
        out = f'm: {self.m}\nr: {self.r}\nv: {self.v}\na: {self.a}'
        return out


def approx_sim_LCE(x_0: State, Update_Func, quantity: str, n: int, dt: float, t_max: float, delta: float):
    sum = 0
    cur_x = x_0
    t = 0
    print(f'Evaluating {quantity} with delta {delta}')
    # while still simulating
    while t < t_max:
        print(f't={t}:')
        # make a copy of the current state
        delta_state = State(np.copy(cur_x.m), np.copy(cur_x.r), np.copy(cur_x.v), np.copy(cur_x.a))
        # increment quantity by some delta
        delta_state.delta_quantity(quantity, delta)
        # get next state by updating current state
        next_state = Update_Func(cur_x, n, dt)
        # get next state of delta state
        next_delta_state = Update_Func(delta_state, n, dt)
        # approximate derivative using these two states
        approx_deriv = (next_delta_state[f'{quantity}_mag'] - next_state[f'{quantity}_mag']) / delta

        print(f'deriv: {approx_deriv}')
        # add to sum
        sum += np.log(np.abs(approx_deriv))
        # update state
        cur_x = next_state
        # increment time
        t += dt
    # return average of the sum
    return sum / n