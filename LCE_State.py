# STATE CLASS, WITH POSITION, VELOCITY, AND ACCELERATION
# USED BY N_BODY_PROBLEM.PY
# NOT USED FOR

import numpy as np
import matplotlib.pyplot as plt


class State:
    def __init__(self, delta, copy, get, string):
        self.delta_func = delta
        self.get_func = get
        self.copy_func = copy
        self.string_func = string

    def delta_quantity(self, quantity: str, value):
        return self.delta_func(quantity, value)

    def copy_state(self):
        return self.copy_func

    def __getitem__(self, item):
        return self.get_func(item)

    def __str(self):
        return self.string_func()


class State3D(State):
    def __init__(self, r):
        super().__init__(self.delta_quantity, self.copy_func, self.__getitem__, self.__str__)
        self.r = r

    def delta_quantity(self, quantity: str, value: float):
        if quantity == 'rx':
            self.r[0] += value
        elif quantity == 'ry':
            self.r[1] += value
        elif quantity == 'rz':
            self.r[2] += value
        else:
            raise Exception('Invalid Input')

    def copy_func(self):
        return State3D(np.copy(self.r))

    def __getitem__(self, item):
        if item == 'rx':
            return self.r[0]
        if item == 'ry':
            return self.r[1]
        if item == 'rz':
            return self.r[2]
        raise Exception('Invalid Input')

    def __str__(self):
        out = f'x: {self.r[0]}, y: {self.r[1]}, z: {self.r[2]}'
        return out


class NBodyState(State):
    # set initial state of all particles in system
    # m is a list of particle masses
    # r is a list of lists of particle position components, etc.
    def __init__(self, m, r, v, a):
        super().__init__(self.delta_quantity, self.copy_func, self.__getitem__, self.__str__)
        self.m = m
        self.r = np.array(r)
        self.v = v
        self.a = a

    # given a certain quantity, add a delta value to that quantity
    def delta_quantity(self, quantity: str, value):
        if quantity == 'rx':
            self.r[0][0] += value
            return
        if quantity == 'ry':
            self.r[0][1] += value
            return
        if quantity == 'rz':
            self.r[0][2] += value
            return
        if quantity == 'vx':
            self.v[0][0] += value
            return
        if quantity == 'vy':
            self.v[0][1] += value
            return
        if quantity == 'vz':
            self.v[0][2] += value
            return
        if quantity == 'ax':
            self.a[0][0] += value
            return
        if quantity == 'ay':
            self.a[0][1] += value
            return
        if quantity == 'az':
            self.a[0][2] += value
            return
        raise Exception('Invalid Input')

    def copy_func(self):
        return NBodyState(np.copy(self.m), np.copy(self.r), np.copy(self.v), np.copy(self.a))

    # overloads [] operator to allow us to use something like System['m'] to get all the mass values of each particle
    def __getitem__(self, item):
        if item == 'm':
            return self.m
        if item == 'rx':
            return self.r[0][0]
        if item == 'ry':
            return self.r[0][1]
        if item == 'rz':
            return self.r[0][2]
        if item == 'vx':
            return self.v[0][0]
        if item == 'vy':
            return self.v[0][1]
        if item == 'vz':
            return self.v[0][2]
        if item == 'ax':
            return self.a[0][0]
        if item == 'ay':
            return self.a[0][1]
        if item == 'az':
            return self.a[0][2]
        raise Exception('Invalid Input')

    # for printing
    def __str__(self):
        out = f'm: {self.m}\nr: {self.r}\nv: {self.v}\na: {self.a}'
        return out


def approx_sim_LCE(StateType, x_0: State, Update_Func, components: list, n: int, dt: float, t_max: float, delta: float):
    LCE_sums = np.zeros(len(components))
    cur_x = x_0
    t = 0
    print(f'Evaluating LCE with delta {delta}')
    coords = []
    if len(x_0.r.shape) > 1:
        for i in range(len(x_0.r)):
            coords.append([])
    # while still simulating
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    while t < t_max:
        print(f't={t}:')
        # get next state by updating current state
        # print(f'Current state: {cur_x}')
        next_state = Update_Func(cur_x, n, dt)
        if len(next_state.r.shape) > 1:
            for i in range(len(next_state.r)):
                coords[i].append(next_state.r[i])
        else:
            coords.append(next_state.r)
        # print(f'next state: {next_state}')
        for i in range(len(components)):
            quantity = components[i]
            # make a copy of the current state
            # print(f'component: {quantity}')
            delta_state = cur_x.copy_func()
            # increment quantity by some delta
            delta_state.delta_quantity(quantity, delta)
            # print(f'delta state: {delta_state}')
            # get next state of delta state
            next_delta_state = Update_Func(delta_state, n, dt)
            # print(f'next delta state: {next_delta_state}')
            # approximate derivative using these two states

            approx_deriv = (next_delta_state[quantity] - next_state[quantity]) / delta
            # print(f'{quantity} deriv: {approx_deriv}')
            # print(f'{quantity} + Δ: {next_delta_state[quantity]}')
            # print(f'{quantity}: {next_state[quantity]}')
            # add to sum
            LCE_sums[i] += np.log(np.abs(approx_deriv))
        # update state
        cur_x = next_state
        # increment time
        t += dt
        # print()
    # return average of the sum
    if len(next_state.r.shape) > 1:
        for i in range(len(next_state.r)):
            print(f'particle {i}')
            coords[i] = np.array(coords[i])
            coords[i] = coords[i].T
            ax.plot(coords[i][0], coords[i][1], coords[i][2])
    else:
        coords = np.array(coords).T
        # plt.plot(coords[0])
        # plt.plot(coords[1])
        # plt.plot(coords[2])
        ax.plot(coords[0], coords[1], coords[2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    return LCE_sums / n


def rossler_lce():
    params = {'a': 0.15,
              'b': 0.2,
              'c': 10.0}

    def Rossler(state: State3D, n: int, dt: float):
        x, y, z = state['rx'], state['ry'], state['rz']
        dx = -y - z
        dy = x + params['a'] * y
        dz = params['b'] + z * (x - params['c'])

        x += dx * dt
        y += dy * dt
        z += dz * dt

        return State3D([x, y, z])

    r = [-1, 1, 1]
    dt = 0.001
    delta = 0.0001
    t_max = 100
    state_0 = State3D(r)
    N = 10
    print(f"Approximate LCE: {approx_sim_LCE(State3D, state_0, Rossler, ['rx', 'ry', 'rz'], n=N, dt=dt, t_max=t_max, delta=delta)}'")





if __name__ == '__main__':
    rossler_lce()
