import numpy as np


class State:
    def __init__(self, m, r, v, a):
        self.m = m
        self.r = r
        self.r_mag = np.dot(r[0], r[0])**0.5
        self.v = v
        self.v_mag = np.dot(v[0], v[0])**0.5
        self.a = a
        self.a_mag = np.dot(a[0], a[0])**0.5

    def delta_quanity(self, quantity: str, value):
        if quantity == 'm':
            self.m += value
            return
        if quantity == 'r':
            # print(f'previous: {self.r}')
            self.r += np.ones(self.r.shape) * value
            # print(f'new: {self.r}')
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

    def __str__(self):
        out = f'm: {self.m}\nr: {self.r}\nv: {self.v}\na: {self.a}'
        return out


def approx_sim_LCE(x_0: State, Update_Func, quantity: str, n: int, dt: float, t_max: float, delta: float):
    sum = 0
    cur_x = x_0
    t = 0
    while t < t_max:
        print(f't={t}:')
        delta_state = State(np.copy(cur_x.m), np.copy(cur_x.r), np.copy(cur_x.v), np.copy(cur_x.a))
        delta_state.delta_quanity(quantity, delta)
        # print(f'cur state:\n{str(cur_x)}')
        # print(f'delta state:\n{str(delta_state)}')
        next_state = Update_Func(cur_x, n, dt)
        next_delta_state = Update_Func(delta_state, n, dt)
        approx_deriv = (next_delta_state[f'{quantity}_mag'] - next_state[f'{quantity}_mag']) / delta
        print(f'deriv: {approx_deriv}')
        sum += np.log(np.abs(approx_deriv))
        cur_x = next_state
        t += dt
    return sum / n