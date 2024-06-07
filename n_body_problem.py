import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from matplotlib import animation
from LCE_State import *


class System:
    def __init__(self, M: Union[list, np.array], R: Union[list, np.array], V: Union[list, np.array]):
        self.m = np.array(M)
        self.r = np.array(R)
        self.v = np.array(V)
        self.n = len(R)
        self.a = np.zeros((self.n, 3))
        self.t = 0
        self.prev_r = [[] for _ in range(self.n)]
        self.prev_v = [[] for _ in range(self.n)]
        self.last_t = []

    def update_accel(self):
        x = self.r[:, 0:1]
        y = self.r[:, 1:2]
        z = self.r[:, 2:3]

        dx = x.T - x
        dy = y.T - y
        dz = z.T - z

        inv_r3 = (dx**2 + dy**2 + dz**2 + Softening**2)**-1.5
        for i in range(self.n):
            inv_r3[i][i] = 0

        ax = G * (dx * inv_r3) @ self.m
        ay = G * (dy * inv_r3) @ self.m
        az = G * (dz * inv_r3) @ self.m

        self.a = np.array([ax, ay, az]).T[0]

    @staticmethod
    def static_accel(r, m, n):
        x = r[:, 0:1]
        y = r[:, 1:2]
        z = r[:, 2:3]

        dx = x.T - x
        dy = y.T - y
        dz = z.T - z

        inv_r3 = (dx ** 2 + dy ** 2 + dz ** 2 + Softening ** 2) ** -1.5
        for i in range(n):
            inv_r3[i][i] = 0

        ax = G * (dx * inv_r3) @ m
        ay = G * (dy * inv_r3) @ m
        az = G * (dz * inv_r3) @ m

        return np.array([ax, ay, az]).T[0]

    def update_particles(self, dt):
        # first half-step kick for velocity
        self.v += self.a * dt / 2.0

        # full-step kick for position
        self.r += self.v * dt

        for i in range(self.n):
            if len(self.prev_r[i]) <= 1:
                self.prev_r[i].append(np.copy(self.r[i]))
                self.prev_v[i].append(np.copy(self.v[i]))
                if i == 0:
                    self.last_t.append(self.t)
            elif not (self.prev_r[i][-1] == self.r[i]).all():
                self.prev_r[i].append(np.copy(self.r[i]))
                self.prev_v[i].append(np.copy(self.v[i]))
                if i == 0:
                    self.last_t.append(self.t)

        self.prev_r = self.prev_r[-PAST_LIM:]
        self.prev_v = self.prev_v[-PAST_LIM:]
        self.last_t = self.last_t[-PAST_LIM:]

        # update acceleration
        self.update_accel()
        # second half-step kick for velocity
        self.v += self.a * dt / 2.0

        self.t += dt

    @staticmethod
    def static_update(state: State, n, dt):
        m = state.m
        r = state.r
        v = state.v
        a = state.a
        # first half-step kick for velocity
        new_v = v + a * dt / 2.0

        # full-step kick for position
        new_r = r + v * dt

        # update acceleration
        new_a = System.static_accel(r, m, n)
        # second half-step kick for velocity
        new_v += new_a * dt / 2.0
        return State(m, new_r, new_v, new_a)

    def compute_energy(self):

        return


lim = 3


def update_figure(t, dt, sys: System, orb_axis, r_axis, v_axis):
    orb_axis.cla()
    r_axis.cla()
    v_axis.cla()
    orb_axis.set_xlim(-lim, lim)
    orb_axis.set_ylim(-lim, lim)
    orb_axis.set_zlim(-lim, lim)
    orb_axis.set_xlabel('x')
    orb_axis.set_ylabel('y')
    orb_axis.set_zlabel('z')

    v_axis.set_xlabel('t')
    v_axis.set_ylabel('m/s(?)')
    r_axis.set_xlabel('t')
    r_axis.set_ylabel('m(?)')

    sys.update_particles(dt)

    # axis.plot(sys.r)
    for p in range(sys.n):
        cur_p = sys.r[p]
        last_p = np.array(sys.prev_r[p][-PAST_LIM:]).T
        if p == 0:
            last_V = np.array(sys.prev_v[p][-PAST_LIM:]).T
            total_last_V = np.array([np.dot(x, x)**0.5 for x in last_V.T])
            last_t = np.array(sys.last_t[-PAST_LIM:])
            v_axis.set_xlim(last_t[0], last_t[-1])
            r_axis.set_xlim(last_t[0], last_t[-1])
            r_axis.plot(last_t, last_p[0], label='x')
            r_axis.plot(last_t, last_p[1], label='y')
            r_axis.plot(last_t, last_p[2], label='z')
            v_axis.plot(last_t, total_last_V, label='speed')
            v_axis.legend()
            r_axis.legend()
        orb_axis.scatter(cur_p[0], cur_p[1], cur_p[2])
        orb_axis.plot(last_p[0][-PAST_LIM:], last_p[1][-PAST_LIM:], last_p[2][-PAST_LIM:])
    # axis.scatter(sys.r[:,0], sys.r[:,1], sys.r[:,2])

    return



def animate_system(sys: System, dt):
    fig = plt.figure()
    orbit_axis = fig.add_subplot(2, 2, 1, projection='3d')
    r_axis = fig.add_subplot(2, 2, 2)
    v_axis = fig.add_subplot(2, 2, 3)

    anim = animation.FuncAnimation(fig, update_figure, fargs=(dt, sys, orbit_axis, r_axis, v_axis),
                                   interval=1, blit=False)
    fig.tight_layout()
    plt.legend()
    plt.show()
    plt.close()


# G = 6.67e-11
G = 10
Softening = 0.1
PAST_LIM = 100
N = 3

M = 1 * np.ones((N, 1))/N
print(f'M: {M.shape}')
R = np.random.randn(N, 3)
print(f'R: {R.shape}')
V = np.random.randn(N, 3) / 5
print(f'V: {V.shape}')

V -= np.mean(M * V, 0) / np.mean(M)

# V = np.zeros((N, 3))

solar_m = np.array([1.989e33, 5.972e27])
solar_r = np.array([[0, 0, 0], [1.5e12, 0, 0]])
solar_v = np.array([[0, 0, 0], [0, 2.98e5, 0]])

sys = System(M, R, V)
solar_sys = System(solar_m, solar_r, solar_v)

state_0 = State(M, R, V, np.zeros((N, 3)))
print(f'Approximate LCE: {approx_sim_LCE(state_0, System.static_update, "v", N, 0.001, 10.0, 0.001)}')
# animate_system(sys, 0.01)
# animate_system(solar_sys, 0.001)

# CALCULATE LYAPUNOV EXPONENT FROM DATA
