import numpy as np
from typing import Union
import matplotlib.pyplot as plt
from matplotlib import animation
from LCE_State import *


# SIMULATES AN N-BODY SYSTEM

# constants for simulation
# G = 6.67e-11
G = 10           # gravitational constant, much bigger than actual to speed up simulation
SOFTENING = 0.1  # softening parameter
PAST_LIM = 100   # number of past time step states to store
AXIS_LIM = 3     # animation axis size limit
N = 3            # number of particles


# System class
class System:
    # set initial state of all particles in system
    # M is an array of particle masses
    # R is a 2D array of particle position components, etc.
    def __init__(self, M: Union[list, np.array], R: Union[list, np.array], V: Union[list, np.array]):
        self.m = np.array(M)
        self.r = np.array(R)
        self.v = np.array(V)
        self.n = len(R)
        self.a = np.zeros((self.n, 3))
        self.t = 0
        # previous values for each component, to track orbits
        self.prev_r = [[] for _ in range(self.n)]
        self.prev_v = [[] for _ in range(self.n)]
        self.last_t = []

    # calculate acceleration on each particle
    def update_accel(self):
        x = self.r[:, 0:1]  # get x coordinates of all particles
        y = self.r[:, 1:2]  # get y coordinates of all particles
        z = self.r[:, 2:3]  # get z coordinates of all particles

        # get differentials between each particle and all other particles (x0-x0,...,x0-x(n-1),...,x(n-1)-x_0,...,x(n-1)-x(n-1))
        dx = x.T - x
        dy = y.T - y
        dz = z.T - z

        # calculate denominator in Newton law of universal gravitation
        inv_r3 = (dx**2 + dy**2 + dz**2 + SOFTENING**2)**-1.5

        # make sure don't make particles cause acceleration on themselves
        for i in range(self.n):
            inv_r3[i][i] = 0

        # calculate each component of acceleration for all particles using matrix multiplication
        ax = G * (dx * inv_r3) @ self.m
        ay = G * (dy * inv_r3) @ self.m
        az = G * (dz * inv_r3) @ self.m

        # store acceleration array
        self.a = np.array([ax, ay, az]).T[0]

    # calculate acceleration on each particle in a standalone state
    @staticmethod
    def static_accel(r, m, n):
        x = r[:, 0:1]
        y = r[:, 1:2]
        z = r[:, 2:3]

        dx = x.T - x
        dy = y.T - y
        dz = z.T - z

        inv_r3 = (dx ** 2 + dy ** 2 + dz ** 2 + SOFTENING ** 2) ** -1.5
        for i in range(n):
            inv_r3[i][i] = 0

        ax = G * (dx * inv_r3) @ m
        ay = G * (dy * inv_r3) @ m
        az = G * (dz * inv_r3) @ m

        return np.array([ax, ay, az]).T[0]

    # update state of system
    def update_particles(self, dt):
        # first half-step kick for velocity
        self.v += self.a * dt / 2.0

        # full-step kick for position
        self.r += self.v * dt

        # add current values to previous value lists, for keeping track of orbits
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

        # cutoff old data so we don't store an insane amount
        self.prev_r = self.prev_r[-PAST_LIM:]
        self.prev_v = self.prev_v[-PAST_LIM:]
        self.last_t = self.last_t[-PAST_LIM:]

        # update acceleration
        self.update_accel()
        # second half-step kick for velocity
        self.v += self.a * dt / 2.0
        # increment time
        self.t += dt

    # update standalone state
    @staticmethod
    def static_update(state: NBodyState, n, dt):
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
        return NBodyState(m, new_r, new_v, new_a)

    # method to compute energy in system, not done yet
    def compute_energy(self):

        return


# update matplotlib figure
def update_figure(t, dt, sys: System, orb_axis):
    # clear axis
    orb_axis.cla()
    # set axis limits
    orb_axis.set_xlim(-AXIS_LIM, AXIS_LIM)
    orb_axis.set_ylim(-AXIS_LIM, AXIS_LIM)
    orb_axis.set_zlim(-AXIS_LIM, AXIS_LIM)
    # set axis labels
    orb_axis.set_xlabel('x')
    orb_axis.set_ylabel('y')
    orb_axis.set_zlabel('z')

    # update particles in system
    sys.update_particles(dt)

    # plot each particle
    for p in range(sys.n):
        cur_p = sys.r[p]
        last_p = np.array(sys.prev_r[p][-PAST_LIM:]).T
        orb_axis.scatter(cur_p[0], cur_p[1], cur_p[2])
        orb_axis.plot(last_p[0][-PAST_LIM:], last_p[1][-PAST_LIM:], last_p[2][-PAST_LIM:])

    return


# animate a given system
def animate_system(sys: System, dt: float):
    fig = plt.figure()
    orbit_axis = fig.add_subplot(1, 1, 1, projection='3d')

    anim = animation.FuncAnimation(fig, update_figure, fargs=(dt, sys, orbit_axis),
                                   interval=1, blit=False)
    fig.tight_layout()
    plt.legend()
    plt.show()
    plt.close()


def main():
    M = 1 * np.ones((N, 1))/N       # mass array of even mass for all particles
    print(f'M: {M.shape}')
    R = np.random.randn(N, 3)       # random position array for all particles
    print(f'R: {R.shape}')
    V = np.random.randn(N, 3) / 5   # random velocity array for all particles
    print(f'V: {V.shape}')

    V -= np.mean(M * V, 0) / np.mean(M)     # convert velocity to center-of-mass frame

    sys = System(M, R, V)


    state_0 = NBodyState(M, R, V, np.zeros((N, 3)))  # Random state to model
    # state_1 = System.static_update(state_0, N, dt=0.01)  # new state at t=dt
    dt = 0.001
    t_max = 100.0
    print(f"Approximate LCE: {approx_sim_LCE(NBodyState, state_0, System.static_update, ['rx', 'ry', 'rz', 'vx', 'vy', 'vz'], n=N, dt=dt, t_max=t_max, delta=0.01)}'")

    # animate_system(sys, 0.01)


if __name__ == '__main__':
    main()

# CALCULATE LYAPUNOV EXPONENT FROM DATA
