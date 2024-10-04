import matplotlib.pyplot as plt
import numpy as np


# SPECIFICALLY HENON MAP


class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class HenonSystem:
    @staticmethod
    def iterate(x_n: State, a, b):
        x = x_n.x
        y = x_n.y
        x_n1 = State(1 - a*x**2 + y, b * x)
        return x_n1


a = 1.4
b = 0.3

x_0 = State(0.25, 0.25)
cur_x = x_0
n = 10000
coords = []
for i in range(n):
    x, y = cur_x.x, cur_x.y
    coords.append([x, y])
    cur_x = HenonSystem.iterate(cur_x, a, b)

coords = np.array(coords).T
plt.scatter(coords[0], coords[1])
plt.show()