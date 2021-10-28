
import numpy as np
import matplotlib.pyplot as plt

T = np.pi
a = np.pi * 2
c = 1.
M = 200
N = 200

def init_equat(x):
    return np.sin(x)

def init_equat_der(x):
    return np.cos(x)

def f(t, x):
    return 1 / (1 + 2 * (t ** 2) * (x ** 2))

plot_graph = False

if __name__ == '__main__':
    # first t, second = x
    y = np.zeros((M+1, N+1))
    tau = T / M
    h = a / N
    t_range = np.linspace(0, T, M+1)
    x_range = np.linspace(0, a, N+1)
    print(tau, h)
    # const
    r = c ** 2 * tau**2 / (h**2)
    # numeric solution on 1-2 layers
    for idx, x in enumerate(x_range):
        y[0, idx] = init_equat(x)
        y[1, idx] = y[0, idx] + tau * init_equat_der(x) + (0.5 * tau ** 2) * (c ** 2 * (-1) * np.sin(x) + f(x, 0))

    # left right bound conditions
    for jdx, t in enumerate(t_range):
        y[jdx, 0] = t
        y[jdx, -1] = t**2

    # scheme
    for jdx in range(1, t_range.shape[0]-1):
        for idx in range(1, x_range.shape[0]-1):
            y[jdx+1, idx] = 2 * y[jdx, idx] - y[jdx-1, idx] + r * (y[jdx, idx+1] - 2 * y[jdx, idx] + y[jdx, idx-1]) \
                            + tau ** 2 * f(t_range[jdx], x_range[idx])

    if plot_graph:

        meshgrid = np.meshgrid(x_range, t_range)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.ravel(meshgrid[1][:]), np.ravel(meshgrid[0][:]), np.ravel(y))
        plt.show()

    # answer
    x_0 = np.pi
    t_0 = np.pi
    idx_x_range_pt = np.argmin(np.abs(x_range - x_0))
    idx_t_range_pt = np.argmin(np.abs(t_range - t_0))
    x_0_pt = x_range[idx_x_range_pt]
    t_0_pt = t_range[idx_t_range_pt]
    print(f'answer: {y[idx_t_range_pt, idx_x_range_pt]}')
    print(idx_t_range_pt, idx_x_range_pt)

