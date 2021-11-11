import numpy as np
import matplotlib.pyplot as plt

T = 1.
a = 1.
M = 200
N = 200

a_lenght = 3
b_lenght = 5

def f_test(t, x):
    return 0.0

def bound_cond_left_test(t):
    return 0.0

def bound_cond_right_test(t):
    return 0.0

def init_equat_test(x):
    if 0 <= x <= a_lenght/2:
        return 2 * b_lenght/a_lenght
    elif a_lenght/2 < x <= a_lenght:
        return (2 * b_lenght / a_lenght) * (a_lenght - x)
    else:
        assert False

def f(t, x):
    return np.exp(t*x)

def bound_cond_left(t):
    return t

def bound_cond_right(t):
    return np.sin(t)

def init_equat(x):
    return 0.

plot_graph = True


def init_and_bound_cond(y, x_range, t_range, test=False):
    for idx in range(x_range.shape[0]):
        if test:
            y[0, idx] = init_equat_test(x_range[idx])
        else:
            y[0, idx] = 0
    # initialize right and left conditions
    for jdx in range(t_range.shape[0]):
        if test:
            y[jdx, 0] = bound_cond_left_test(t_range[jdx])
            y[jdx, -1] = bound_cond_right_test(t_range[jdx])
        else:
            y[jdx, 0] = bound_cond_left(t_range[jdx])
            y[jdx, -1] = bound_cond_right(t_range[jdx])
    return y


def solve(test):
    y = np.zeros((M + 1, N + 1))
    tau = T / M
    h = a / N
    t_range = np.linspace(0, T, M + 1)
    x_range = np.linspace(0, a, N + 1)
    A = tau / h ** 2
    # should be N+1 or not
    alpha = np.zeros(N + 1)
    beta = np.zeros(N + 1)
    # init conditions
    y = init_and_bound_cond(y, x_range, t_range, test)
    for jdx in range(t_range.shape[0] - 1):
        alpha[1] = 0
        beta[1] = 0
        for idx in range(1, x_range.shape[0] - 1):
            alpha[idx + 1] = 1 / (2 + 1 / A - beta[idx])
            if test:
                f_var = f_test(t_range[jdx] + tau / 2, x_range[idx])
            else:
                f_var = f(t_range[jdx] + tau / 2, x_range[idx])

            beta[idx + 1] = (beta[idx] + y[jdx, idx] / A + tau * f_var) / (2 + 1 / A - beta[idx])
        # x_range.shape[0] = N+1 | N-1, N-2, ... 0 | where N=201 (199, 198, 197 .. 0)
        if test:
            f_var = f_test(t_range[jdx] + tau / 2, x_range[N])
        else:
            f_var = f(t_range[jdx] + tau / 2, x_range[N])
        y[jdx + 1, N] = (beta[N] + y[jdx, N] / A + tau * f_var) / (2 + 1 / A - beta[N])
        for i, idx in enumerate(reversed(range(x_range.shape[0] - 1))):
            y[jdx + 1, idx] = alpha[idx + 1] * y[jdx + 1, idx + 1] + beta[idx + 1]
    # visualize graph
    if plot_graph:
        meshgrid = np.meshgrid(x_range, t_range)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.ravel(meshgrid[1][:]), np.ravel(meshgrid[0][:]), np.ravel(y))
        plt.show()
    # answer
    x_0 = 0.5
    t_0 = 1.
    idx_x_range_pt = np.argmin(np.abs(x_range - x_0))
    idx_t_range_pt = np.argmin(np.abs(t_range - t_0))
    x_0_pt = x_range[idx_x_range_pt]
    t_0_pt = t_range[idx_t_range_pt]
    print(f'answer: {y[idx_t_range_pt, idx_x_range_pt]}')
    print(f'Found in indeces: {(t_0_pt, x_0_pt)}')


if __name__ == '__main__':
    test_use = False
    solve(test_use)
