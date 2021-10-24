import numpy as np
import logging
import matplotlib.pyplot as plt

class Dirichle():


    def __init__(self, right_function, bounds_x, bounds_y, bound_functions, delta_x, delta_y, eps = 1e-4):
        self.right_function = right_function
        self.bounds_x = bounds_x
        self.bounds_y = bounds_y
        self.bound_functions = bound_functions
        self.delta_x = delta_x
        self.delta_y = delta_y


class DirirchleJacobiSolver(Dirichle):


    def __init__(self, right_function, bounds_x, bounds_y, bound_functions, delta_x, delta_y, eps = 1e-4, itermax = 300):
        super().__init__(right_function, bounds_x, bounds_y, bound_functions, delta_x, delta_y, eps)
        self.itermax = itermax
        self.eps = eps
        self.num_pts_x_ax, self.num_pts_y_ax = self.get_num_grid_points_per_axis()
        print(f'Num x pts: {self.num_pts_x_ax}, Num y pts: {self.num_pts_y_ax}')
        self.x_pts, self.y_pts = self.get_axis_points_values()
        self.meshgrid = np.meshgrid(self.x_pts, self.y_pts)
        self.y_solution = np.zeros((itermax, self.num_pts_x_ax, self.num_pts_y_ax), dtype = np.float32)
        # init for initial solution
        self.init_initial_solution()
        # initialize bound conditions
        self.init_bound_conditions()
        # init_const_for_iter_process
        self.init_consts()


    def init_consts(self, k1, k2):
        self.koeff_right_f = (-1/2) * (self.delta_x ** 2 * self.delta_y ** 2) / (self.delta_y ** 2 + self.delta_x ** 2)
        self.koeff_1 = self.delta_y ** 2 / (2 * (self.delta_x ** 2 + self.delta_y ** 2))
        self.koeff_2 = self.delta_x ** 2 / (2 * (self.delta_x ** 2 + self.delta_y ** 2))


    def get_axis_points_values(self):
        x_left_b, x_right_b = self.bounds_x
        y_left_b, y_right_b = self.bounds_y
        x_pts = np.linspace(x_left_b, x_right_b, self.num_pts_x_ax)
        y_pts = np.linspace(y_left_b, y_right_b, self.num_pts_y_ax)
        return x_pts, y_pts


    def get_num_grid_points_per_axis(self):
        left_b_x, right_b_x = self.bounds_x
        left_b_y, right_b_y = self.bounds_y
        range_x = np.abs(right_b_x - left_b_x)
        range_y = np.abs(right_b_y - left_b_y)
        num_pts_x_axis = int(range_x // self.delta_x)
        num_pts_y_axis = int(range_y // self.delta_y)
        return num_pts_x_axis + 1, num_pts_y_axis + 1


    def init_bound_conditions(self):
        x_left, x_right = self.bounds_x
        y_left, y_right = self.bounds_y
        # left_side
        left_bound = self.bound_functions[0]
        self.y_solution[:, 0, :] = left_bound(self.y_pts)
        # upper_side
        upper_bound = self.bound_functions[1]
        self.y_solution[:, :, -1] = upper_bound(self.x_pts)
        # right_side
        right_bound = self.bound_functions[2]
        self.y_solution[:, -1, :] = right_bound(self.y_pts)
        # down_side
        down_bound = self.bound_functions[3]
        self.y_solution[:, :, 0] = down_bound(self.x_pts)


    def init_initial_solution(self, strategy = 'zero'):
        if strategy == 'zero':
            self.y_solution[0, :, :] = 0.0
        elif strategy == 'random':
            num_x, num_y = self.y_solution.shape[1:]
            self.y_solution[0, :, :] = np.random.randn(num_x, num_y) * 0.01
        else:
            logging.info(f'Wrong strategy: {strategy}!')


    def solve_iterates(self):
        # for next layer
        for it in range(1, self.y_solution.shape[0]):
            for it_x in range(1, self.y_solution.shape[1]-1):
                for it_y in range(1, self.y_solution.shape[2]-1):
                    self.y_solution[it, it_x, it_y] = self.get_new_value(it, it_x, it_y)
            if np.abs(self.y_solution[it-1] - self.y_solution[it]).max < self.eps:
                print(f'Stop on iteration: {it}')
                break

    def get_new_value(self, it, it_x, it_y):
        u_n_p_1_m = self.y_solution[it-1, it_x + 1, it_y]
        u_n_m_1_m = self.y_solution[it-1, it_x - 1, it_y]
        u_n_m_p_1 = self.y_solution[it-1, it_x, it_y + 1]
        u_n_m_m_1 = self.y_solution[it-1, it_x, it_y - 1]
        x_n_m = self.x_pts[it_x]
        y_n_m = self.y_pts[it_y]
        f_n_m = self.right_function(x_n_m, y_n_m)
        return self.koeff_right_f * f_n_m + self.koeff_1 * (u_n_p_1_m + u_n_m_1_m) + self.koeff_2 * (u_n_m_p_1 + u_n_m_m_1)

    def plot_graph(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(np.ravel(self.meshgrid[0][:]), np.ravel(self.meshgrid[1][:]), np.ravel(self.y_solution[-1][:]))
        plt.show()


    def get_value(self, x_coor, y_coor):
        # find closest x_coor, y_coor
        id_min_x = np.argmin(self.x_pts - x_coor)
        id_min_y = np.argmin(self.y_pts - y_coor)
        # we call
        x_coor_closest, y_coor_closest = self.x_pts[id_min_x], self.y_pts[id_min_y]
        print(f'x: {x_coor}, x_closest: {x_coor_closest}, y: {y_coor}, y_closest: {y_coor_closest}')
        return self.y_solution[-1, id_min_x, id_min_y]


class DirichleTriangleSolver(DirirchleJacobiSolver):


    def __init__(self, right_function, bounds_x, bounds_y, bound_functions, delta_x, delta_y, tau_delta, eps = 1e-4, itermax=300):
        super().__init__(right_function, bounds_x, bounds_y, bound_functions, delta_x, delta_y, eps, itermax)
        self.delta_tau = tau_delta

    def init_consts(self, k1, k2):
        self.r1 = (self.delta_tau * k1) / (self.delta_x ** 2)
        self.r2 = (self.delta_tau * k2) / (self.delta_y ** 2)

