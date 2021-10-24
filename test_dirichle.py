import numpy as np
from unittest import TestCase
from dirichle import DirirchleJacobiSolver
from functions_dirichle_task import FunctionsDirichleTask

class ClassTestDirichle(TestCase):

    def test_grid_points(self, bound_x = (0, 1), bound_y = (0, 1), delta_x = 1/60, delta_y = 1/60, solution = (0.5, 0.5)):
        right_f = FunctionsDirichleTask.get_right_function()
        bound_conds = FunctionsDirichleTask.get_bound_conditions()
        dirichle_solver = DirirchleJacobiSolver(right_f, bound_x, bound_y, bound_conds, delta_x, delta_y)
        dirichle_solver.solve_iterates()
        dirichle_solver.plot_graph()
        x, y = solution
        answer = dirichle_solver.get_value(x_coor=x, y_coor=y)
        print(f'answer in point u({x}, {y}): {answer:.14f}')
