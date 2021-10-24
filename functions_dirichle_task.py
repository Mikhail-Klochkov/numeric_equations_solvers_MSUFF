import numpy as np

class FunctionsDirichleTask():

    def __init__(self):
        pass

    @staticmethod
    def right_function(x, y):
        return -x**2 - 2*y**2

    @staticmethod
    def left_side_rectangle_bound_cond(x):
        return (1/4) * (x - 0.5)**2

    @staticmethod
    def upper_side_rectangle_bound_cond(x):
        return (1 / 4) * (x - 0.5)**2

    @staticmethod
    def right_side_rectangle_bound_cond(x):
        return (1 / 4) * (x - 0.5)**2

    @staticmethod
    def down_side_rectangle_bound_cond(x):
        return (1 / 4) * (x - 0.5)**2

    @staticmethod
    def get_right_function():
        return FunctionsDirichleTask.right_function

    @staticmethod
    def get_bound_conditions():
        # left, upper, right, down
        return (FunctionsDirichleTask.left_side_rectangle_bound_cond,
                FunctionsDirichleTask.upper_side_rectangle_bound_cond,
                FunctionsDirichleTask.right_side_rectangle_bound_cond,
                FunctionsDirichleTask.down_side_rectangle_bound_cond)

