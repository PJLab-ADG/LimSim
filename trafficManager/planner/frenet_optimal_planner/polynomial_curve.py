"""
Author: Xuan Zhao
Date: 2022-06-14 12:15:01
Description: 
Polynomial curve fitting including quartic and quintic

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
import numpy as np


# QuarticPolynomial
class QuarticPolynomial:
    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time**2, 4 * time**3], [6 * time, 12 * time**2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time, axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = (
            self.a0
            + self.a1 * t
            + self.a2 * t**2
            + self.a3 * t**3
            + self.a4 * t**4
        )

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + 3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class QuinticPolynomial:
    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        # calc coefficient of quintic polynomial
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array(
            [
                [T**3, T**4, T**5],
                [3 * T**2, 4 * T**3, 5 * T**4],
                [6 * T, 12 * T**2, 20 * T**3],
            ]
        )
        b = np.array(
            [
                xe - self.a0 - self.a1 * T - self.a2 * T**2,
                vxe - self.a1 - 2 * self.a2 * T,
                axe - 2 * self.a2,
            ]
        )
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = (
            self.a0
            + self.a1 * t
            + self.a2 * t**2
            + self.a3 * t**3
            + self.a4 * t**4
            + self.a5 * t**5
        )

        return xt

    def calc_first_derivative(self, t):
        xt = (
            self.a1
            + 2 * self.a2 * t
            + 3 * self.a3 * t**2
            + 4 * self.a4 * t**3
            + 5 * self.a5 * t**4
        )

        return xt

    def calc_second_derivative(self, t):
        xt = (
            2 * self.a2
            + 6 * self.a3 * t
            + 12 * self.a4 * t**2
            + 20 * self.a5 * t**3
        )

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

        return xt
