"""
This module provides functions for converting between Frenet and Cartesian coordinates in 2D space.

Functions:
    - frenet_to_cartesian2D(rx, ry, ryaw, rkappa, state): 
        Converts Frenet coordinates to Cartesian coordinates.
    - cartesian_to_frenet2D(rs, rx, ry, ryaw, rkappa, state): 
        Converts Cartesian coordinates to Frenet coordinates.

References:
    Modified from: https://blog.csdn.net/u013468614/article/details/108748016
"""

import math
from typing import Tuple
from utils.trajectory import State


def frenet_to_cartesian2D(rx: float,
                          ry: float,
                          ryaw: float,
                          rkappa: float,
                          state: State) -> Tuple[float, float, float ,float]:
    """
    Converts a Frenet coordinate to a 2D Cartesian coordinate.
    
    Args:
        rx (float): x-coordinate of the Frenet coordinate.
        ry (float): y-coordinate of the Frenet coordinate.
        ryaw (float): yaw of the Frenet coordinate.
        rkappa (float): curvature of the Frenet coordinate.
        state (object): object containing the state of the Frenet coordinate.
    
    Returns:
        x (float): x-coordinate of the 2D Cartesian coordinate.
        y (float): y-coordinate of the 2D Cartesian coordinate.
        v (float): velocity of the 2D Cartesian coordinate.
        yaw (float): yaw of the 2D Cartesian coordinate.
    """
    cos_theta_r = math.cos(ryaw)
    sin_theta_r = math.sin(ryaw)

    x = rx - sin_theta_r * state.d
    y = ry + cos_theta_r * state.d

    one_minus_kappa_r_d = 1 - rkappa * state.d
    v = math.sqrt(one_minus_kappa_r_d ** 2 * state.s_d ** 2 + state.d_d ** 2)
    if v == 0:
        yaw = ryaw
    else:
        yaw = math.asin(state.d_d / v) + ryaw

    return x, y, v, yaw


def cartesian_to_frenet2D(rs: float,
                          rx: float,
                          ry: float,
                          ryaw: float,
                          rkappa: float,
                          state: State) -> Tuple[float, float, float, float]:
    """
    Converts a given state from Cartesian coordinates to Frenet coordinates.
    
    Args:
        rs (float): The s coordinate of the reference point.
        rx (float): The x coordinate of the reference point.
        ry (float): The y coordinate of the reference point.
        ryaw (float): The yaw of the reference point.
        rkappa (float): The curvature of the reference point.
        state (State): The state to be converted.
    
    Returns:
        s (float): The s coordinate of the given state.
        s_d (float): The s-dot coordinate of the given state.
        d (float): The d coordinate of the given state.
        d_d (float): The d-dot coordinate of the given state.
    """
    s = rs
    dx = state.x - rx
    dy = state.y - ry

    cos_theta_r = math.cos(ryaw)
    sin_theta_r = math.sin(ryaw)
    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d = math.copysign(math.sqrt(dx * dx + dy * dy), cross_rd_nd)

    delta_theta = state.yaw - ryaw
    sin_delta_theta = math.sin(delta_theta)
    cos_delta_theta = math.cos(delta_theta)
    one_minus_kappa_r_d = 1 - rkappa * d
    s_d = state.vel * cos_delta_theta / one_minus_kappa_r_d
    d_d = state.vel * sin_delta_theta

    return s, s_d, d, d_d
