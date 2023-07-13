"""
This module provides functions to calculate various costs for a given path in a traffic scenario.
These costs include smoothness, velocity difference, time, obstacle, guidance, acceleration, jerk,
stop, and lane change costs. The module also provides a main function to run the calculations.
"""
from typing import Union
import numpy as np
import obstacle_cost
from trafficManager.common.vehicle import Vehicle
from utils.cubic_spline import Spline2D
from utils.obstacles import ObsType
from utils.trajectory import Trajectory


def smoothness(trajectory: Trajectory,
               ref_line: Spline2D, weight_config: dict) -> float:
    """
    Calculate the smoothness cost of a given path.

    Args:
        trajectory (Trajectory): The path to evaluate.
        ref_line (Spline2D): The reference line for the path.
        weight_config (dict): The weight configuration for the cost calculation.

    Returns:
        float: The smoothness cost.
    """
    cost_yaw_diff = 0
    cost_cur = 0
    for state in trajectory.states:
        if state.s >= ref_line.s[-1] or state.laneID != trajectory.states[0].laneID:
            break
        cost_yaw_diff += (state.yaw - ref_line.calc_yaw(state.s)) ** 2
        cost_cur += state.cur ** 2

    return weight_config["W_YAW"] * cost_yaw_diff + weight_config["W_CUR"] * cost_cur


def vel_diff(trajectory: Trajectory, ref_vel_list: Union[float, np.ndarray],
             weight_config: dict) -> float:
    """
    Calculate the velocity difference cost of a given path.

    Args:
        trajectory (Trajectory): The path to evaluate.
        ref_vel_list (Union[float, np.ndarray]): The reference velocity list for the path.
        weight_config (dict): The weight configuration for the cost calculation.

    Returns:
        float: The velocity difference cost.
    """
    velocities = np.array([state.vel for state in trajectory.states])
    cost_vel_diff = np.linalg.norm(velocities - ref_vel_list, 2)**2
    return weight_config["W_VEL_DIFF"] * cost_vel_diff


def time(trajectory: Trajectory, weight_config: dict) -> float:
    """
    Calculate the time cost of a given path.

    Args:
        path (Path): The path to evaluate.
        weight_config (dict): The weight configuration for the cost calculation.

    Returns:
        float: The time cost.
    """
    return weight_config["W_T"] * trajectory.states[-1].t


def obs(vehicle: Vehicle, trajectory: Trajectory,
        obs_list: list, config: dict, offset_frame:int = 0 ) -> float:
    """
    Calculate the obstacle cost of a given trajectory.

    Args:
        vehicle (Vehicle): The vehicle object.
        trajectory (Trajectory): The trajectory to evaluate.
        obs_list (list): A list of Obstacle objects.
        config (dict): The configuration for the cost calculation.
        offset_frame (int>0): The offset frame for start frame of vehicle trajectory.

    Returns:
        float: The obstacle cost.
    """
    cost_obs = 0
    for obstacle in obs_list:
        if obstacle.type == ObsType.OTHER:
            cost_obs += obstacle_cost.calculate_static(
                vehicle, obstacle, trajectory, config)
        elif obstacle.type == ObsType.CAR:
            cost_obs += obstacle_cost.calculate_car(vehicle, obstacle,
                                                    trajectory, config,offset_frame)
        elif obstacle.type == ObsType.PEDESTRIAN:
            cost_obs += obstacle_cost.calculate_pedestrian(
                vehicle, obstacle, trajectory, config)

    return cost_obs


def guidance(trajectory: Trajectory, weight_config: dict) -> float:
    """
    Calculate the guidance cost of a given trajectory.

    Args:
        trajectory (Trajectory): The trajectory to evaluate.
        weight_config (dict): The weight configuration for the cost calculation.

    Returns:
        float: The guidance cost.
    """
    # cost_guidance = 0
    # for state in trajectory.states:
    #     cost_guidance += state.d ** 2
    offset = np.array([state.d for state in trajectory.states])
    cost_guidance = np.sum(np.power(offset, 2))
    return weight_config["W_GUIDE"] * cost_guidance


def ref_waypoints_guidance(trajectory: Trajectory,
                           waypoints: list, weight_config: dict) -> None:
    """
    Calculate the waypoint reference cost for guidance.

    Args:
        trajectory (Trajectory): The trajectory to evaluate.
        waypoints (list): The list of waypoints for the trajectory.
        weight_config (dict): The weight configuration for the cost calculation.

    Returns:
        None: This function is not yet implemented.
    """
    pass  # todo: add waypoint reference cost for guidance


def acc(trajectory: Trajectory, weight_config: dict) -> float:
    """
    Calculate the acceleration cost of a given trajectory.

    Args:
        trajectory (Trajectory):The trajectory to evaluate.
        weight_config (dict): The weight configuration for the cost calculation.

    Returns:
        float: The acceleration cost.
    """
    cost_acc = 0
    for state in trajectory.states:
        cost_acc += state.acc ** 2
    return weight_config["W_ACC"] * cost_acc


def jerk(trajectory: Trajectory, weight_config: dict) -> float:
    """
    Calculate the jerk cost of a given trajectory.

    Args:
        trajectory (Trajectory): The trajectory to evaluate.
        weight_config (dict): The weight configuration for the cost calculation.

    Returns:
        float: The jerk cost.
    """
    cost_jerk = 0
    for state in trajectory.states:
        cost_jerk += state.s_ddd ** 2 + state.d_ddd ** 2
    return weight_config["W_JERK"] * cost_jerk


def stop(weight_config):
    return weight_config["W_STOP"]


def changelane(weight_config):
    return weight_config["W_CHANGELANE"]
