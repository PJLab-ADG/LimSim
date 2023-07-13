"""
Author: Licheng Wen
Date: 2022-07-12 14:53:05
Description: 

For details:
https://www.notion.so/pjlab-adg/Cost-Function-15a37bb612e24b03817937b0f3c9f6b4

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
import math
import numpy as np
from typing import Tuple, Optional
from trafficManager.common.vehicle import Vehicle

from utils.trajectory import Trajectory


def rotate_yaw(yaw: float) -> np.ndarray:
    """
    Rotate the yaw angle.

    Args:
        yaw (float): The yaw angle in radians.

    Returns:
        np.ndarray: The rotation matrix for the given yaw angle.
    """
    return np.array([[np.cos(yaw), np.sin(yaw)],
                     [-np.sin(yaw), np.cos(yaw)]], dtype=np.float32,)


def check_collsion_new(
    ego_center: np.ndarray,
    ego_length: float,
    ego_width: float,
    ego_yaw: float,
    obs_center: np.ndarray,
    obs_length: float,
    obs_width: float,
    obs_yaw: float,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Check if there is a collision between ego vehicle and obstacle.

    Args:
        ego_center (np.ndarray): The center coordinates of the ego vehicle.
        ego_length (float): The length of the ego vehicle.
        ego_width (float): The width of the ego vehicle.
        ego_yaw (float): The yaw angle of the ego vehicle.
        obs_center (np.ndarray): The center coordinates of the obstacle.
        obs_length (float): The length of the obstacle.
        obs_width (float): The width of the obstacle.
        obs_yaw (float): The yaw angle of the obstacle.

    Returns:
        Tuple[bool, Optional[np.ndarray]]: A tuple containing a boolean value indicating
                                           if there is a collision, and the nearest corner
                                           of the obstacle if there is no collision.
    
    References:
        https://juejin.cn/post/6974320430538883108
    """

    # x,y一半的长度
    ego_shape = np.array([ego_length / 2, ego_width / 2])
    obs_shape = np.array([obs_length / 2, obs_width / 2])

    rotate_ego = rotate_yaw(-ego_yaw)
    rotate_obs = rotate_yaw(-obs_yaw)

    iRotate_ego = np.linalg.inv(rotate_ego)  # A的逆矩阵，后续有用

    obs_corner = (
        np.array(
            [
                [-obs_shape[0], -obs_shape[1]],
                [-obs_shape[0], obs_shape[1]],
                [obs_shape[0], obs_shape[1]],
                [obs_shape[0], -obs_shape[1]],
            ]
        )
        @ rotate_yaw(-obs_yaw).T
    ) + obs_center

    # aBase = np.array([iRotate_ego.dot(item) for item in ego_corner])
    iAPos = iRotate_ego.dot(ego_center)

    bBase = np.array([iRotate_ego.dot(item) for item in obs_corner])
    iBPos = iRotate_ego.dot(obs_center)

    posTotal = abs(iBPos - iAPos)

    rotateC = iRotate_ego.dot(rotate_obs)
    bh = np.array([abs(item) for item in rotateC]).dot(obs_shape)
    test_matrix = posTotal - bh - ego_shape
    if test_matrix[0] <= 0 and test_matrix[1] <= 0:
        return True, None
    else:
        obs_relative_corner = bBase - iAPos
        dist = np.hypot(obs_relative_corner[:, 0], obs_relative_corner[:, 1])
        index = np.where(dist == np.min(dist))
        return False, obs_relative_corner[index][0]


def calculate_static(vehicle: Vehicle, obs: dict, 
                     trajectory: Trajectory, config: dict) -> float:
    """
    Calculate the static cost of the vehicle.

    Args:
        vehicle (Vehicle): The ego vehicle object.
        obs (dict): The obstacle information.
        trajectory (Trajectory): The Trajectory object.
        config (dict): The configuration dictionary.

    Returns:
        float: The static cost value.
    """
    cost = 0
    car_width = vehicle.width
    car_length = vehicle.length
    dist_thershold = math.hypot(
        car_length + obs["length"], car_width + obs["width"])

    # rotate and translate the obstacle
    for i in range(0, len(trajectory.states), 2):
        state = trajectory.states[i]
        # todo: can change to AABB filt
        dist = math.hypot(state.x - obs["pos"]
                          ["x"], state.y - obs["pos"]["y"],)
        if dist > dist_thershold:
            continue
        result, nearest_corner = check_collsion_new(
            np.array([state.x, state.y]),
            car_length,
            car_width,
            state.yaw,
            np.array([obs["pos"]["x"], obs["pos"]["y"]]),
            obs["length"],
            obs["width"],
            obs["pos"]["yaw"],
        )
        if result:
            cost += math.inf
            return cost
        elif abs(nearest_corner[0]) > car_length or abs(nearest_corner[1]) > car_width:
            continue
        else:
            if abs(nearest_corner[0]) > car_length / 2:
                cost += (
                    1 - (abs(nearest_corner[0]) -
                         car_length / 2) / (car_length / 2)
                ) * config["weights"]["W_COLLISION"]
            if abs(nearest_corner[1]) > car_width / 2:
                cost += (
                    1 - (abs(nearest_corner[1]) -
                         car_width / 2) / (car_width / 2)
                ) * config["weights"]["W_COLLISION"]

    return cost


def calculate_pedestrian(vehicle: Vehicle,
                         obs: dict, trajectory: Trajectory,
                         config: dict) -> float:
    """
    Calculate the pedestrian cost of the vehicle.

    Args:
        vehicle (Vehicle): The ego vehicle object.
        obs (dict): The obstacle information.
        trajectory (Trajectory): The Trajectory object.
        config (dict): The configuration dictionary.
        T (float): The time duration.

    Returns:
        float: The pedestrian cost value.
    """
    reaction_time = 2.0  # important param for avoid pedestrian
    cost = 0

    car_width = vehicle.width
    car_length = vehicle.length

    dist_to_collide = (
        reaction_time * trajectory.states[0].vel
        + 1 * car_length  # Reaction dist + Hard Collision
    )
    for i in range(0, min(len(trajectory.states), int(reaction_time / config["DT"])), 2):
        dist = math.hypot(
            trajectory.states[i].x -
            obs["pos"]["x"], trajectory.states[i].y - obs["pos"]["y"],
        )
        if dist > dist_to_collide:
            continue

        result, nearest_corner = check_collsion_new(
            np.array([trajectory.states[i].x, trajectory.states[i].y]),
            car_length,
            car_width,
            trajectory.states[i].yaw,
            np.array([obs["pos"]["x"], obs["pos"]["y"]]),
            obs["length"],
            obs["width"] + car_width * 1.0,
            0,
        )
        if result:
            cost += math.inf
            return cost
        elif (
            nearest_corner[0] > dist_to_collide
            or nearest_corner[0] < -car_length
            or abs(nearest_corner[1]) > 1.0 * car_width
        ):
            continue
        else:
            if nearest_corner[0] < -0.5 * car_length:
                cost += (
                    1 - (nearest_corner[0] + car_length *
                         0.5) / (-car_length * 0.5)
                ) * config["weights"]["W_COLLISION"]
            if nearest_corner[0] > 0.5 * car_length:
                cost += (
                    1
                    - (nearest_corner[0] - car_length * 0.5)
                    / (dist_to_collide - car_length * 0.5)
                ) * config["weights"]["W_COLLISION"]
                # cost += config["weights"]["W_COLLISION"] * 10
            if abs(nearest_corner[1]) > car_width / 2:
                cost += (
                    1 - (abs(nearest_corner[1]) -
                         car_width / 2) / (0.5 * car_width)
                ) * config["weights"]["W_COLLISION"]

    return cost


def calculate_car(vehicle: Vehicle, obs: dict, 
                  trajectory: Trajectory, config: dict, offset_frame: int) -> float:
    """
    Calculate the car cost of the vehicle.

    Args:
        vehicle (Vehicle): The ego vehicle object.
        obs (dict): The obstacle information.
        trajectory (Trajectory): The Trajectory object.
        config (dict): The configuration dictionary.

    Returns:
        float: The car cost value.
    """
    cost = 0
    car_length = vehicle.length
    car_width = vehicle.width
    
    if vehicle.lane_id == obs.lane_id and vehicle.current_state.s > obs.current_state.s: # obs car is behind ego car on the same lane
            return cost
    # ATTENSION: for speed up, we only check every 2 points
    for i in range(0, min(len(trajectory.states), len(obs.future_trajectory.states) - offset_frame), 2):
        ego_state = trajectory.states[i]
        obs_state = obs.future_trajectory.states[i + offset_frame]
        dist = math.hypot(
            ego_state.x - obs_state.x,
            ego_state.y - obs_state.y,
        )
        dist_to_collide = (
            3 * (max(0, ego_state.vel - obs_state.vel))  # TTC
            + 0.5* ego_state.vel  # Reaction dist
            + 1 * car_length  # Hard Collision
        )
        if dist > dist_to_collide:
            # if obs far away at beginning, we don't care
            return cost
        result, nearest_corner = check_collsion_new(
            np.array([ego_state.x, ego_state.y]),
            car_length*1.5,
            car_width*1.1,
            ego_state.yaw,
            np.array([obs_state.x, obs_state.y]),
            obs.shape.length,
            obs.shape.width,
            obs_state.yaw,
        )
        if result:
            cost += math.inf
            return cost
        elif (
            nearest_corner[0] > dist_to_collide
            or nearest_corner[0] < -1.5 * car_length
            or abs(nearest_corner[1]) > 0.9 * car_width
        ):
            continue
        else:
            if abs(nearest_corner[1]) > 0.5 * car_width:
                cost += (
                    1 - (abs(nearest_corner[1]) -
                         car_width * 0.5) / (car_width * 0.2)
                ) * config["weights"]["W_COLLISION"]
            if nearest_corner[0] > 0.5 * car_length:
                cost += (
                    1
                    - (nearest_corner[0] - car_length * 0.5)
                    / (dist_to_collide - car_length * 0.5)
                ) * config["weights"]["W_COLLISION"] * 10
            if nearest_corner[0] < -0.5 * car_length:
                cost += (
                    1 - (nearest_corner[0] + car_length *
                         0.5) / (-car_length * 1.0)
                ) * config["weights"]["W_COLLISION"]

    return cost
