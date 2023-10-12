"""
Author: Licheng Wen
Date: 2022-11-08 10:22:23
Description: 
Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""
from __future__ import annotations
from utils.roadgraph import AbstractLane
from utils.cubic_spline import Spline2D
import numpy as np
from dataclasses import dataclass, field
from collections import deque
import warnings
import math

import logger

logging = logger.get_logger(__name__)


# from typing_extensions import Self


class Rectangle:

    def __init__(self, center: list[float], length: float, width: float,
                 yaw: float) -> None:
        self.center = np.array(center)
        self.length = length
        self.width = width
        self.yaw = yaw
        self.rotateMat = self.getRotateMat()
        self.iRotate = np.linalg.inv(self.rotateMat)
        self.sCorners = self.getSCorners()
        self.corners = self.getCorners()

    def getRotateMat(self) -> np.ndarray:
        return np.array([[np.cos(self.yaw), -np.sin(self.yaw)],
                         [np.sin(self.yaw), np.cos(self.yaw)]],
                        dtype=np.float32)

    # shape corners
    def getSCorners(self) -> np.ndarray:
        return np.array([[self.length / 2, self.width / 2],
                         [-self.length / 2, self.width / 2],
                         [-self.length / 2, -self.width / 2],
                         [self.length / 2, -self.width / 2]])

    def getCorners(self) -> np.ndarray:
        TRCorners = self.sCorners.T
        rotCorners: np.ndarray = np.dot(self.rotateMat, TRCorners)
        return rotCorners.T + self.center


class RecCollide:

    def __init__(self, recA: Rectangle, recB: Rectangle) -> None:
        self.recA = recA
        self.recB = recB

    def rotateTransCritic(self, A: Rectangle, B: Rectangle) -> np.ndarray:
        aBase: np.ndarray = A.iRotate.dot(A.corners.T)
        aBase = aBase.T
        aPos = A.iRotate.dot(A.center)

        bBase: np.ndarray = A.iRotate.dot(B.corners.T)
        bBase = bBase.T
        bPos = A.iRotate.dot(B.center)

        posTotal = abs(bPos - aPos)
        rotateC = A.iRotate.dot(B.rotateMat)
        bh = np.array([abs(item) for item in rotateC
                       ]).dot(np.array([B.length / 2, B.width / 2]))

        return posTotal - bh - np.array([A.length / 2, A.width / 2])

    def isCollide(self) -> bool:
        critic1 = self.rotateTransCritic(self.recA, self.recB)

        if critic1[0] < 0 and critic1[1] < 0:
            critic2 = self.rotateTransCritic(self.recB, self.recA)
            if critic2[0] < 0 and critic2[1] < 0:
                return True
        return False


@dataclass
class CertesianState:
    t: float = 1.0
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    cur: float = 0.0
    vel: float = 0.0
    acc: float = 0.0


@dataclass
class FrenetState:
    laneID: str = None  # lane ID
    t: float = 2.0
    s: float = 0.0  # lane pos
    routeIdx: int = 0
    s_d: float = 0.0
    s_dd: float = 0.0
    s_ddd: float = 0.0
    d: float = 0.0
    d_d: float = 0.0
    d_dd: float = 0.0
    d_ddd: float = 0.0


@dataclass
class State(CertesianState, FrenetState):
    """State of the trajectory."""

    def __post_init__(self):
        if self.vel == 0 and self.s_d != 0:
            self.vel = math.sqrt(self.s_d**2 + self.d_d**2)

    """
    Modified from: https://blog.csdn.net/u013468614/article/details/108748016
    """

    def complete_cartesian2D(self, rx: float, ry: float, ryaw: float,
                             rkappa: float) -> None:
        cos_theta_r = math.cos(ryaw)
        sin_theta_r = math.sin(ryaw)

        self.x = rx - sin_theta_r * self.d
        self.y = ry + cos_theta_r * self.d
        if self.s_d <= 1e-1:
            self.s_d = 1e-1
            self.vel = 1e-1
            self.yaw = None
        else:
            one_minus_kappa_r_d = 1 - rkappa * self.d
            self.vel = math.sqrt(one_minus_kappa_r_d**2 * self.s_d**2 +
                                 self.d_d**2)
            self.yaw = math.asin(self.d_d / self.vel) + ryaw
        return

    def complete_frenet2D(self, rs: float, rx: float, ry: float, ryaw: float,
                          rkappa: float) -> None:
        self.s = rs
        dx = self.x - rx
        dy = self.y - ry

        cos_theta_r = math.cos(ryaw)
        sin_theta_r = math.sin(ryaw)
        cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
        self.d = math.copysign(math.sqrt(dx * dx + dy * dy), cross_rd_nd)

        delta_theta = self.yaw - ryaw
        sin_delta_theta = math.sin(delta_theta)
        cos_delta_theta = math.cos(delta_theta)
        one_minus_kappa_r_d = 1 - rkappa * self.d
        self.s_d = self.vel * cos_delta_theta / one_minus_kappa_r_d
        self.d_d = self.vel * sin_delta_theta
        return


@dataclass
class Trajectory:
    """Trajectory class."""

    states: list[State] = field(default_factory=list)
    cost: float = 0.0

    def __len__(self):
        return len(self.states)

    def pop_last_state(self) -> tuple:
        """
        return the last state of the trajectory:
        x, y, yaw, vel, acc, laneID, lanPos
        """
        last_state = self.states.pop(0)
        return last_state.x, last_state.y, last_state.yaw, last_state.vel, last_state.acc

    def pop_last_state_r(self) -> tuple:
        """
        return the last state of the trajectory for replay model:
        x, y, yaw, vel, acc, laneID, lanPos, routeIdx
        """
        last_state = self.states.pop(0)
        return last_state.x, last_state.y, last_state.yaw, last_state.vel, last_state.acc, last_state.laneID, last_state.s, last_state.routeIdx

    @property
    def xQueue(self) -> deque[float]:
        return deque([state.x for state in self.states])

    @property
    def yQueue(self) -> deque[float]:
        return deque([state.y for state in self.states])

    @property
    def yawQueue(self) -> deque[float]:
        return deque([state.yaw for state in self.states])

    @property
    def velQueue(self) -> deque[float]:
        return deque([state.vel for state in self.states])

    @property
    def accQueue(self) -> deque[float]:
        return deque([state.acc for state in self.states])

    @property
    def laneIDQueue(self) -> deque[str]:
        return deque([state.laneID for state in self.states])

    @property
    def lanePosQueue(self) -> deque[float]:
        return deque([state.s for state in self.states])

    @property
    def routeIdxQueue(self) -> deque[float]:
        return deque([state.routeIdx for state in self.states])

    @staticmethod
    def concatenate(list_of_trajectories: list[Trajectory]) -> Trajectory:
        """Concatenate a list of trajectories into one trajectory."""
        states = []
        cost = 0
        t = 0
        for trajectory in list_of_trajectories:
            cost += trajectory.cost
            for state in trajectory.states:
                state.t += t
                states.append(state)
            t = states[-1].t if len(states) > 0 else t
        return Trajectory(states, cost)

    def concatenate(self, other_traj):
        if self.states != []:
            t = self.states[-1].t
        else:
            t = 0
            self.states.append(other_traj.states[0])
        for i in range(1, len(other_traj.states)):
            self.states.append(other_traj.states[i])
            self.states[-1].t += t
        self.cost += other_traj.cost

    def frenet_to_cartesian(self, lanes: list[AbstractLane],
                            init_state: State) -> None:
        warnings.filterwarnings("error")
        if not isinstance(lanes, list):
            lanes = [lanes]
        lane_idx = 0
        already_s = 0
        for i in range(len(self.states)):
            csp = lanes[lane_idx].course_spline
            if self.states[i].s - already_s > csp.s[-1] - 0.1:
                if lane_idx < len(lanes) - 1:
                    lane_idx += 1
                    # caution: 0.1 is the overlap length
                    already_s += csp.s[-1] - 0.1
                    csp = lanes[lane_idx].course_spline
                else:
                    del self.states[i:]
                    break
            rx, ry = csp.calc_position(self.states[i].s - already_s)
            ryaw = csp.calc_yaw(self.states[i].s - already_s)
            rkappa = csp.calc_curvature(self.states[i].s - already_s)
            self.states[i].complete_cartesian2D(rx, ry, ryaw, rkappa)
            self.states[i].laneID = lanes[lane_idx].id

        # deal with all state with state.yaw =none
        for i in range(len(self.states)):
            if self.states[i].yaw is None:
                if i == 0:
                    self.states[i].yaw = init_state.yaw
                else:
                    self.states[i].yaw = self.states[i - 1].yaw

        for i in range(len(self.states)):
            if len(self.states) == 1:
                self.states[0].acc = init_state.acc
                break
            if i == len(self.states) - 1:
                self.states[-1].acc = self.states[-2].acc
            else:
                self.states[i].acc = (self.states[i + 1].vel - self.states[i].vel
                                      ) / (self.states[i + 1].t - self.states[i].t)

        # https://blog.csdn.net/m0_37454852/article/details/86514444
        # https://baike.baidu.com/item/%E6%9B%B2%E7%8E%87/9985286
        for i in range(1, len(self.states) - 1):
            try:
                dy = ((self.states[i + 1].y - self.states[i].y) /
                      (self.states[i + 1].x - self.states[i].x) +
                      (self.states[i].y - self.states[i - 1].y) /
                      (self.states[i].x - self.states[i - 1].x)) / 2
                ddy = ((self.states[i + 1].y - self.states[i].y) /
                       (self.states[i + 1].x - self.states[i].x) -
                       (self.states[i].y - self.states[i - 1].y) /
                       (self.states[i].x - self.states[i - 1].x)) / (
                           (self.states[i + 1].x - self.states[i - 1].x) / 2)
            except RuntimeWarning:
                self.states[i].cur = 0
                continue
            k = abs(ddy) / (1 + dy**2)**1.5
            self.states[i].cur = k
        # insert the first and last point
        if (len(self.states) >= 3):
            self.states[0].cur = self.states[1].cur
            self.states[-1].cur = self.states[-2].cur

        return

    def cartesian_to_frenet(self, csp: Spline2D) -> None:
        """
        Where s is by default monotonically increasing in the direction of the trajectory, and only the s,s',d,d' coordinates are updated
        """
        for index, state in enumerate(self.states):
            rs = csp.find_nearest_rs(state.x, state.y)

            # Step 2: cartesian_to_frenet1D
            rx, ry = csp.calc_position(rs)
            ryaw = csp.calc_yaw(rs)
            rkappa = csp.calc_curvature(rs)
            state.complete_frenet2D(rs, rx, ry, ryaw, rkappa)

    def is_nonholonomic(self) -> bool:
        return all([state.s_d < 1.5 * state.d_d] for state in self.states)
