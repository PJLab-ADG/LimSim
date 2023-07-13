"""
generate real time scenario evluation results
"""

from typing import List
import numpy as np
from evaluation.math_utils import normalize
from simModel.common.carFactory import Vehicle

from utils.roadgraph import AbstractLane, NormalLane
from utils.obstacles import Rectangle


class RealTimeEvaluation():
    """A real time evaluation class for revealing ego state
    """

    def __init__(self, dt: float = 0.1) -> None:
        self.dt: float = dt
        self.ego: Vehicle = None
        self.current_lane: AbstractLane = None
        self.agents: List[Vehicle] = None
        self.ref_yaw: float = 0.0
        self.result: np.ndarray = None
        self.new_yaw: float = 0.0

    def update_data(self, ego: Vehicle, current_lane: AbstractLane,
                    agents: List[Vehicle]):
        """Use frame information to update the data

        Args:
            current_state (State): current state of ego car
            current_lane (AbstractLane): the lane which ego car lies on
            agents (List[Vehicle]): agents in the current lane
            previous_state (State): state of ego car in last frame
        """
        self.ego = ego
        self.current_lane = current_lane
        self.agents = agents

    def _evaluate_orientation(self) -> float:
        """The orientation offset with respect to center line of lane

        Returns:
            float: orientation offset
        """
        delay = 0.1

        if isinstance(self.current_lane, NormalLane):
            # use delay to prevent sudden change of value
            # new_yaw can be seen as a combination of current ref_yaw and ref_yaw
            # of last NormalLane (before 10 frames)
            if len(self.ego.laneIDQ) >= 11 and self.ego.laneIDQ[-1] != self.ego.laneIDQ[-11]:
                ref_yaw = self.current_lane.course_spline.calc_yaw(
                    self.ego.lanePos)
                self.new_yaw += delay * (ref_yaw - self.ref_yaw)
                if abs(ref_yaw - self.new_yaw) < 1e-3:
                    self.ref_yaw = ref_yaw
            else:
                self.ref_yaw = self.current_lane.course_spline.calc_yaw(
                    self.ego.lanePos)
                self.new_yaw = self.ref_yaw

        result = (np.sin(normalize(self.ego.yaw)) -
                  np.sin(normalize(self.new_yaw)))**2
        return result

    def _evaluate_discomfort(self) -> float:
        """the jerk of the ego car 

        Returns:
            float: jerk
        """
        if len(self.ego.accelQ) < 2:
            return 0.0

        jerk: float = (self.ego.accelQ[-1] - self.ego.accelQ[-2]) / self.dt
        return jerk**2

    def _evaluate_consumption(self) -> float:
        """the instantaneous energy consumption of ego car, measured by (v * a) ** 2

        Returns:
            float: result
        """
        return (self.ego.speed * self.ego.accel)**2 + self.ego.speed**3

    def _evaluate_collision_risk(self) -> float:
        """the time-to-collision (ttc) to the agent which is in front of and slower than ego vehicle

        Returns:
            float: ttc
        """
        threshold = 20.0  # (s), represent infinity
        delta_t = 0.1  # (s), discretize range [0, threshold]

        ego_center = np.array([self.ego.x, self.ego.y])
        ego_velocity = self.ego.speed * np.array([np.cos(self.ego.yaw),
                                                  np.sin(self.ego.yaw)])
        ego_box = Rectangle(width=self.ego.width,
                            length=self.ego.length,
                            yaw=self.ego.yaw)

        ego_long_center = ego_center + ego_velocity * threshold / 2
        ego_long_box = Rectangle(width=self.ego.width,
                                 length=self.ego.length +
                                 np.hypot(*ego_velocity) * threshold,
                                 yaw=self.ego.yaw)

        agent_centers = np.array([[agent.x, agent.y] for agent in self.agents])
        agent_boxes = np.array([
            Rectangle(width=agent.width,
                      length=agent.length,
                      yaw=agent.yaw)
            for agent in self.agents
        ])
        agent_velocities = np.array([[
            agent.speed * np.cos(agent.yaw), agent.speed * np.sin(agent.yaw)
        ] for agent in self.agents])
        agent_long_centers = agent_centers + agent_velocities * threshold / 2
        agent_long_boxes = np.array([
            Rectangle(width=agent.width,
                      length=agent.length + np.hypot(*velocity) * threshold,
                      yaw=agent.yaw)
            for velocity, agent in zip(agent_velocities, self.agents)
        ])

        # check if trajectory (in current velocity and orientation) intersects
        possible_agent_mask = np.where([
            ego_long_box.in_collision(ego_long_center, agent_long_box,
                                      agent_long_center)
            for agent_long_box, agent_long_center in zip(
                agent_long_boxes, agent_long_centers)
        ])[0]

        # no potential collision
        if possible_agent_mask.size == 0:
            return threshold

        # iterate all possible ttc
        for t in np.arange(0, threshold, delta_t):
            if np.any([
                    ego_box.in_collision(ego_center, agent_box, agent_center)
                    for agent_box, agent_center in zip(
                        agent_boxes[possible_agent_mask],
                        agent_centers[possible_agent_mask])
            ]):
                return t

            ego_center += ego_velocity * delta_t
            agent_centers += agent_velocities * delta_t

        return threshold

    def _evaluate_offset(self) -> float:
        """the offset to the center line of current lane, measured by d**2

        Returns:
            float: the square of lateral offset with respect to center line
        """
        if self.ego.lanePos > self.current_lane.spline_length:
            return 0.0

        # compute d coordinate
        if self.ego.lanePos < self.ego.width / 2:
            return 0.0
        _, d = self.current_lane.course_spline.cartesian_to_frenet1D(
            self.ego.x, self.ego.y)
        return d**2

    def normalize_result(self) -> np.ndarray:
        points = np.zeros_like(self.result)
        # offset
        points[0] = min(self.result[0] / 1.75**2, 1.0) * 4
        # discomfort
        points[1] = min(self.result[1] / 18, 1.0) * 4
        # collision
        points[2] = 4.0 - 0.15 * self.result[2]
        # orientation
        points[3] = min(self.result[3], 1.0) * 4
        # consumption
        points[4] = min(self.result[4] / 4000, 1.0) * 4

        return np.maximum(points, 1.0)

    def output_result(self) -> List[List[float]]:
        """evaluate the scenario evaluation result and output to a dict.

        Returns:
            List[List[float]] (5, 2): An array of five elements, 
            representing coordinates of different dimensions.
        """
        self.result = np.array([
            self._evaluate_offset(),
            self._evaluate_discomfort(),
            self._evaluate_collision_risk(),
            self._evaluate_orientation(),
            self._evaluate_consumption()
        ])
        points = self.normalize_result()
        return points.tolist()
