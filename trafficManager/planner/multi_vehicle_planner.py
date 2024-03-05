import time
from typing import Dict, List

from common.observation import Observation
from common.vehicle import Behaviour, Vehicle, VehicleType
from decision_maker.abstract_decision_maker import (
    EgoDecision,
    MultiDecision,
    SingleStepDecision,
)
from planner.abstract_planner import AbstractMultiPlanner
from predictor.abstract_predictor import Prediction

import logger
import trafficManager.planner.trajectory_generator as traj_generator
from utils.roadgraph import AbstractLane, JunctionLane, NormalLane, RoadGraph
from utils.trajectory import Trajectory, State
from utils.obstacles import DynamicObstacle, ObsType, Rectangle

logging = logger.get_logger(__name__)


class MultiVehiclePlanner(AbstractMultiPlanner):
    def plan(self,
             controlled_observation: Observation,
             roadgraph: RoadGraph,
             uncontrolled_prediction: Prediction,
             T,
             config,
             multi_decision: MultiDecision = None) -> Dict[Vehicle, Trajectory]:
        plan_result: Dict[int, Trajectory] = {}
        for vehicle in controlled_observation.vehicles:
            start = time.time()
            if vehicle.vtype == VehicleType.OUT_OF_AOI:
                continue
            if config["EGO_PLANNER"] and vehicle.vtype == VehicleType.EGO:
                continue

            current_lane = roadgraph.get_lane_by_id(vehicle.lane_id)

            obs_list = self.extract_obstacles(controlled_observation,
                                              uncontrolled_prediction,
                                              vehicle, roadgraph)
            decision_list = self.find_decision(vehicle, multi_decision, T,
                                               config)
            # Plan for current vehicle
            path = self.generate_trajectory(
                roadgraph, T, config, vehicle, current_lane, obs_list, decision_list
            )
            logging.debug(
                f"Vehicle {vehicle.id} Total planning time: {time.time() - start}"
            )
            plan_result[vehicle.id] = path

        return plan_result

    def generate_trajectory(
        self, roadgraph: RoadGraph, T, config, vehicle: Vehicle, current_lane: AbstractLane, obs_list, decision_list
    ):
        next_lane = roadgraph.get_available_next_lane(
            current_lane.id, vehicle.available_lanes
        )
        lanes = [current_lane, next_lane] if next_lane != None else [
            current_lane]

        if vehicle.behaviour == Behaviour.KL:
            if self.is_waiting_for_green_light(current_lane, next_lane):
                # Stop
                path = traj_generator.stop_trajectory_generator(
                    vehicle, lanes, obs_list, roadgraph, config, T, redLight=True
                )
            else:
                # Keep Lane
                if vehicle.current_state.s_d >= 10 / 3.6:
                    path = None
                    if config["USE_DECISION_MAKER"] and decision_list is not None:
                        path = traj_generator.decision_trajectory_generator(
                            vehicle, lanes, obs_list, config, T, decision_list,
                        )
                        if path is None:
                            logging.info(
                                "Fail to plan DECISION KL path for vehicle %s back to normal planner", vehicle.id)
                    if path is None:
                        path = traj_generator.lanekeeping_trajectory_generator(
                            vehicle, lanes, obs_list, config, T
                        )
                else:
                    path = traj_generator.stop_trajectory_generator(
                        vehicle, lanes, obs_list, roadgraph, config, T,
                    )
        elif vehicle.behaviour == Behaviour.STOP:
            # Stopping
            path = traj_generator.stop_trajectory_generator(
                vehicle, lanes, obs_list, roadgraph, config, T,
            )
        elif vehicle.behaviour == Behaviour.LCL:
            # Turn Left
            logging.debug(
                f"vehicle {vehicle.id} is planning to change to left lane")
            left_lane = roadgraph.get_lane_by_id(current_lane.left_lane())
            path = None
            if config["USE_DECISION_MAKER"] and decision_list is not None:
                path = traj_generator.decision_trajectory_generator(
                    vehicle, lanes, obs_list, config, T, decision_list,
                )
                if path is None:
                    logging.info(
                        f"Fail to plan DECISION LCL path for vehicle {vehicle.id}. Back to normal planner"
                    )
            if path is None:
                path = traj_generator.lanechange_trajectory_generator(
                    vehicle, left_lane, obs_list, config, T,
                )
        elif vehicle.behaviour == Behaviour.LCR:
            # Turn Right

            right_lane = roadgraph.get_lane_by_id(current_lane.right_lane())
            logging.debug(
                f"vehicle {vehicle.id} is planning to change from {vehicle.lane_id} to right lane {right_lane.id}")
            path = None
            if config["USE_DECISION_MAKER"] and decision_list is not None:
                path = traj_generator.decision_trajectory_generator(
                    vehicle, lanes, obs_list, config, T, decision_list,
                )
                if path is None:
                    logging.info(
                        f"Fail to plan DECISION LCR path for vehicle {vehicle.id}. Back to normal planner")
            if path is None:
                path = traj_generator.lanechange_trajectory_generator(
                    vehicle, right_lane, obs_list, config, T,
                )

        elif vehicle.behaviour == Behaviour.IN_JUNCTION:
            # in Junction. for now just stop trajectory
            path = traj_generator.stop_trajectory_generator(
                vehicle, lanes, obs_list, roadgraph, config, T,
            )
        else:
            logging.error(
                f"Vehicle {vehicle.id} has unknown behaviour {vehicle.behaviour}")

        return path

    def find_decision(
        self, vehicle, mul_decisions, T, config
    ) -> List[SingleStepDecision]:
        if mul_decisions is not None:
            for dec_vehicle, decision_list in mul_decisions.results.items():
                if (
                    dec_vehicle.id == vehicle.id
                    and decision_list[-1].expected_time - T >= config["MIN_T"]
                ):
                    return decision_list
        return None

    def is_in_intersection(
        self, current_lane: AbstractLane, next_lane: AbstractLane
    ) -> bool:
        if not isinstance(current_lane, NormalLane):
            return False
        if next_lane is None:
            return False
        if not isinstance(next_lane, JunctionLane):
            return False
        return True

    def is_waiting_for_green_light(
        self, current_lane: AbstractLane, next_lane: AbstractLane
    ) -> bool:
        """check if vehicle is waiting for green light at intersection

        Args:
            current_lane (AbstractLane): the lane vehicle lies on
            next_lane (AbstractLane): the next lane

        Returns:
            bool: True is vehicle is in intersection and waiting for green light.
        """
        if not self.is_in_intersection(current_lane, next_lane):
            return False
        return next_lane.currTlState in ("R", "r")

    def extract_obstacles(self, observation: Observation,
                          predictions, ego: Vehicle, roadgraph: RoadGraph):
        obs_list = []
        # Process static obstacle
        for obs in observation.obstacles:
            obs_list.append(obs)

        veh_edges = set()
        for lane_id in ego.available_lanes:
            lane = roadgraph.get_lane_by_id(lane_id)
            if isinstance(lane, JunctionLane):
                veh_edges.add(lane.affJunc)
            elif isinstance(lane, NormalLane):
                veh_edges.add(lane.affiliated_edge.id)
        # Process dynamic obstacles
        for vehicle, prediction in predictions.results.items():
            if vehicle.id == ego.id:
                continue
            if not prediction:
                continue
            shape = Rectangle(vehicle.length, vehicle.width)
            current_state = State(x=prediction[0].x,
                                  y=prediction[0].y,
                                  s=prediction[0].s,
                                  d=prediction[0].d,
                                  yaw=prediction[0].yaw,
                                  vel=prediction[0].vel)
            dynamic_obs = DynamicObstacle(obstacle_id=vehicle.id,
                                          shape=shape,
                                          obstacle_type=ObsType.CAR,
                                          current_state=current_state,
                                          lane_id=vehicle.lane_id)
            for i in range(1, len(prediction)):
                state = State(
                    x=prediction[i].x,
                    y=prediction[i].y,
                    s=prediction[i].s,
                    d=prediction[i].d,
                    yaw=prediction[i].yaw,
                    vel=prediction[i].vel,
                )
                dynamic_obs.future_trajectory.states.append(state)

            obs_list.append(dynamic_obs)
        return obs_list
