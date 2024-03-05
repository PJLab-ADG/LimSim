"""
Author: Licheng Wen
Date: 2022-06-15 10:19:19
Description: 
Planner for a single vehicle

Copyright (c) 2022 by PJLab, All Rights Reserved. 
"""

import math
from typing import List
import numpy as np

from common.vehicle import Vehicle
from common import cost

from utils.roadgraph import AbstractLane, JunctionLane, RoadGraph, NormalLane
from utils.obstacles import ObsType, Obstacle
from utils.trajectory import State, Trajectory

from trafficManager.planner.frenet_optimal_planner import frenet_optimal_planner
from trafficManager.decision_maker.abstract_decision_maker import SingleStepDecision

import logger

logging = logger.get_logger(__name__)


def check_path(vehicle, path):
    for state in path.states:
        if state.vel > vehicle.max_speed:  # Max speed check
            # print("Max speed violation", vehicle.id, state.vel)
            return False
        elif (state.s_dd > vehicle.max_accel or
              state.s_dd < vehicle.max_decel):  # Max acceleration check
            # print("Max accel violation ", vehicle.id, state.s_dd)
            return False
        # elif abs(state.cur) > config["MAX_CURVATURE"]:  # Max curvature check
        #     print("Max curvature exceeded")
        #     return False

    if path.cost == math.inf:  # Collision check
        return False
    else:
        return True


def lanechange_trajectory_generator(
    vehicle: Vehicle,
    target_lane: AbstractLane,
    obs_list,
    config,
    T,
) -> Trajectory:
    state_in_target_lane = vehicle.get_state_in_lane(target_lane)
    target_vel = vehicle.target_speed
    dt = config["DT"]
    d_t_sample = config["D_T_S"] / 3.6
    n_s_d_sample = config["N_D_S_SAMPLE"]
    s_sample = config["S_SAMPLE"]
    n_s_sample = config["N_S_SAMPLE"]

    sample_t = [config["MIN_T"] / 1.5]  # Sample course time
    sample_vel = np.linspace(
        max(1e-9, state_in_target_lane.vel - d_t_sample * n_s_d_sample),
        min(state_in_target_lane.vel + d_t_sample * n_s_d_sample,
            target_lane.speed_limit), 5)
    vel = min(state_in_target_lane.vel, target_vel)
    sample_s = np.empty(0)
    for t in sample_t:
        sample_s = np.append(
            sample_s,
            np.arange(
                state_in_target_lane.s + t * (max(5.0, vel)),
                state_in_target_lane.s + t *
                (target_vel + s_sample * n_s_sample * 1.01),
                s_sample,
            ),
        )

    # Step 2: Calculate Paths
    best_path = None
    best_cost = math.inf
    for t in sample_t:
        for s in sample_s:
            for s_d in sample_vel:
                target_state = State(t=t, s=s, d=0, s_d=s_d)
                path = frenet_optimal_planner.calc_spec_path(
                    state_in_target_lane, target_state, target_state.t, dt)
                if not path.states:
                    continue
                path.frenet_to_cartesian(target_lane, vehicle.current_state)
                path.cost = (
                    cost.smoothness(path, target_lane.course_spline,
                                    config["weights"]) * dt +
                    cost.vel_diff(path, target_vel, config["weights"]) * dt +
                    cost.guidance(path, config["weights"]) * dt +
                    cost.acc(path, config["weights"]) * dt +
                    cost.jerk(path, config["weights"]) * dt +
                    cost.obs(vehicle, path, obs_list, config) +
                    cost.changelane(config["weights"]))
                if not path.is_nonholonomic():
                    continue
                if path.cost < best_cost:
                    best_cost = path.cost
                    best_path = path

    if best_path is not None:
        logging.debug(
            f"Vehicle {vehicle.id} found a lane change path with cost: {best_cost}")
        return best_path

    # return stop path
    logging.info(
        f"vehicle {vehicle.id} found no lane change paths, calculating a stop path instead.")

    stop_path = frenet_optimal_planner.calc_stop_path(state_in_target_lane,
                                                      vehicle.max_decel,
                                                      sample_t[0], dt, config)
    stop_path.frenet_to_cartesian(target_lane, state_in_target_lane)
    stop_path.cost = (cost.smoothness(stop_path, target_lane.course_spline,
                                      config["weights"]) * dt +
                      cost.guidance(stop_path, config["weights"]) * dt +
                      cost.acc(stop_path, config["weights"]) * dt +
                      cost.jerk(stop_path, config["weights"]) * dt +
                      cost.stop(config["weights"]))
    return stop_path


def stop_trajectory_generator(vehicle: Vehicle,
                              lanes: List[AbstractLane],
                              obs_list: List[Obstacle],
                              roadgraph: RoadGraph,
                              config,
                              T,
                              redLight: bool = False) -> Trajectory:
    current_lane = lanes[0]
    course_spline = current_lane.course_spline
    current_state = vehicle.current_state
    course_t = config["MIN_T"]  # Sample course time
    dt = config["DT"]
    max_acc = vehicle.max_accel
    car_length = vehicle.length

    # Step 1: find the right stopping position
    s = np.linspace(
        current_state.s,
        min(
            course_spline.s[-1],
            current_state.s + current_state.s_d * course_t + 3 * car_length,
        ),
        100,
    )
    if redLight:
        min_s = s[-1] - 5
    else:
        min_s = s[-1] + 100
    for obs in obs_list:
        if obs.type == ObsType.OTHER:
            obs_s, obs_d = course_spline.cartesian_to_frenet1D(
                obs.current_state.x, obs.current_state.y)
            if obs_s == s[0] or obs_s == s[-1]:
                continue
            obs_near_d = max(0, abs(obs_d) - obs.shape.width / 2)
            if obs_near_d < current_lane.width / 2:
                min_s = min(min_s, obs_s - obs.shape.length / 2 - car_length)
        elif obs.type == ObsType.PEDESTRIAN:
            obs_s, obs_d = course_spline.cartesian_to_frenet1D(
                obs.current_state.x, obs.current_state.y)
            if obs_s == s[0] or obs_s == s[-1]:
                continue
            obs_near_d = max(0, abs(obs_d) - obs.shape.width / 2)
            if obs_near_d < current_lane.width / 1.5:
                min_s = min(min_s, obs_s - obs.shape.length / 2 - car_length)
        elif obs.type == ObsType.CAR:
            if isinstance(current_lane, JunctionLane):
                # check if in same junction
                veh_junction_id = vehicle.lane_id.split("_")[0]
                obs_junction_id = obs.lane_id.split("_")[0]
                nextlane_id = current_lane.next_lane_id
                if veh_junction_id != obs_junction_id and (
                    obs.lane_id != nextlane_id
                    or (
                        obs.lane_id == nextlane_id
                        and obs.current_state.s >= course_spline.s[-1]
                    )
                ):
                    continue
                if (
                    obs.lane_id in (nextlane_id, vehicle.lane_id)
                    and abs(obs.current_state.vel - vehicle.current_state.vel) < 0.5
                ):
                    continue

                for i in range(0, min(len(obs.future_trajectory.states), 20), 3):
                    obs_s, obs_d = course_spline.cartesian_to_frenet1D(
                        obs.future_trajectory.states[i].x,
                        obs.future_trajectory.states[i].y)
                    if obs_s <= s[0] or obs_s >= s[-1]:
                        next_lane = roadgraph.get_lane_by_id(
                            current_lane.next_lane_id)
                        nextlane_spline = next_lane.course_spline
                        obs_s, obs_d = nextlane_spline.cartesian_to_frenet1D(
                            obs.future_trajectory.states[i].x,
                            obs.future_trajectory.states[i].y)
                        obs_s += current_lane.course_spline.s[-1]
                    obs_near_d = max(0, abs(obs_d) - obs.shape.width / 2)
                    if obs_near_d < current_lane.width / 2:
                        min_s = min(min_s, obs_s -
                                    obs.shape.length - car_length)
            else:  # in normal lane
                if isinstance(roadgraph.get_lane_by_id(obs.lane_id), NormalLane):
                    edge_1 = current_lane.affiliated_edge
                    edge_2 = roadgraph.get_lane_by_id(
                        obs.lane_id).affiliated_edge
                    if edge_1 != edge_2:
                        continue
                    obs_state_on_current_lane = obs.update_frenet_coord_in_lane(
                        current_lane)
                    obs_s, obs_d = obs_state_on_current_lane.s, obs_state_on_current_lane.d
                    if obs_s <= s[0] or obs_s >= s[-1]:
                        continue
                    obs_near_d = max(0, abs(obs_d) - obs.shape.width / 2)
                    if obs_s > vehicle.current_state.s and obs_near_d < current_lane.width / 2:
                        # 2.0 meter as a constant parking distance
                        min_s = min(
                            min_s, obs_s - obs.shape.length / 2 - car_length / 2 - 2.0
                        )
                # if obs.lane_id == current_lane.id:
                #     obs_s, obs_d = obs.current_state.s, obs.current_state.d
                #     if obs_s <= s[0] or obs_s >= s[-1]:
                #         continue
                #     obs_near_d = max(0, abs(obs_d) - obs.shape.width / 2)
                #     if obs_near_d < current_lane.width / 2:
                #         # 2.0 meter as a constant parking distance
                #         min_s = min(
                #             min_s, obs_s - obs.shape.length / 2 - car_length / 2 - 2.0
                #         )
                else:
                    # next_lane = roadgraph.get_available_next_lane(
                    #     current_lane.id, vehicle.available_lanes
                    # )
                    next_lanes = list(current_lane.next_lanes.values())
                    next_lanes = [l[0] for l in next_lanes]
                    if (
                        next_lanes is not None
                        and obs.lane_id in next_lanes
                        and obs.current_state.s <= obs.shape.length + vehicle.length
                    ):
                        min_s = min(
                            min_s,
                            current_lane.course_spline.s[-1]
                            - obs.shape.length / 2
                            - car_length / 2
                            - 2.0,
                        )
    # Step 2:
    path = Trajectory()
    if (current_state.vel <= 1.0 and
            (min_s - current_state.s) <= car_length):  # already stopped, keep it
        logging.debug(f"Vehicle {vehicle.id} Already stopped")
        path = Trajectory()
        for t in np.arange(0, course_t, dt):
            path.states.append(
                State(t=t, s=current_state.s, d=current_state.d))
        path.frenet_to_cartesian(lanes, vehicle.current_state)
        path.cost = (
            cost.smoothness(path, lanes[0].course_spline, config["weights"]) *
            dt + cost.guidance(path, config["weights"]) * dt +
            cost.acc(path, config["weights"]) * dt +
            cost.jerk(path, config["weights"]) * dt)
        return path
    if ((min_s - current_state.s) >
            current_state.s_d * course_t / 1.5):  # no need to stop
        logging.debug(f"Vehicle {vehicle.id} No need to stop")
        if (min_s - current_state.s) < 5.0 / 3.6 * course_t:
            target_s = min_s
            target_state = State(s=target_s, s_d=5.0 / 3.6, d=0)
        else:
            # 20 km/h is the speed limit in junction lane
            target_vel = min(20.0 / 3.6, lanes[0].speed_limit)
            target_s = (current_state.s +
                        (current_state.s_d +
                         (target_vel - current_state.s_d) / 1.3) * course_t)
            target_state = State(s=target_s, s_d=target_vel, d=0)
            if target_vel > current_state.s_d:
                current_state.s_dd = max(1e-2, current_state.s_dd)

        path = frenet_optimal_planner.calc_spec_path(current_state,
                                                     target_state, course_t, dt
                                                     )
        # print("no need path", [state.s for state in path.states], [
        #       state.s_d for state in path.states])
        path.frenet_to_cartesian(lanes, current_state)
        path.cost = (
            cost.smoothness(path, lanes[0].course_spline, config["weights"]) *
            dt + cost.guidance(path, config["weights"]) * dt +
            cost.acc(path, config["weights"]) * dt +
            cost.jerk(path, config["weights"]) * dt)
        return path
    elif (min_s - current_state.s) < max(current_state.s_d**2 / (2 * max_acc),
                                         car_length / 4):  # need emergency stop
        logging.debug(f"Vehicle {vehicle.id} Emergency Brake")
        path = frenet_optimal_planner.calc_stop_path(current_state,
                                                     vehicle.max_decel,
                                                     course_t, dt, config)
        path.frenet_to_cartesian(lanes, current_state)
        path.cost = (
            cost.smoothness(path, lanes[0].course_spline, config["weights"]) *
            dt + cost.guidance(path, config["weights"]) * dt +
            cost.acc(path, config["weights"]) * dt +
            cost.jerk(path, config["weights"]) * dt +
            cost.stop(config["weights"]))
        return path

    # normal stop
    logging.debug(f"Vehicle {vehicle.id} Normal stopping")
    if (min_s - current_state.s) < car_length:
        sample_d = [current_state.d]
    else:
        sample_d = [0]
        # sample_d = np.arange(-road_width / 2, road_width / 2 * 1.01, d_road_w)
    sample_stop_t = np.linspace(max(current_state.s_d / 3.0, 0.1),
                                max(current_state.s_d / 1.0, 0.1), 4)
    best_path = None
    best_cost = math.inf
    for d in sample_d:
        for stop_t in sample_stop_t:
            target_state = State(s=min_s, s_d=0, d=d)
            path = frenet_optimal_planner.calc_spec_path(
                current_state, target_state, stop_t, dt)
            t = path.states[-1].t
            s = path.states[-1].s
            d = path.states[-1].d
            while len(path.states) < course_t / dt:
                t += dt
                path.states.append(State(t=t, s=s, d=d))

            path.frenet_to_cartesian(lanes, current_state)
            path.cost = (cost.smoothness(path, lanes[0].course_spline,
                                         config["weights"]) * dt +
                         cost.guidance(path, config["weights"]) * dt +
                         cost.jerk(path, config["weights"]) * dt +
                         cost.stop(config["weights"]))
            if path.cost < best_cost:
                best_cost = path.cost
                best_path = path
    return best_path


def lanekeeping_trajectory_generator(vehicle: Vehicle,
                                     lanes: List[AbstractLane], obs_list,
                                     config, T) -> Trajectory:
    road_width = lanes[0].width
    current_state = vehicle.current_state
    target_vel = vehicle.target_speed

    # Step 1: Sample target states
    d_road_w = config["D_ROAD_W"]
    d_t_sample = config["D_T_S"] / 3.6
    n_s_d_sample = config["N_D_S_SAMPLE"]
    dt = config["DT"]

    sample_d = np.linspace(-road_width / 3,
                           road_width / 3,
                           num=int(road_width / d_road_w) +
                           1)  # sample target lateral offset
    sample_d = sample_d[sample_d != 0]
    center_d = [0]
    sample_t = [config["MIN_T"]]  # Sample course time
    # sample target longtitude vel(Velocity keeping)
    # decelerate when traveling into next road segement
    if current_state.vel * sample_t[0] > lanes[0].course_spline.s[
            -1] - current_state.s:
        speed_limit = 25 / 3.6
        if len(lanes) > 1:
            speed_limit = min(speed_limit, lanes[1].speed_limit)
        sample_vel = np.linspace(
            min(current_state.vel, 10 / 3.6), speed_limit, 4)
    else:
        sample_vel = np.linspace(
            max(1e-9, current_state.vel - d_t_sample * n_s_d_sample),
            min(
                max(current_state.vel, target_vel) +
                d_t_sample * n_s_d_sample * 1.01, lanes[0].speed_limit),
            5,
        )

    # Step 2: Generate Center line trajectories
    center_paths = frenet_optimal_planner.calc_frenet_paths(
        current_state, center_d, sample_t, sample_vel, dt, config)
    best_path = None
    best_cost = math.inf
    if center_paths is not None:
        for path in center_paths:
            path.frenet_to_cartesian(lanes, current_state)
            path.cost = (
                cost.smoothness(
                    path, lanes[0].course_spline, config["weights"])
                * dt + cost.vel_diff(path, target_vel, config["weights"]) * dt +
                cost.guidance(path, config["weights"]) * dt +
                cost.acc(path, config["weights"]) * dt +
                cost.jerk(path, config["weights"]) * dt +
                cost.obs(vehicle, path, obs_list, config))
            if check_path(vehicle, path) and path.cost < best_cost:
                best_cost = path.cost
                best_path = path

    if best_path is not None:
        return best_path

    # Step 3: If no valid path, Generate nudge trajectories
    paths = frenet_optimal_planner.calc_frenet_paths(current_state, sample_d,
                                                     sample_t, sample_vel, dt,
                                                     config)
    best_cost = math.inf
    if paths is not None:
        for path in paths:
            path.frenet_to_cartesian(lanes, current_state)
            path.cost = (
                cost.smoothness(
                    path, lanes[0].course_spline, config["weights"])
                * dt + cost.vel_diff(path, target_vel, config["weights"]) * dt +
                cost.guidance(path, config["weights"]) * dt +
                cost.acc(path, config["weights"]) * dt +
                cost.jerk(path, config["weights"]) * dt +
                cost.obs(vehicle, path, obs_list, config))
            if check_path(vehicle, path) and path.cost < best_cost:
                best_cost = path.cost
                best_path = path

    if best_path is not None:
        logging.debug(
            f"Vehicle {vehicle.id} finds a lanekeeping NUDGE path with minimum cost: {best_cost}"
        )
        return best_path

    # Step 4: if no nudge path is found, Calculate a emergency stop path
    logging.debug(
        f"Vehicle {vehicle.id} No lane keeping path found, Calculate a emergency brake path.{vehicle.max_decel}"
    )

    stop_path = frenet_optimal_planner.calc_stop_path(current_state,
                                                      vehicle.max_decel*1.5,
                                                      sample_t[0], dt, config)
    stop_path.frenet_to_cartesian(lanes, current_state)
    stop_path.cost = (
        cost.smoothness(
            stop_path, lanes[0].course_spline, config["weights"]) * dt
        + cost.guidance(stop_path, config["weights"]) * dt
        + cost.acc(stop_path, config["weights"]) * dt
        + cost.jerk(stop_path, config["weights"]) * dt
        + cost.stop(config["weights"])
    )
    return stop_path


def decision_trajectory_generator(
    vehicle: Vehicle,
    lanes: List[AbstractLane],
    obs_list,
    config,
    T,
    decision_list: List[SingleStepDecision],
) -> Trajectory:
    d_road_w = config["D_ROAD_W"]
    d_vel = config["D_T_S"] / 3.6
    dt = config["DT"]

    fullpath = Trajectory()
    current_time = T
    current_state = vehicle.current_state
    has_lane_change = False
    for (idx, decision) in enumerate(decision_list):
        if T >= decision.expected_time:
            # decision time is in the past
            continue
        if (
            idx + 1 < len(decision_list)
            and decision_list[idx + 1].action == decision.action
        ):
            # skip for the same action
            continue
        # if lane change
        if decision.action == "LCL" or decision.action == "LCR" or has_lane_change:
            if decision.expected_state.s > lanes[0].spline_length:
                # vehicle is out of the lane
                break
            decision.expected_state.s, decision.expected_state.d = lanes[
                0
            ].course_spline.cartesian_to_frenet1D(
                decision.expected_state.x,
                decision.expected_state.y)
            has_lane_change = True

        seg_time = decision.expected_time - current_time
        sample_d = np.linspace(
            decision.expected_state.d - d_road_w,
            decision.expected_state.d + d_road_w,
            5,
        )
        sample_vel = np.linspace(
            max(1e-9, decision.expected_state.vel - 3 * d_vel),
            min(decision.expected_state.vel + d_vel, lanes[0].speed_limit),
            10,
        )

        seg_paths = frenet_optimal_planner.calc_frenet_paths(
            current_state, sample_d, [seg_time], sample_vel, dt, config
        )
        offset_frame = len(fullpath.states)
        best_path = None
        best_cost = math.inf
        for path in seg_paths:
            path.frenet_to_cartesian(lanes, current_state)
            path.cost = (
                cost.smoothness(
                    path, lanes[0].course_spline, config["weights"]) * dt
                + cost.vel_diff(path, vehicle.target_speed,
                                config["weights"]) * dt
                + cost.guidance(path, config["weights"]) * dt
                + cost.acc(path, config["weights"]) * dt
                + cost.jerk(path, config["weights"]) * dt
                + cost.obs(vehicle, path, obs_list, config, offset_frame)
            )
            if check_path(vehicle, path) and path.cost < best_cost:
                best_cost = path.cost
                best_path = path

        if best_path is not None:
            current_state = best_path.states[-1]
            fullpath.concatenate(best_path)
            current_time = decision.expected_time
        else:
            fullpath = None
            logging.info(
                "cannot generate path for id %s at traj_idx %d, decision %s",
                vehicle.id, idx, decision.action,
            )
            break
        if (
            current_time - T > config["MIN_T"]
            or current_time - T > fullpath.states[-1].t
        ):
            # finish planning
            break

    if fullpath is not None and len(fullpath.states) > 0:
        return fullpath
    else:  # no valid path found
        return None


if __name__ == "__main__":
    pass
