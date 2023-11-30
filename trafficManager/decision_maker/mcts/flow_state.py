"""
Author: Licheng Wen
Date: 2023-04-13 16:44:31
Description: 
Flow state is used to represent the state of the flow in the MCTS tree.

Copyright (c) 2023 by PJLab, All Rights Reserved. 
"""
import itertools
import math
import random
from typing import Dict

import numpy as np
from common.vehicle import State, Vehicle, Behaviour, VehicleType
from common.obstacle_cost import check_collsion_new
from utils.roadgraph import RoadGraph, NormalLane, JunctionLane
from utils import data_copy
from abstract_decision_maker import MultiDecision
from predictor.abstract_predictor import Prediction


class FlowState:
    """
    states_list:[[veh_1,veh_2,...],[veh_1,veh_2,...],...]

    actions:{'id1':[action1,action2,...],'id2':[action1,action2,...],...}
    """

    def __init__(
        self,
        states_list: list,
        road_graph: RoadGraph,
        actions: Dict[str, list],
        complete_decisions: MultiDecision,
        prediction: Prediction,
        time: float,
        config: Dict,
    ) -> None:
        self.states_list = states_list
        self.time = time
        self.road_graph = road_graph
        self.actions = actions
        self.complete_decisions = complete_decisions
        self.prediction = prediction
        self.num_moves = None
        self.config = config

        self.next_action = []
        if (
            self.time >= self.config["MAX_DECISION_TIME"]
            or len(self.states_list[-1]) == 0
        ):
            return

        decision_idx = int(self.time // self.config["DECISION_RESOLUTION"])
        prediction_idx = int(self.time // self.config["DT"])

        # detect collision for states and prediction states
        # todo: extract as a function
        current_vehs = self.states_list[-1]
        for i, decision_veh in enumerate(current_vehs):
            current_state = decision_veh.current_state
            for other_veh, decisions in self.complete_decisions.results.items():
                if decision_idx < len(decisions):
                    is_collide = self._check_collision(
                        decision_veh,
                        current_state,
                        other_veh,
                        decisions[decision_idx].expected_state,
                    )
                    if is_collide:
                        self.num_moves = 0
                        return
            for other_veh, states in self.prediction.results.items():
                if other_veh.vtype != VehicleType.OUT_OF_AOI:
                    continue
                if prediction_idx < len(states):
                    is_collide = self._check_collision(
                        decision_veh, current_state, other_veh, states[prediction_idx]
                    )
                    if is_collide:
                        self.num_moves = 0
                        return
            for idx in range(i):
                other_decision_veh = current_vehs[idx]
                is_collide = self._check_collision(
                    decision_veh,
                    current_state,
                    other_decision_veh,
                    other_decision_veh.current_state,
                )
                if is_collide:
                    self.num_moves = 0
                    return

        # available actions for vehicles
        actions_list = []
        for veh in current_vehs:
            lane = road_graph.get_lane_by_id(veh.lane_id)
            actions = []
            if (
                (veh.lane_id not in veh.available_lanes)
                or abs(veh.current_state.d) >= lane.width / 4
                or veh.behaviour == Behaviour.OVERTAKE
            ):
                # need to lane change
                if (
                    isinstance(lane, NormalLane)
                    and lane.spline_length - veh.current_state.s
                    > veh.current_state.vel * self.config["DECISION_RESOLUTION"]
                ):
                    # enough space to change lane
                    actions.extend(["KS", "AC", "DC"])
                    if (
                        veh.behaviour == Behaviour.OVERTAKE
                        or veh.behaviour == Behaviour.LCL
                    ) and (
                        (lane.left_lane() is not None)
                        or lane.width / 2 - veh.current_state.d
                        >= self.config["LATERAL_SPEED"]
                        * self.config["DECISION_RESOLUTION"]
                    ):
                        actions.append("LCL")
                    if (
                        veh.behaviour == Behaviour.OVERTAKE
                        or veh.behaviour == Behaviour.LCR
                    ) and (
                        lane.right_lane() is not None
                        or veh.current_state.d + lane.width / 2
                        >= self.config["LATERAL_SPEED"]
                        * self.config["DECISION_RESOLUTION"]
                    ):
                        actions.append("LCR")
                else:
                    actions.append("DC")
            else:
                # alreadty in the available lane
                actions.extend(["KS", "AC", "DC"])
            actions_list.append(actions)

        self.next_actions = list(itertools.product(*actions_list))
        self.num_moves = len(self.next_actions)
        return

    def next_state(self, check_tried=False):
        next_action = random.choice(self.next_actions)
        if check_tried:
            self.next_actions.remove(next_action)
        next_time = self.time + self.config["DECISION_RESOLUTION"]
        # actions_next_step = copy.deepcopy(self.actions)
        actions_next_step = data_copy.deepcopy(self.actions)
        vehs_next_step = []

        current_vehs = self.states_list[-1]
        for idx, veh in enumerate(current_vehs):
            # veh_next_step = copy.deepcopy(veh)
            veh_next_step = data_copy.deepcopy(veh)
            veh_next_state = veh_next_step.current_state
            action = next_action[idx]
            lane = self.road_graph.get_lane_by_id(veh.lane_id)
            if action == "KS":
                veh_next_state.s += (
                    veh_next_state.vel * self.config["DECISION_RESOLUTION"]
                )
                if abs(veh_next_state.d) < lane.width / 4:
                    # allow for centreline adjustment
                    veh_next_state.d = 0
            elif action == "AC":
                veh_next_state.vel += (
                    self.config["DEFAULT_ACC"] *
                    self.config["DECISION_RESOLUTION"]
                )
                veh_next_state.vel = min(
                    veh_next_state.vel, veh_next_step.max_speed)
                veh_next_state.s += (
                    veh_next_state.vel * self.config["DECISION_RESOLUTION"]
                    + 0.5
                    * self.config["DEFAULT_ACC"]
                    * self.config["DECISION_RESOLUTION"]
                    * self.config["DECISION_RESOLUTION"]
                )
            elif action == "DC":
                veh_next_state.vel -= (
                    self.config["DEFAULT_ACC"] *
                    self.config["DECISION_RESOLUTION"]
                )
                veh_next_state.vel = max(veh_next_state.vel, 0)
                veh_next_state.s += max(
                    0,
                    (
                        veh_next_state.vel * self.config["DECISION_RESOLUTION"]
                        - 0.5
                        * self.config["DEFAULT_ACC"]
                        * self.config["DECISION_RESOLUTION"]
                        * self.config["DECISION_RESOLUTION"]
                    ),
                )
            elif action == "LCL":
                veh_next_state.d += (
                    self.config["LATERAL_SPEED"] *
                    self.config["DECISION_RESOLUTION"]
                )
                veh_next_state.s += (
                    veh_next_state.vel * self.config["DECISION_RESOLUTION"]
                )
            elif action == "LCR":
                veh_next_state.d -= (
                    self.config["LATERAL_SPEED"] *
                    self.config["DECISION_RESOLUTION"]
                )
                veh_next_state.s += (
                    veh_next_state.vel * self.config["DECISION_RESOLUTION"]
                )
            else:
                print("[EEROR] Unknown action: ", action)
                exit(1)

            current_lane = self.road_graph.get_lane_by_id(veh.lane_id)
            # 车道更新
            if action == "LCL" or action == "LCR":
                if veh_next_state.d > current_lane.width / 2:
                    next_lane = self.road_graph.get_lane_by_id(
                        current_lane.left_lane())
                    if next_lane is None:  # 车道变更失败
                        veh_next_state.d = current_lane.width / 2
                    else:
                        veh_next_step.lane_id = next_lane.id
                        veh_next_state.d -= current_lane.width / 2 + next_lane.width / 2
                elif veh_next_state.d < -current_lane.width / 2:
                    next_lane = self.road_graph.get_lane_by_id(
                        current_lane.right_lane()
                    )
                    if next_lane is None:
                        veh_next_state.d = -current_lane.width / 2
                    else:
                        veh_next_step.lane_id = next_lane.id
                        veh_next_state.d += current_lane.width / 2 + next_lane.width / 2

            current_lane = self.road_graph.get_lane_by_id(
                veh_next_step.lane_id)
            # 处理超出 available lanes的情况
            while veh_next_state.s > current_lane.spline_length:
                next_lane = self.road_graph.get_available_next_lane(
                    current_lane.id, veh_next_step.available_lanes
                )
                if next_lane is None:
                    veh_next_step.lane_id = None
                    break
                veh_next_state.s -= current_lane.spline_length
                veh_next_step.lane_id = next_lane.id
                current_lane = self.road_graph.get_lane_by_id(
                    veh_next_step.lane_id)
            if veh_next_step.lane_id == None:
                # at the end of available_lanes range, not decision for this vehicle any more
                continue

            # calculate x and y coordinate
            actual_lane = self.road_graph.get_lane_by_id(veh_next_step.lane_id)
            (
                veh_next_state.x,
                veh_next_state.y,
            ) = actual_lane.course_spline.frenet_to_cartesian1D(
                veh_next_state.s, veh_next_state.d
            )

            veh_next_state.laneID = veh_next_step.lane_id
            actions_next_step[veh_next_step.id].append(action)
            vehs_next_step.append(veh_next_step)

        return FlowState(
            self.states_list + [vehs_next_step],
            self.road_graph,
            actions_next_step,
            self.complete_decisions,
            self.prediction,
            next_time,
            self.config,
        )

    def terminal(self):
        if (
            self.time >= self.config["MAX_DECISION_TIME"]
            or len(self.states_list[-1]) == 0
        ):
            # exceed max decision time or all vehicles have finished decision in available lanes
            return True
        if self.num_moves == 0:
            # agents collison
            return True
        # 是否判断提前结束？？
        return False

    def reward(self):
        # reward have to limit to [0,1]
        if self.num_moves == 0:
            return 0.0
        # find last exist frame index for each vehicle
        last_exist_frame_idx = {}
        for frame_idx in range(len(self.states_list)):
            for veh in self.states_list[frame_idx]:
                if veh.id not in last_exist_frame_idx:
                    last_exist_frame_idx[veh.id] = [veh]
                else:
                    last_exist_frame_idx[veh.id].append(veh)

        rewards_for_each_veh = []
        max_decision_num = len(self.states_list)
        for veh_id, veh_list in last_exist_frame_idx.items():
            reward_self = 0.0
            reward_others = 0.0
            # reward about the procedure
            for idx in range(len(veh_list)):
                veh = veh_list[idx]
                # keep in lane center reward
                if abs(veh.current_state.d) < 0.5:
                    reward_self += 0.2 / max_decision_num
                # speed reward
                if idx > 0 and (
                    self.actions[veh_id][idx - 1] == "AC"
                    or self.actions[veh_id][idx - 1] == "KS"
                ):
                    reward_self += 0.2 / max_decision_num
                # action coninuity reward
                if (
                    idx > 1
                    and self.actions[veh_id][idx - 2] == self.actions[veh_id][idx - 1]
                ):
                    reward_self += 0.2 / max_decision_num
                # in available lanes reward
                if veh.lane_id in veh.available_lanes:
                    reward_self += 0.2 / max_decision_num

            # reward about the result
            if veh_list[-1].lane_id in veh_list[-1].available_lanes:
                if abs(veh_list[-1].current_state.d) < 0.5:
                    reward_self += 0.8
                else:
                    reward_self += 0.2

            # todo: calculate reward/penalty for other vehicles
            rewards_for_each_veh.append(min(1.0, reward_self + reward_others))

        total_reward = sum(rewards_for_each_veh) / len(rewards_for_each_veh)
        return max(0.0, min(1.0, total_reward))

    def _check_collision(
        self, veh1: Vehicle, state1: State, veh2: Vehicle, state2: State
    ) -> bool:
        if veh1 == veh2:
            print("Decision vehicle has already decision?!")
            exit(1)

        dist = math.hypot(state1.x - state2.x, state1.y - state2.y)
        dist_thershold = math.hypot(
            veh1.length + veh2.length, veh1.width + veh2.width)
        if dist > dist_thershold:
            return False

        is_collide, _ = check_collsion_new(
            np.array([state1.x, state1.y]),
            veh1.length * 2,
            veh1.width * 1.5,
            state1.yaw,
            np.array([state2.x, state2.y]),
            veh2.length,
            veh2.width,
            state2.yaw,
        )

        return is_collide
