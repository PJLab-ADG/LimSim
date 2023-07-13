"""
Author: Licheng Wen
Date: 2023-04-12 16:51:21
Description: 

Copyright (c) 2023 by PJLab, All Rights Reserved. 
"""

from itertools import combinations
import math
from common.observation import Observation
from decision_maker.abstract_decision_maker import (
    AbstractEgoDecisionMaker,
    AbstractMultiDecisionMaker,
    EgoDecision,
    MultiDecision,
    SingleStepDecision,
)
from predictor.abstract_predictor import Prediction

from utils.roadgraph import RoadGraph, JunctionLane, NormalLane
from utils.trajectory import State
from common.vehicle import Behaviour, Vehicle, VehicleType
from mcts import mcts
from mcts.flow_state import FlowState

import logger

logging = logger.get_logger(__name__)


class EgoDecisionMaker(AbstractEgoDecisionMaker):
    def make_decision(
        self,
        observation: Observation,
        road_graph: RoadGraph,
        prediction: Prediction = None,
    ) -> EgoDecision:
        # Implement customized decision maker here
        pass


class MultiDecisionMaker(AbstractMultiDecisionMaker):
    def _judge_interactions(
        self, observation: Observation, roadgraph: RoadGraph
    ) -> dict:
        # interaction which interaction(id_i,id_j) = True
        # means if there's a interaction between vehicle i and vehicle j
        interaction = {
            (veh_i.id, veh_j.id): False for veh_i in observation.vehicles
            for veh_j in observation.vehicles
            if veh_i.id != veh_j.id and veh_i.vtype != VehicleType.OUT_OF_AOI
            and veh_j.vtype != VehicleType.OUT_OF_AOI
        }
        # vehicle pairs with interaction
        # todo: add OVERTAKE behaviour support

        for veh_i, veh_j in combinations(observation.vehicles, 2):
            if VehicleType.OUT_OF_AOI in (veh_i.vtype, veh_j.vtype):
                continue
            if veh_i.id != veh_j.id:
                lane_i = roadgraph.get_lane_by_id(veh_i.lane_id)
                lane_j = roadgraph.get_lane_by_id(veh_j.lane_id)
                if isinstance(lane_i, JunctionLane) or isinstance(
                    lane_j, JunctionLane
                ):  # in junction
                    dist = math.sqrt(
                        (veh_i.current_state.x - veh_j.current_state.x) ** 2
                        + (veh_i.current_state.y - veh_j.current_state.y) ** 2
                    )
                    if dist < 20:
                        interaction[(veh_i.id, veh_j.id)] = interaction[
                            (veh_j.id, veh_i.id)
                        ] = True
                        continue
                if veh_i.lane_id == veh_j.lane_id:  # in same lane
                    # make sure veh_j is in front of veh_i
                    if veh_i.current_state.s > veh_j.current_state.s:
                        veh_i, veh_j = veh_j, veh_i
                    s_dist = veh_j.current_state.s - veh_i.current_state.s
                    inter_dist = (3 + 0.5) * veh_i.current_state.s_d + veh_i.length
                    if s_dist < inter_dist:
                        interaction[(veh_i.id, veh_j.id)] = interaction[
                            (veh_j.id, veh_i.id)
                        ] = True
                        continue
                # veh_j is in next lane of veh_i
                if (
                    isinstance(lane_i, NormalLane)
                    and lane_j.id
                    in [lane_i.next_lanes[key][0] for key in lane_i.next_lanes.keys()]
                ) or (
                    isinstance(lane_i, JunctionLane)
                    and lane_j.id == lane_i.next_lane_id
                ):
                    s_dist = (
                        lane_i.spline_length
                        - veh_i.current_state.s
                        + veh_j.current_state.s
                    )
                    inter_dist = (3 + 0.5) * veh_i.current_state.s_d + veh_i.length
                    if s_dist < inter_dist:
                        interaction[(veh_i.id, veh_j.id)] = interaction[
                            (veh_j.id, veh_i.id)
                        ] = True
                        continue
                # veh_i is in next lane of veh_j
                if (
                    isinstance(lane_j, NormalLane)
                    and lane_i.id
                    in [lane_j.next_lanes[key][0] for key in lane_j.next_lanes.keys()]
                ) or (
                    isinstance(lane_j, JunctionLane)
                    and lane_i.id == lane_j.next_lane_id
                ):
                    s_dist = (
                        lane_j.spline_length
                        - veh_j.current_state.s
                        + veh_i.current_state.s
                    )
                    inter_dist = (3 + 0.5) * veh_j.current_state.s_d + veh_j.length
                    if s_dist < inter_dist:
                        interaction[(veh_i.id, veh_j.id)] = interaction[
                            (veh_j.id, veh_i.id)
                        ] = True
                        continue
                # veh_i and veh_j are in adjacent lanes and one of them is changing lane
                if (
                    isinstance(lane_i, NormalLane)
                    and isinstance(lane_j, NormalLane)
                    and lane_i.affiliated_edge == lane_j.affiliated_edge
                ):
                    if (
                        abs(veh_i.current_state.s - veh_j.current_state.s)
                        < veh_i.length + veh_j.length
                    ):
                        if (
                            veh_i.behaviour == Behaviour.LCL
                            and lane_j.id == lane_i.left_lane()
                        ) or (
                            veh_i.behaviour == Behaviour.LCR
                            and lane_j.id == lane_i.right_lane()
                        ):
                            interaction[(veh_i.id, veh_j.id)] = interaction[
                                (veh_j.id, veh_i.id)
                            ] = True
                            continue
                        if (
                            veh_j.behaviour == Behaviour.LCL
                            and lane_i.id == lane_j.left_lane()
                        ) or (
                            veh_j.behaviour == Behaviour.LCR
                            and lane_i.id == lane_j.right_lane()
                        ):
                            interaction[(veh_i.id, veh_j.id)] = interaction[
                                (veh_j.id, veh_i.id)
                            ] = True
                            continue
        return interaction

    def _grouping(self, observation: Observation, interaction: dict) -> dict:
        max_group_size = 3
        # group_info = {groupid: [veh_a, veh_b, ...]}
        group_info = {}
        group_idx = {}
        group_id = 0

        # calculate group_info
        #  group vehicles into several groups where vehicles in a group have at least one potential interaction with a vehicle in the same group.
        for veh_i in observation.vehicles:
            if veh_i.vtype == VehicleType.OUT_OF_AOI:
                continue
            if (
                veh_i.id in group_idx
                and len(group_info[group_idx[veh_i.id]]) >= max_group_size
            ):
                continue
            if veh_i.id not in group_idx:
                group_id += 1
                group_info[group_id] = [veh_i]
                group_idx[veh_i.id] = group_id
            for veh_j in observation.vehicles:
                if veh_j.vtype == VehicleType.OUT_OF_AOI:
                    continue
                if veh_j.id not in group_idx and interaction[(veh_i.id, veh_j.id)]:
                    group_info[group_idx[veh_i.id]].append(veh_j)
                    group_idx[veh_j.id] = group_idx[veh_i.id]
                if len(group_info[group_id]) >= max_group_size:
                    break
        return group_info

    def make_decision(
        self,
        T: float,
        observation: Observation,
        road_graph: RoadGraph,
        prediction: Prediction = None,
        config: dict = None,
    ) -> MultiDecision:
        """
        Implement decision maker here
        Step 0: Determine if there are any vehicles that require decision-making.
        Step 1: Extract vehicles with potential for interaction in the judge_interactions step.
        Step 2: Group vehicles with interaction potential, setting a maximum number of vehicles per 
                decision group and establishing priority relationships between decision groups.
        Step 3: Perform MCTS decision-making on each decision group in sequence, searching for 
                the best decision sequence within each group.
        """
        # Step 0: Determine if there are any vehicles that require decision-making.
        if not observation.vehicles:
            print("[ERROR] DecisionMaker: No vehicles to make decision.")
            return MultiDecision()

        # Step 1: Extract vehicles with potential for interaction in the judge_interactions step.
        interaction = self._judge_interactions(observation, road_graph)

        # Step 2: Group vehicles with interaction potential, setting a maximum number of
        # vehicles per decision group and establishing priority relationships between decision groups.
        group_info = self._grouping(observation, interaction)

        # Step 3: Perform MCTS decision-making on each decision group in sequence,
        # searching for the best decision sequence within each group.
        complete_decisions = MultiDecision()
        for group_idx, vehs_in_group in group_info.items():
            # decide for group with group_idx
            mcts_init_state = [veh for veh in vehs_in_group]
            actions = {veh.id: [] for veh in vehs_in_group}

            current_node = mcts.Node(
                FlowState(
                    [mcts_init_state],
                    road_graph,
                    actions,
                    complete_decisions,
                    prediction,
                    time=0,
                    config=config,
                )
            )

            for t in range(
                int(config["MAX_DECISION_TIME"] / config["DECISION_RESOLUTION"])
            ):
                current_node = mcts.uct_search(200 / (t / 2 + 1), current_node)
                if current_node is None:
                    # decision failed
                    break
                # print("Best Child: ", current_node.visits / (200 / (t / 2 + 1)) * 100, "%")
                temp_best = current_node
                while temp_best.children:
                    temp_best = mcts.best_child(temp_best, 0)
                if temp_best.state.terminal() and temp_best.state.reward() > 0.8:
                    # Best child is at end
                    break

            while (
                current_node is not None
                and current_node.children
                and not current_node.state.terminal()
            ):
                current_node = mcts.best_child(current_node, 0)
            if current_node is None or current_node.state.reward() < 0.5:
                logging.warning(
                    "Decision failed for group %d, ignoring these vehicles", group_idx,
                )
                continue
            logging.debug("Final reward: %f", current_node.state.reward())
            decisions = {}
            for i in range(1, len(current_node.state.states_list)):
                for veh in current_node.state.states_list[i]:
                    decision_at_t = SingleStepDecision()
                    decision_at_t.expected_state = veh.current_state
                    decision_at_t.expected_time = T + i * config["DECISION_RESOLUTION"]
                    decision_at_t.action = current_node.state.actions[veh.id][i - 1]
                    if veh.id not in decisions:
                        decisions[veh.id] = []
                    decisions[veh.id].append(decision_at_t)
            for veh in vehs_in_group:
                if (
                    veh.id not in decisions
                    or decisions[veh.id][-1].expected_state.laneID
                    not in veh.available_lanes
                ):
                    logging.warning(
                        "Decision failed for vehicle %s, ignoring decision for it.",
                        veh.id,
                    )
                    continue
                complete_decisions.results[veh] = decisions[veh.id]

        # print detailed decision results, uncomment if necessary
        # for veh, decisions in complete_decisions.results.items():
        #     print(veh.id, end="\t")
        #     for decision in decisions:
        #         print(decision.action, end="\t")
        #     print()
        return complete_decisions
