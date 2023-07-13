'''
Author: Licheng Wen
Date: 2022-11-08 17:36:16
Description: 
Copyright (c) 2022 by PJLab, All Rights Reserved. 
'''
from __future__ import annotations
from cubic_spline import Spline2D
import numpy as np
from typing import Dict, Set
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC


import logger

logging = logger.get_logger(__name__)


OVERLAP_DISTANCE = 0.1  # junction overlap distance


@dataclass
class Junction:
    id: str = None
    incoming_edges: set[str] = field(default_factory=set)
    outgoing_edges: set[str] = field(default_factory=set)
    JunctionLanes: set[str] = field(default_factory=set)
    affGridIDs: set[tuple[int]] = field(default_factory=set)
    shape: list[tuple[float]] = None


@dataclass
class Edge:
    id: str = None
    lane_num: int = 0
    lane_width: float = 0
    lanes: Set[str] = field(default_factory=set)
    from_junction: str = None
    to_junction: str = None
    next_edge_info: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )  # next edge and the corresponding **self** normal lane
    obstacles: dict = field(default_factory=dict)
    affGridIDs: set[tuple[int]] = field(default_factory=set)
    _waypoints_x: list[float] = None
    _waypoints_y: list[float] = None

    @property
    def edge_width(self):
        ew = 0
        for lane in self.lanes:
            ew += lane.width
        return ew

    def __hash__(self):
        return hash(self.id)

    def __repr__(self) -> str:
        return f"Edge(id={self.id})"
        # return f"Edge(id={self.id}, lane_num={len(self.lanes)}, from_junction={self.from_junction}, to_junction={self.to_junction})\n"


@dataclass
class AbstractLane(ABC):
    """
    Abstract lane class.
    """
    id: str
    width: float = 0
    speed_limit: float = 13.89
    sumo_length: float = 0
    course_spline: Spline2D = None

    @property
    def spline_length(self):
        return self.course_spline.s[-1]

    def getPlotElem(self):
        s = np.linspace(0, self.course_spline.s[-1], num=50)
        self.center_line = [
            self.course_spline.calc_position(si) for si in s
        ]
        self.left_bound = [
            self.course_spline.frenet_to_cartesian1D(si, self.width / 2) for si in s
        ]
        self.right_bound = [
            self.course_spline.frenet_to_cartesian1D(si, -self.width / 2) for si in s
        ]


@dataclass
class NormalLane(AbstractLane):
    """
    Normal lane from edge 
    """
    affiliated_edge: Edge = None
    next_lanes: Dict[str, tuple[str, str]] = field(
        default_factory=dict
    )  # next_lanes[to_lane_id: normal lane] = (via_lane_id, direction)

    def left_lane(self) -> str:
        lane_index = int(self.id.split("_")[-1])
        left_lane_id = f"{self.affiliated_edge.id}_{lane_index + 1}"
        for lane in self.affiliated_edge.lanes:
            if lane == left_lane_id:
                return left_lane_id
        # logging.error(f"cannot find left lane of {self.id}")
        return None

    def right_lane(self) -> str:
        lane_index = int(self.id.split("_")[-1])
        right_lane_id = f"{self.affiliated_edge.id}_{lane_index - 1}"
        for lane in self.affiliated_edge.lanes:
            if lane == right_lane_id:
                return right_lane_id
        # logging.error(f"cannot find right lane of {self.id}")
        return None

    def __hash__(self):
        return hash(self.id)

    def __repr__(self) -> str:
        # return f"NormalLane(id={self.id}, width = {self.width})"
        return f"NormalLane(id={self.id})"


@dataclass(unsafe_hash=True)
class JunctionLane(AbstractLane):
    """
    Junction lane in intersection 
    """
    last_lane_id: str = None
    next_lane_id: str = None  # next lane's id
    affJunc: str = None   # affiliated junction ID
    tlLogic: str = None
    tlsIndex: int = 0
    currTlState: str = None   # current traffic light phase state: r, g, y etc.
    # remain time (second)  switch to next traffic light phase.
    switchTime: float = 0.0
    nexttTlState: str = None   # next traffic light phase state: r, g, y etc.

    def __repr__(self) -> str:
        return f"JunctionLane(id={self.id} tlState={self.currTlState} switchTime={self.switchTime})"
        # return f"JunctionLane(id={self.id}, width = {self.width}, next_lane={self.next_lane})"


@dataclass
class TlLogic:
    id: str = None
    tlType: str = None   # static or actuated
    preDefPhases: list[str] = None

    def currPhase(self, currPhaseIndex: int) -> str:
        return self.preDefPhases[currPhaseIndex]

    def nextPhase(self, currPhaseIndex: int) -> str:
        if currPhaseIndex < len(self.preDefPhases)-1:
            return self.preDefPhases[currPhaseIndex+1]
        else:
            return self.preDefPhases[0]


@dataclass
class RoadGraph:
    """
    Road graph of the map
    """

    edges: Dict[str, Edge] = field(default_factory=dict)
    lanes: Dict[str, AbstractLane] = field(default_factory=dict)
    junction_lanes: Dict[str, JunctionLane] = field(default_factory=dict)

    def get_lane_by_id(self, lane_id: str) -> AbstractLane:
        if lane_id in self.lanes:
            return self.lanes[lane_id]
        elif lane_id in self.junction_lanes:
            return self.junction_lanes[lane_id]
        else:
            logging.debug(f"cannot find lane {lane_id}")
            return None

    def get_next_lane(self, lane_id: str) -> AbstractLane:
        lane = self.get_lane_by_id(lane_id)
        if isinstance(lane, NormalLane):
            next_lanes = list(lane.next_lanes.values())
            if len(next_lanes) > 0:
                # first_next_lane = list(lane.next_lanes.values())[0][0]
                return self.get_lane_by_id(next_lanes[0][0])
            else:
                return None
        elif isinstance(lane, JunctionLane):
            return self.get_lane_by_id(lane.next_lane_id)
        return None

    def get_available_next_lane(self, lane_id: str, available_lanes: list[str]) -> AbstractLane:
        lane = self.get_lane_by_id(lane_id)
        if isinstance(lane, NormalLane):
            for next_lane_i in lane.next_lanes.values():
                if next_lane_i[0] in available_lanes:
                    return self.get_lane_by_id(next_lane_i[0])
        elif isinstance(lane, JunctionLane):
            if lane.next_lane_id in available_lanes:
                return self.get_lane_by_id(lane.next_lane_id)
        return None

    def __str__(self):
        return 'edges: {}, \nlanes: {}, \njunctions lanes: {}'.format(
            self.edges.keys(), self.lanes.keys(),
            self.junction_lanes.keys()
        )


if __name__ == "__main__":
    pass
