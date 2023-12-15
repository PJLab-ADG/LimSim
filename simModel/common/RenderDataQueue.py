import multiprocessing
from typing import Tuple, Dict, List, Union
from utils.roadgraph import RoadGraph
from rich import print

class ERD:
    # edge render data
    def __init__(
        self, id: str, num_lanes: int,
    ) -> None:
        self.id = id
        self.num_lanes = num_lanes


class LRD:
    # lane render data
    def __init__(
        self, id: str, 
        left_bound: List[float],
        right_bound: List[float]
    ) -> None:
        self.id = id
        self.left_bound = left_bound
        self.right_bound = right_bound

    @property
    def edge(self):
        return self.id.rsplit('_', 1)[0]


class JLRD:
    # junction lane render data
    def __init__(
        self, id: str,
        center_line: List[float],
        currTlState: str
    ) -> None:
        self.id = id
        self.center_line = center_line
        self.currTlState = currTlState


class RGRD:
    # roadgraph render data
    def __init__(self) -> None:
        self.lanes: Dict[str, LRD] = {}
        self.junction_lanes: Dict[str, JLRD] = {}
        self.edges: Dict[str, ERD] = {}

    def get_lane_by_id(self, lid: str) -> Union[LRD, JLRD]:
        if lid[0] == ':':
            return self.junction_lanes[lid]
        else:
            return self.lanes[lid]

    def get_edge_by_id(self, eid: str) -> ERD:
        return self.edges[eid]


class VRD:
    # vehicle render data
    def __init__(
        self, id: str, x: float, y: float, 
        yaw: float, deArea: float,
        length: float, width: float,
        trajectoryXQ: List[float],
        trajectoryYQ: List[float]
    ) -> None:
        self.id = id
        self.x = x
        self.y = y 
        self.yaw = yaw
        self.deArea = deArea
        self.length = length
        self.width = width
        self.trajectoryXQ = trajectoryXQ
        self.trajectoryYQ = trajectoryYQ


class RenderDataQueue:
    def __init__(self, max_size: int) -> None:
        self.queue = multiprocessing.Manager().list()
        self.max_size = max_size

    def put(self, item: Tuple[RGRD, Dict[str, List[VRD]]]):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(item)

    def get(self) -> Tuple[RGRD, Dict[str, List[VRD]]]:
        return self.queue[-1] if self.queue else None

