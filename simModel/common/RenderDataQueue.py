import multiprocessing
from typing import Tuple, Dict, List, Set
from utils.roadgraph import RoadGraph
from rich import print


class VRD:
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


class RRD:
    def __init__(self, edges: Set[str], junctions: Set[str]) -> None:
        self.edges = edges
        self.junctions = junctions


class RenderData:
    def __init__(
        self, VehiclesRenderData: Dict[str, List[VRD]], 
        roadgraph: RoadGraph
    ) -> None:
        self.VehiclesRenderData = VehiclesRenderData
        self.roadgraph = roadgraph

class RenderDataQueue:
    def __init__(self, max_size: int) -> None:
        self.queue = multiprocessing.Manager().list()
        self.max_size = max_size

    def put(self, item: Tuple[RRD, Dict[str, List[VRD]]]):
        print('receive VRDDict: ', item[1]['carInAoI'])
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(item)

    def get(self) -> Tuple[RRD, Dict[str, List[VRD]]]:
        return self.queue[-1] if self.queue else None
