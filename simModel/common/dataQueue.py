import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class LaneInfo:
    id: str
    type: str
    leftBorders: List[List[float]]
    rightBorders: List[List[float]]
    centerLine: List[List[float]]


@dataclass
class JunctionInfo:
    boundary: List[List[float]]
    junctionLanes: List[str]


@dataclass
class NetInfo:
    boundries: List[float] = field(default_factory=list)
    lanes: Dict[str, LaneInfo] = field(default_factory=dict)
    junctions: Dict[str, JunctionInfo] = field(default_factory=dict)


@dataclass
class VehicleInfo:
    id: int
    length: float
    width: float
    x: float
    y: float
    hdg: float
    planXQ: list[float]
    planYQ: list[float]


@dataclass
class EgoVehicleInfo(VehicleInfo):
    deArea: float
    roadId: str
    drivingScore: float
    velQ: list[float]
    accQ: list[float]
    planVelQ: list[float]
    planAccQ: list[float]


@dataclass
class TlsInfo:
    currentState: str


@dataclass
class DynamicInfo:
    egoInfo: EgoVehicleInfo
    vehInAoI: Dict[int, VehicleInfo] = field(default_factory=dict)
    outOfAoI: Dict[int, VehicleInfo] = field(default_factory=dict)
    tls: Dict[Tuple[str, str], TlsInfo] = field(default_factory=dict)


class RenderQueue:
    def __init__(self, max_size: int) -> None:
        self.queue = multiprocessing.Manager().list()
        self.max_size = max_size

    def put(self, item: tuple[DynamicInfo, float]):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(item)

    def get(self):
        return self.queue[-1] if self.queue else None


class FocusPos:
    def __init__(self, max_size: int = 1) -> None:
        self.queue = multiprocessing.Manager().list()
        self.max_size = max_size

    def setPos(self, pos: List[float]):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(pos)

    def getPos(self):
        return self.queue[-1] if self.queue else None


class SimPause:
    def __init__(self) -> None:
        self.pause = multiprocessing.Manager().Value("i", 0)

    def pauseSim(self):
        self.pause.value = 1

    def resumeSim(self):
        self.pause.value = 0
