from dataclasses import dataclass, field
from collections import deque


@dataclass
class State:
    roadId: str = None
    laneId: str = None
    direction: int = 0
    x: float = 0.0  # Cartesian
    y: float = 0.0  # Cartesian
    hdg: float = 0.0  # Cartesian
    vel: float = 0.0  # Cartesian & Frenet
    acc: float = 0.0  # Cartesian & Frenet
    scf: float = 0.0  # Frenet - s-axis coordinate
    tcf: float = 0.0  # Frenet - t-axis coordinate
    yaw: float = 0.0  # Frenet - yaw


class Vehicle(object):

    def __init__(
        self,
        id: str,
        arrivalTime: float = 0.0,
        fromNone: int = 0,
        tomNone: int = 0,
        direction: int = 0,
        initVel: float = 0.0,
        length: float = 0.0,
        width: float = 0.0,
    ):
        # inital attributes
        self.id: str = id
        self.arrivalTime: float = arrivalTime
        self.roadId: str = fromNone
        self.toNode: str = tomNone
        self.laneId: str = None
        self.direction: int = direction
        self.vel: float = initVel
        self.length: float = length
        self.width: float = width
        self.x: float = 0.0  # Cartesian
        self.y: float = 0.0  # Cartesian
        self.hdg: float = 0.0  # Cartesian
        self.vel: float = 0.0  # Cartesian & Frenet
        self.acc: float = 0.0  # Cartesian & Frenet
        self.scf: float = 0.0  # Frenet - s-axis coordinate
        self.tcf: float = None  # Frenet - t-axis coordinate
        self.yaw: float = 0.0  # Frenet - yaw

        # motion status
        self.routeIdxQ = deque(maxlen=100)
        self.route: list[str] = None
        self.planTra = Trajectory()
        self.planner = None
        self.deArea = 50
        self.drivingScore = None
        self.EXPECT_VEL = 20
        self.PLAN_LEN = 4  # 3 seconds
        self.PLAN_GAP = 2  # 1 second

        maxlen = 20
        self.xQ = deque(maxlen=maxlen)
        self.yQ = deque(maxlen=maxlen)
        self.hdgQ = deque(maxlen=maxlen)
        self.velQ = deque(maxlen=maxlen)
        self.accQ = deque(maxlen=maxlen)
        self.yawQ = deque(maxlen=maxlen)
        self.roadIdQ = deque(maxlen=maxlen)

        self.planXQ: list[float] = None
        self.planYQ: list[float] = None
        self.planHdgQ: list[float] = None
        self.planVelQ: list[float] = None
        self.planAccQ: list[float] = None
        self.planYawQ: list[float] = None
        self.planRoadIdQ: list[str] = None

    def setState(self, state: State):
        (
            self.roadId,
            self.laneId,
            self.direction,
            self.x,
            self.y,
            self.hdg,
            self.vel,
            self.acc,
            self.scf,
            self.tcf,
            self.yaw,
        ) = (
            state.roadId,
            state.laneId,
            state.direction,
            state.x,
            state.y,
            state.hdg,
            state.vel,
            state.acc,
            state.scf,
            state.tcf,
            state.yaw,
        )

    def move(self, new_status: list):
        # check whether vehicle is in road or junctions
        self.setState(new_status)

    def stop(self):
        self.vel = 0
        self.acc = 0

    def getState(self, status, netInfo):
        scf, tcf, yaw, vel, acc = status
        ret = netInfo.frenet2Cartesian(
            self, self.roadId, self.laneId, self.direction, scf, tcf, yaw
        )
        if ret != None:
            roadId, laneId, direction, x, y, hdg, scf, tcf = ret
        else:
            return
        return State(roadId, laneId, direction, x, y, hdg, vel, acc, scf, tcf, yaw)


@dataclass
class Trajectory:
    """Trajectory class."""

    states: list[State] = field(default_factory=list)

    def __len__(self):
        return len(self.states)

    @property
    def xQ(self) -> deque[float]:
        return deque([state.x for state in self.states])

    @property
    def yQ(self) -> deque[float]:
        return deque([state.y for state in self.states])

    @property
    def hdgQ(self) -> deque[float]:
        return deque([state.hdg for state in self.states])

    @property
    def yawQ(self) -> deque[float]:
        return deque([state.yaw for state in self.states])

    @property
    def velQ(self) -> deque[float]:
        return deque([state.vel for state in self.states])

    @property
    def accQ(self) -> deque[float]:
        return deque([state.acc for state in self.states])

    @property
    def roadIdQ(self) -> deque[str]:
        return deque([state.roadId for state in self.states])

    @property
    def lanePosQueue(self) -> deque[float]:
        return deque([state.s for state in self.states])

    @property
    def routeIdxQueue(self) -> deque[float]:
        return deque([state.routeIdx for state in self.states])


class DummyVehicle(Vehicle):
    def __init__(self, id, state) -> None:
        super().__init__(id)
        self.id: str = id
        self.x, self.y, self.scf, self.tcf, self.roadId = state
        self.velQ = []
        self.accQ = []
        self.planVelQ = []
        self.planAccQ = []
        self.drivingScore = [[], []]
