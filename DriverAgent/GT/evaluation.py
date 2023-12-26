from trafficManager.common.vehicle import Behaviour, Vehicle, VehicleType
from utils.trajectory import Trajectory, State, Rectangle, RecCollide
from typing import Dict, List, Union

class CollisionException(Exception):
    def __init__(self, ErrorInfo: str) -> None:
        super().__init__(self)
        self.errorinfo = ErrorInfo
    
    def __str__(self) -> str:
        return self.errorinfo
    
class RedLightException(Exception):
    def __init__(self, ErrorInfo: str) -> None:
        super().__init__(self)
        self.errorinfo = ErrorInfo
    
    def __str__(self) -> str:
        return self.errorinfo

class GT_Evaluation:
    def __init__(self, dt: float = 0.1) -> None:
        self.dt: float = dt
        self.decision_score: float = 0.0
        self.eval_score: float = 0.0

    # vehicle trajectory collision check
    def VTCollisionCheck(self, trajectory: Dict[str, Trajectory], vehicles: Dict[int, Vehicle]) -> bool:
        ego_id = None
        for id, vehicle in vehicles.items():
            if vehicle.vtype == VehicleType.EGO:
                ego_id = id
                break
        if ego_id is None:
            raise ValueError("No ego vehicle found in the vehicle list")
        
        tjA = trajectory[ego_id]
        for veh_id in trajectory.keys():
            if veh_id == ego_id:
                continue
            tjB = trajectory[veh_id]
            duration = min(len(tjA.states), len(tjB.states))
            for i in range(0, duration, 3):
                stateA = tjA.states[i]
                stateB = tjB.states[i]
                recA = Rectangle([stateA.x, stateA.y],
                                 vehicles[ego_id].length, vehicles[ego_id].width, stateA.yaw)
                recB = Rectangle([stateB.x, stateB.y],
                                 vehicles[veh_id].length, vehicles[veh_id].width, stateB.yaw)
                rc = RecCollide(recA, recB)
                if rc.isCollide():
                    raise CollisionException("vehicle ego and vehicle {} will have a collision at {} seconds".format(veh_id, stateA.t))
        return False
    
    def PassRedLight():
        pass

    def TrajectoryCost(self):
        # 1. jerk

        # 2. acc
        
        # 3. lower speed than other car

        # 4. exceed speed limit

        pass