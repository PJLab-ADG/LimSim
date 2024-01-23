import sqlite3
import time

from simModel.Model import Model
from utils.trajectory import Rectangle, RecCollide

class CollisionException(Exception):
    def __init__(self, ErrorInfo: str) -> None:
        super().__init__(self)
        self.errorinfo = ErrorInfo
    
    def __str__(self) -> str:
        return self.errorinfo

class CollisionChecker:
    def __init__(self):
        pass
    
    # 应该在model里面，实施检测碰撞，还有没有路径，或者两个地方都搞一个也行
    def CollisionCheck(self, model: Model) -> bool:
        # vehicle trajectory collision need to be checked in every frame
        for key, value in model.ms.vehINAoI.items():
            if value.id == model.ms.ego.id:
                continue
            recA = Rectangle([model.ms.ego.x, model.ms.ego.y],
                                model.ms.ego.length, model.ms.ego.width, model.ms.ego.yaw)
            recB = Rectangle([value.x, value.y],
                                value.length, value.width, value.yaw)
            rc = RecCollide(recA, recB)
            # if the car collide, stop the simulation
            if rc.isCollide():
                raise CollisionException("you have a collision with vehicle {}".format(key))
        return False
    
class LaneChangeException(Exception):
    def __init__(self) -> None:
        super().__init__(self)
        self.errorinfo = "you need to change lane, but your lane change is not successful"
    
    def __str__(self) -> str:
        return self.errorinfo  
    
def record_result(model: Model, start_time: float, result: bool, reason: str = "", error: Exception = None) -> None:
    conn = sqlite3.connect(model.dataBase)
    cur = conn.cursor()
    # add result data
    cur.execute(
        """INSERT INTO resultINFO (
            egoID, result, total_score, complete_percentage, drive_score, use_time, fail_reason
            ) VALUES (?,?,?,?,?,?,?);""",
        (
            model.ms.ego.id, result, 0, 0, 0, time.time() - start_time, reason
        )
    )
    conn.commit()
    conn.close()
    return 