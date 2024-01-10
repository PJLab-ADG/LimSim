from trafficManager.common.vehicle import Behaviour, Vehicle, VehicleType
from utils.trajectory import Trajectory, State, Rectangle, RecCollide
from typing import Dict, List
from simModel.egoTracking.model import Model
import numpy as np

# TODO: 1. 把这个写成离线的，然后不再对应decision的评价，那么如何提取好的决策结果保存到memory呢？也可以手动添加我觉得
# 2. 修改dirveagent代码结构，把system mesage等信息外挂，为了规范LLM的输出，可以写example，同时写好output，并入到这个仓库里
# 3. 离线跑一轮，看看效果和评价准确性
# 4. 增加memory，确定形式和检索方法
# 5. reflection可以把连续的几帧决策过程和结果都输入给它（好像reflection只给失败的决策也可以），同时补偿一些额外知识给它，要确定reflection的输出结果和格式是什么，然后怎么运用
# 6. 修改换道的规划，保持速度不变

class CollisionException(Exception):
    def __init__(self, ErrorInfo: str) -> None:
        super().__init__(self)
        self.errorinfo = ErrorInfo
    
    def __str__(self) -> str:
        return self.errorinfo

class GT_Evaluation:
    def __init__(self) -> None:
        self.complete_percentage: float = 0.0

        self.decision_score: List[float] = []
        self.current_score = 100.0 # based on the past 30 frames
        self.final_score: float = 0.0 # based on the whole routes

        self.red_junctionlane_record = []
        self.current_reasoning = []

        self
    
    # vehicle trajectory collision check
    def CollisionCheck(self, model: Model) -> bool:
        # 1. get history trajectory
        ego_history_x = list(model.ms.ego.xQ)[::-1]
        ego_history_y = list(model.ms.ego.yQ)[::-1]
        ego_history_yaw = list(model.ms.ego.yawQ)[::-1]

        if len(ego_history_x) < 10:
            return False
        
        for key, value in model.ms.vehINAoI.items():
            vehicle_history_x = list(value.xQ)[::-1]
            vehicle_history_y = list(value.yQ)[::-1]
            vehicle_history_yaw = list(value.yawQ)[::-1]
            duration = min(len(ego_history_x), len(vehicle_history_x))
            # max step is 30
            duration = min(duration, 30)
            for i in range(0, duration):
                recA = Rectangle([ego_history_x[i], ego_history_y[i]],
                                 model.ms.ego.length, model.ms.ego.width, ego_history_yaw[i])
                recB = Rectangle([vehicle_history_x[i], vehicle_history_y[i]],
                                 value.length, value.width, vehicle_history_yaw[i])
                rc = RecCollide(recA, recB)
                # if the car collide, stop the simulation
                if rc.isCollide():
                    self.current_reasoning.append("you have a collision with vehicle {}".format(key))
                    raise CollisionException("you have a collision with vehicle {}".format(key))
        return False
    
    def PassRedLight(self, model: Model) -> bool:
        # get the new 10 frames info, if the current lane is junction and last lane is normallane
        # judge the traffic light
        if len(model.ms.ego.laneIDQ) < 11:
            return False
        if model.ms.ego.laneID[0] == ":" and model.ms.ego.laneIDQ[-11][0] != ":" and model.ms.ego.laneID not in self.red_junctionlane_record:
            current_lane = model.nb.getJunctionLane(model.ms.ego.laneID)
            if current_lane.tlLogic != None:
                if current_lane.currTlState == "r" or current_lane.currTlState == "R":
                    self.red_junctionlane_record.append(model.ms.ego.laneID)
                    self.current_reasoning.append("you pass the red light")
                    return True
        return False        

    def TrajectoryCost(self, model: Model) -> Dict:
        # TODO: 修改为打分制度，以便整合到最终的整体评估上
        acc_weight = 0.5
        efficiency_weight = 0.5
        speeding_weight = 0.5
        decision_cost = Dict()
        # 1. acc -- evaluate continuity of the action choose
        ego_history_acc = list(model.ms.ego.accelQ)[::-1]
        duration = min(len(ego_history_acc), 30)
        decision_cost["acc"] = 0
        for i in range(0, duration):
            decision_cost["acc"] += acc_weight * ego_history_acc[i]**2

        # 2. lower speed than other car -- evall speed in last 30 frame
        # find the car drive in the same edge
        # calculate the eval speed
        ego_history_speed = list(model.ego.speedQ)[::-1]
        index = min(len(ego_history_speed), 30)
        ego_eval_speed = sum(ego_history_speed[:index])/index
        decision_cost["efficiency"] = 0.0
        vehicle_num = 0
        for key, value in model.ms.vehINAoI.items():
            if value.edgeID == model.ms.ego.edgeID or value.laneID == model.ms.ego.laneID:
                vehicle_history_speed = list(value.speedQ)[::-1]
                index = min(len(vehicle_history_speed), 30)
                eval_speed = sum(vehicle_history_speed[:index])/index
                if eval_speed > ego_eval_speed:
                    decision_cost["efficiency"] += (eval_speed - ego_eval_speed) * efficiency_weight
                    vehicle_num += 1
        if vehicle_num > 0:
            decision_cost["efficiency"] /= vehicle_num
        
        if decision_cost["efficiency"] > 3.0:
            self.current_reasoning.append("your speed is so low that you have a low efficiency")

        # 3. exceed speed limit, (speeding frames/30 frames)*(vel - speed_limit) as score
        decision_cost["speed_limit"] = 0.0
        lane_id = model.ms.ego.laneID
        if lane_id[0] == ":":
            speed_limit = model.nb.getJunctionLane(lane_id).speed_limit
        else:
            speed_limit = model.nb.getLane(lane_id).speed_limit
        ego_speed = np.array(ego_history_speed[:index])
        np.where(ego_speed > speed_limit, ego_speed - speed_limit, 0)
        decision_cost["speed_limit"] = speeding_weight * np.sum(ego_speed) * (np.count_nonzero(ego_speed)/index)
        if decision_cost["speed_limit"] > 0:
            self.current_reasoning.append("you exceed the speed limit")
        
        return decision_cost
    
    # TODO：还是需要TTC来指示

    # 或许再加一个跟随导航指令？

    def Evaluate(self, model: Model) -> None:
        # 1. collision check
        try:
            self.CollisionCheck(model)
        except CollisionException as e:
            self.complete_percentage = model.ms.ego.lanePos/model.ms.ego.lane_length
            self.final_score = self.current_score * self.complete_percentage
            return
        
        # 2. pass red light check
        if self.PassRedLight(model):
            self.complete_percentage = model.ms.ego.lanePos/model.ms.ego.lane_length
            self.final_score = self.current_score * self.complete_percentage
            return
        
        # 3. trajectory cost
        decision_cost = self.TrajectoryCost(model)
        self.current_score -= sum(decision_cost.values())
        self.decision_score.append(self.current_score)
        self.complete_percentage = model.ms.ego.lanePos/model.ms.ego.lane_length
        self.final_score = self.current_score * self.complete_percentage
        return
    
    @property
    def final_score(self) -> float:
        if self.lanePosQ:
            return self.lanePosQ[-1]
        else:
            raise TypeError('Please call Model.updateVeh() at first.')