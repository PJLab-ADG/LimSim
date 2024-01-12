from trafficManager.common.vehicle import Behaviour, Vehicle, VehicleType
from utils.trajectory import Trajectory, State, Rectangle, RecCollide
from typing import Dict, List
from simModel.Replay import ReplayModel
import numpy as np
from DriverAgent.collision_statistics import compute_time_to_collision
import sqlite3
import logger, logging


# TODO: 1. 把这个写成离线的，然后不再对应decision的评价，那么如何提取好的决策结果保存到memory呢？也可以手动添加我觉得
# 2. 修改dirveagent代码结构，把system mesage等信息外挂，为了规范LLM的输出，可以写example，同时写好output，并入到这个仓库里
# 3. 离线跑一轮，看看效果和评价准确性
# 4. 增加memory，确定形式和检索方法
# 5. reflection可以把连续的几帧决策过程和结果都输入给它（好像reflection只给失败的决策也可以），同时补偿一些额外知识给它，要确定reflection的输出结果和格式是什么，然后怎么运用
# 6. 修改换道的规划，保持速度不变

# sum = 1.0
penalty_weight = {
    "acc": 0.1,
    "efficiency": 0.3,
    "speed_limit": 0.2,
    "ttc": 0.3,
    "red_light": 0.3
}


class CollisionException(Exception):
    def __init__(self, ErrorInfo: str) -> None:
        super().__init__(self)
        self.errorinfo = ErrorInfo
    
    def __str__(self) -> str:
        return self.errorinfo

class Decision_Evaluation:
    def __init__(self, database: str, timeStep) -> None:
        self.complete_percentage: float = 0.0

        self.decision_score: List[float] = [] # decision eval based on the current 10 frames
        self.current_score = 100.0 # Driving score based on the whole routes
        self.final_score: float = 0.0 # based on the whole routes

        self.red_junctionlane_record = []
        self.current_reasoning = ""

        self.current_time = timeStep

        self.ttc_statistics = compute_time_to_collision(database)
        self.driving_mile = 0.0
        self.last_pos = 0.0

        # create database
        conn = sqlite3.connect(database)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS evaluationINFO")
        conn.commit()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS evaluationINFO(
                frame REAL PRIMARY KEY,
                traffic_light_score REAL,
                acc_score REAL,
                efficiency_score REAL,
                speed_limit_score REAL,
                ttc_score REAL,
                decision_score REAL,
                caution TEXT
            );"""
        )
        conn.commit()
        conn.close()

        self.logger = logger.setup_app_level_logger(logger_name="Evaluation", file_name="llm_decision_result.log")
        self.logging = logging.getLogger("Evaluation").getChild(__name__)

    def cal_route_length(self, model: ReplayModel) -> float:
        route_length = 0.0
        _, LLRDict, _ = model.ego.getLaneLevelRoute(model.rb)
        for edgeID, laneDict in LLRDict.items():
            for laneType, laneIDset in laneDict.items():
                if laneType == 'edgeLanes':
                    laneID = laneIDset.pop()
                    route_length += model.rb.getLane(laneID).sumo_length
                if laneType == "junctionLanes":
                    laneID = laneIDset.pop()
                    route_length += model.rb.getJunctionLane(laneID).sumo_length
        self.route_length = route_length
        return route_length
    
    def CalculateDrivingMile(self, model: ReplayModel) -> None:
        # if the car just go to new edge
        if self.last_pos - model.sr.ego.lanePos > 1:
            self.driving_mile += model.sr.ego.lanePos
        else:
            self.driving_mile += model.sr.ego.lanePos - self.last_pos
        self.last_pos = model.sr.ego.lanePos
        return

    def CollisionCheck(self, model: ReplayModel) -> bool:
        # vehicle trajectory collision need to be checked in every frame
        for key, value in model.sr.vehINAoI.items():
                recA = Rectangle([model.sr.ego.x, model.sr.ego.y],
                                 model.sr.ego.length, model.sr.ego.width, model.sr.ego.yaw)
                recB = Rectangle([value.x, value.y],
                                 value.length, value.width, value.yaw)
                rc = RecCollide(recA, recB)
                # if the car collide, stop the simulation
                if rc.isCollide():
                    self.current_reasoning += "you have a collision with vehicle {}\n".format(key)
                    raise CollisionException("you have a collision with vehicle {}".format(key))
        return False
    
    def PassRedLight(self, model: ReplayModel) -> bool:
        # get the new 10 frames info, if the current lane is junction and last lane is normallane
        # judge the traffic light
        if len(model.sr.ego.laneIDQ) < 11:
            return False
        if model.sr.ego.laneID[0] == ":" and model.sr.ego.laneIDQ[-11][0] != ":" and model.sr.ego.laneID not in self.red_junctionlane_record:
            current_lane = model.rb.getJunctionLane(model.sr.ego.laneID)
            if current_lane.tlLogic != None:
                if current_lane.currTlState == "r" or current_lane.currTlState == "R":
                    self.red_junctionlane_record.append(model.sr.ego.laneID)
                    self.current_reasoning += "you pass the red light\n"
                    return True
        return False        

    def Trajectoryscore(self, model: ReplayModel) -> Dict:
        # TODO: 修改为打分制度，以便整合到最终的整体评估上
        # TODO: 打分制度的问题在于，如何把这几个score归一化到一个范围内，然后再加权，归一化的范围比较难界定
        # 以及效率还没有加入到最终评分里
        ttc_thresholds = 10
        decision_score = dict()
        # 1. acc -- evaluate continuity of the action choose
        # score = 1/(sum(abs(acc))/time + 1)
        ego_history_acc = list(model.sr.ego.accelQ)[::-1]
        duration = min(len(ego_history_acc), 10)
        decision_score["acc"] = 0

        for i in range(0, duration):
            decision_score["acc"] += ego_history_acc[i]**2
        decision_score["acc"] = 1 / (decision_score["acc"] / duration + 1) * 100

        lane_id = model.sr.ego.laneID
        if lane_id[0] == ":":
            speed_limit = model.rb.getJunctionLane(lane_id).speed_limit
        else:
            speed_limit = model.rb.getLane(lane_id).speed_limit
            
        # 2. lower speed than other car -- evall speed in last 10 frame
        # find the car drive in the same edge
        # calculate the eval speed
        ego_history_speed = list(model.ego.speedQ)[::-1]
        index = min(len(ego_history_speed), 10)
        ego_eval_speed = sum(ego_history_speed[:index])/index
        decision_score["efficiency"] = 0.0
        vehicle_num = 0
        all_vehicle_eval_speed = 0.0
        for key, value in model.sr.vehINAoI.items():
            if value.laneID.split("_")[0] == model.sr.ego.laneID.split("_")[0]:
                vehicle_history_speed = list(value.speedQ)[::-1]
                index = min(len(vehicle_history_speed), 10)
                eval_speed = sum(vehicle_history_speed[:index])/index
                all_vehicle_eval_speed += eval_speed
                # if eval_speed > ego_eval_speed:
                #     decision_score["efficiency"] += (eval_speed - ego_eval_speed) * efficiency_weight
                vehicle_num += 1
        # if there is no car in same edge, need to compare with speed limit
        if vehicle_num > 0 and all_vehicle_eval_speed > 0.1:
            decision_score["efficiency"] = ego_eval_speed / all_vehicle_eval_speed / vehicle_num * 100
        else:
            decision_score["efficiency"] = 100
        if decision_score["efficiency"] > 100:
            decision_score["efficiency"] = 100
        if decision_score["efficiency"] < 60.0:
            self.current_reasoning += "your speed is so low that you have a low efficiency\n"

        # 3. exceed speed limit, speed_limit/(speeding frames/ num)*100 as score
        decision_score["speed_limit"] = 0.0
        
        ego_speed = np.array(ego_history_speed[:index])
        ego_speed = np.where(ego_speed > speed_limit, ego_speed - speed_limit, 0)  
        if np.count_nonzero(ego_speed) > 0:
            decision_score["speed_limit"] = speed_limit / (np.sum(ego_speed) / np.count_nonzero(ego_speed)) * 100
        else:
            decision_score["speed_limit"] = 100
        if decision_score["speed_limit"] >= 100:
            decision_score["speed_limit"] = 100 # 没有超速
        else:
            self.current_reasoning += "you exceed the speed limit\n"

        # 4. ttc
        decision_score["ttc"] = 0.0
        index, _ = np.where(self.ttc_statistics == model.timeStep)
        ttc_step = np.sum(self.ttc_statistics[(index[0]-10):index[0], 1]) / 10
        decision_score["ttc"] = ttc_step / ttc_thresholds * 100
        if decision_score["ttc"] > 100:
            decision_score["ttc"] = 100
        if decision_score["ttc"] < 50:
            self.current_reasoning += "you have a risk of collision\n"
        return decision_score
    
    def SaveDatainDB(self, model: ReplayModel, decision_score) -> None:
        conn = sqlite3.connect(model.dataBase)
        cur = conn.cursor()
        # add evaluation data
        cur.execute(
            """INSERT INTO evaluationINFO (
                frame, traffic_light_score, acc_score, efficiency_score, speed_limit_score, ttc_score, decision_score, caution
                ) VALUES (?,?,?,?,?,?,?,?);""",
            (
                model.timeStep, decision_score["red_light"], decision_score["acc"], decision_score["efficiency"], decision_score["speed_limit"], decision_score["ttc"], self.decision_score[-1], self.current_reasoning
            )
        )
        conn.commit()
        conn.close()

    def Evaluate(self, model: ReplayModel) -> None:
        self.current_reasoning = ""
        # update car mile
        self.CalculateDrivingMile(model)
        # 1. collision check
        try:
            self.CollisionCheck(model)
        except CollisionException as e:
            # 存数据
            self.cal_route_length(model)
            self.current_score *= 0.6
            self.logger.error("the result is failed, reason is ", e)
            self.logger.info("your final score is {}".format(round(self.final_s, 3)))
            self.logger.info("your driving mile is {} m, the route length is {} m, the complete percentage is {}%".format(round(self.driving_mile, 3), round(self.route_length, 3), round(self.complete_percentage*100, 3)))
            self.logger.info("your driving time is {} s".format((model.timeStep - self.current_time)/10))
            raise e
        
        if model.timeStep - self.current_time > 10 and model.timeStep % 10 == 0:
            current_decision_score = 0 # current decision score, based on all item score
            current_decision_item_score = dict() # each item score
            # 2. pass red light check
            if self.PassRedLight(model):
                self.current_score *= 0.7
                current_decision_item_score["red_light"] = 70
                # current_decision_score *= (1 - penalty_weight["red_light"] * 70 / 100)
            else:
                current_decision_item_score["red_light"] = 100
            # 3. trajectory score
            current_decision_item_score.update(self.Trajectoryscore(model))
            for key, value in current_decision_item_score.items():
                # if value < 100:
                current_decision_score += penalty_weight[key] * value
                # current_decision_score *= (1 - penalty_weight[key] * value / 100)
            # current_decision_score /= 5
            self.decision_score.append(current_decision_score)
            self.SaveDatainDB(model, current_decision_item_score)

        if model.tpEnd:
            self.cal_route_length(model)
            self.logger.info("your final score is {}".format(round(self.final_s, 3)))
            self.logger.info("your driving mile is {} m, the route length is {} m, the complete percentage is {}%".format(round(self.driving_mile, 3), round(self.route_length, 3), round(self.complete_percentage*100, 3)))
            self.logger.info("your driving time is {} s".format((model.timeStep - self.current_time)/10))
        return
    
    @property
    def final_s(self) -> float:
        self.final_score = self.current_score * self.complete_p * (1 - (sum(self.decision_score)/len(self.decision_score)) / 100)
        return self.final_score
    
    @property
    def complete_p(self) -> float:
        self.complete_percentage = self.driving_mile / self.route_length
        return self.complete_percentage