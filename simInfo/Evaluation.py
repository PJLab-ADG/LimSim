from simModel.CarFactory import Vehicle, egoCar
from utils.trajectory import Trajectory, State, Rectangle, RecCollide
from utils.roadgraph import RoadGraph, JunctionLane
from typing import Dict, List
from simModel.Replay import ReplayModel
import numpy as np
import sqlite3
import logger, logging

class AccJerk_Ref():
    def __init__(self, cc = 0.0, nc = 0.0, ac = 0.0) -> None:
        # upper bound
        self.cautious = cc
        self.normal = nc
        self.agressive = ac

class Hyper_Parameter():
    def __init__(self) -> None:
        # sum = 1.0
        self.score_weight = {
                            "comfort": 0.25,
                            "efficiency": 0.25,
                            "safety": 0.5
                            }
        # confort refers to the acc
        self.longitudinal_acc = AccJerk_Ref(cc = 0.9, nc = 1.47, ac = 3.07)
        self.longitudinal_dec = AccJerk_Ref(cc = -0.9, nc = -2.0, ac = -5.08)
        self.lateral_acc = AccJerk_Ref(cc = 0.9, nc = 4.0, ac = 5.6)
        self.lateral_dec = AccJerk_Ref(cc = -0.9, nc = -4.0, ac = -5.6)

        # confort refers to the jerk
        self.positive_jerk = AccJerk_Ref(cc = 0.6, nc = 0.9, ac = 2.0)
        self.negative_jerk = AccJerk_Ref(cc = -0.6, nc = -0.9, ac = -2.0)
        self.TTC_THRESHOLD = 5

        self.stop_distance = 15.0
        self.judge_speed = 0.5
        self.speed_limit_k = 1.0 # the speed_limit_k lower, penalty bigger

    def calculate_acc_score(self, acc: float, acc_ref: AccJerk_Ref) -> float:
        # Piecewise linear function, 0.6 when normal and 0.0 when aggressive
        self.b1 = 1.0
        self.b2 = 0.6
        self.k1 = 0 # [0-cautious]
        self.k2 = (self.b2 - self.b1) / (acc_ref.normal - acc_ref.cautious)
        self.k3 = (0 - self.b2) / (acc_ref.agressive - acc_ref.normal)

        if abs(acc) <= abs(acc_ref.cautious):
            return self.k1 * (acc - 0) + self.b1
        elif abs(acc) <= abs(acc_ref.normal):
            return self.k2 * (acc - acc_ref.cautious) + self.b1
        elif abs(acc) <= abs(acc_ref.agressive):
            return self.k3 * (acc - acc_ref.normal) + self.b2
        else:
            return 0.0

class Decision_Score():
    def __init__(self) -> None:
        self.comfort = 0.0
        self.efficiency = 0.0
        self.speed_limit = 0
        self.safety = 0.0
        self.red_light = 0.0

        self.longitudinal_acc = 0.0
        self.longitudinal_jerk = 0.0
        self.lateral_acc = 0.0
        self.lateral_jerk = 0.0

    def __repr__(self) -> str:
        return "comfort score is {}, efficiency score is {}, speed limit score is {}, safety score is {}, red light score is {}".format(self.comfort, self.efficiency, self.speed_limit, self.safety, self.red_light)

    def comfort_score(self):
        self.comfort = (self.longitudinal_acc + self.longitudinal_jerk + self.lateral_acc + self.lateral_jerk) / 4
        return self.comfort
    
    def score(self, hyper_parameter: Hyper_Parameter) -> float:
        """Calculate the score for each frame decision.
        decision_score = (comfort_score + efficiency_score + safety_score) * red_light_penalty * speed_limit_penalty

        Args:
            hyper_parameter (Hyper_Parameter): The hyper-parameter set in this program, see it in the Hyper_Parameter class.

        Returns:
            float: The score for each frame decision.
        """
        return (hyper_parameter.score_weight["comfort"] * self.comfort + hyper_parameter.score_weight["efficiency"] * self.efficiency + hyper_parameter.score_weight["safety"] * self.safety) * self.red_light * self.speed_limit

class Score_List(list):
    def __init__(self):
        super().__init__()
        self.penalty = 1.0

    def eval_score(self, hyper_parameter: Hyper_Parameter)-> float:
        """Calculate the driving score for the LLM Driver in this route.
        eval_score = (eval_comfort_score + eval_efficiency_score + eval_safety_score) * red_light_penalty * speed_limit_penalty * collision_penalty;
        collision_penalty = 0.6 if ego has collision, else 1.0;
        red_light_penalty = multiplication of red light scores among all decision scores;
        speed_limit_penalty = 0.9 ^ ((number of speeding frames / total number of decisions) * 10).
        
        Args:
            hyper_parameter (Hyper_Parameter): The hyper-parameter set in this program, see it in the Hyper_Parameter class.

        Returns:
            float: Driving score for the LLM Driver
        """
        comfort = 0.0
        efficiency = 0.0
        speed_limit = 0
        safety = 0.0
        red_light = 1.0
        for score_item in self:
            score_item.comfort = score_item.comfort_score()
            comfort += score_item.comfort
            efficiency += score_item.efficiency
            speed_limit += 1 if score_item.speed_limit == 0.9 else 0
            safety += score_item.safety
            red_light *= score_item.red_light
        comfort /= len(self)
        efficiency /= len(self)
        safety /= len(self)
        speed_limit_penalty = 0.9 ** (speed_limit / len(self) * 10)
        return (hyper_parameter.score_weight["comfort"] * comfort + hyper_parameter.score_weight["efficiency"] * efficiency + hyper_parameter.score_weight["safety"] * safety) * red_light * speed_limit_penalty * self.penalty * 100
    
    def fail_result(self):
        self.penalty = 0.6

class Decision_Evaluation:
    def __init__(self, database: str, timeStep) -> None:
        self.complete_percentage: float = 0.0

        self.decision_score = Score_List() # decision eval based on the current 10 frames
        self.final_score: float = 0.0 # based on the whole decision in one route

        self.red_junctionlane_record = []
        self.current_reasoning = ""

        self.current_time = timeStep

        # odometer
        self.driving_mile = 0.0

        # create database
        conn = sqlite3.connect(database)
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS evaluationINFO")
        conn.commit()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS evaluationINFO(
                frame REAL PRIMARY KEY,
                traffic_light_score REAL,
                comfort_score REAL,
                efficiency_score REAL,
                speed_limit_score REAL,
                collision_score REAL,
                decision_score REAL,
                caution TEXT
            );"""
        )
        conn.commit()
        conn.close()

        self.logger = logger.setup_app_level_logger(logger_name="Evaluation", file_name="llm_decision_result.log")
        self.logging = logging.getLogger("Evaluation").getChild(__name__)
        
        self.hyper_parameter = Hyper_Parameter()

        self.ttc_score = []

    def cal_route_length(self, model: ReplayModel) -> float:
        """calculate the target route length

        Args:
            model (ReplayModel)

        Returns:
            float: length
        """
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
        # if the car just go to new edge, add the length of the last edge
        if model.sr.ego.laneIDQ[-11].split("_")[0] != model.sr.ego.laneIDQ[-1].split("_")[0]:
            if model.sr.ego.laneIDQ[-11][0] == ":":
                self.driving_mile += model.rb.getJunctionLane(model.sr.ego.laneIDQ[-11]).sumo_length
            else:
                self.driving_mile += model.rb.getLane(model.sr.ego.laneIDQ[-11]).sumo_length
        return

    def PassRedLight(self, model: ReplayModel) -> bool:
        # get the new 10 frames info, if the current lane is junction and last lane is normallane
        # judge the traffic light
        if len(model.sr.ego.laneIDQ) < 11:
            return False
        if model.sr.ego.laneID[0] == ":" and model.sr.ego.laneIDQ[-11][0] != ":" and model.sr.ego.laneID not in self.red_junctionlane_record:
            current_lane = model.rb.getJunctionLane(model.sr.ego.laneID)
            if current_lane.currTlState != None:
                if current_lane.currTlState == "r" or current_lane.currTlState == "R":
                    self.red_junctionlane_record.append(model.sr.ego.laneID)
                    self.current_reasoning += "you pass the red light\n"
                    return True
        return False        

    def get_comfort_score(self, ego_vehicle: egoCar, decision_score: Decision_Score) -> float:
        """calculate the comfort score
        score = (acc_score + jerk_score + lateral_acc_score + lateral_jerk_score) / 4
        acc_score = k * (acc_eval - [acc_ref]) + b
        which   k = (b2 - b1) / (acc_ref.normal - acc_ref.cautious)
                b1 = 1.0
                b2 = 0.6

        Args:
            ego_vehicle (egoCar)
            decision_score (Decision_Score)

        Returns:
            float: comfort score
        """
        ego_history_acc = list(ego_vehicle.accelQ)[-11::]

        # 1. calculate longitudinal acc score
        ego_longitudinal_acc = sum(ego_history_acc[1::])/10
        decision_score.longitudinal_acc = self.hyper_parameter.calculate_acc_score(ego_longitudinal_acc, self.hyper_parameter.longitudinal_acc if ego_longitudinal_acc > 0 else self.hyper_parameter.longitudinal_dec)

        # 2. calculate longitudinal jerk score
        ego_longitudinal_jerk = np.mean((np.array(ego_history_acc[1::]) - np.array(ego_history_acc[0:-1]))/0.1)
        decision_score.longitudinal_jerk = self.hyper_parameter.calculate_acc_score(ego_longitudinal_jerk, self.hyper_parameter.positive_jerk if ego_longitudinal_jerk > 0 else self.hyper_parameter.negative_jerk)
        
        # 3. calculate curvature k
        # ref from https://zhuanlan.zhihu.com/p/619658901
        ego_speed = list(ego_vehicle.speedQ)[-11::]
        ego_x = list(ego_vehicle.xQ)[-13::]
        ego_y = list(ego_vehicle.yQ)[-13::]
        ego_xdot = (np.array(ego_x[1::]) - np.array(ego_x[0:-1])) / 0.1
        ego_ydot = (np.array(ego_y[1::]) - np.array(ego_y[0:-1])) / 0.1
        ego_xdd = (ego_xdot[1::] - ego_xdot[0:-1]) / 0.1
        ego_ydd = (ego_ydot[1::] - ego_ydot[0:-1]) / 0.1
        try:
            k = (ego_xdot[1::]*ego_ydd - ego_ydot[1::]*ego_xdd) / (ego_xdot[1::]**2 + ego_ydot[1::]**2)**(3/2)
        except:
            k = 0

        # 4. calculate lateral acc score
        ego_histort_lateral_acc = np.abs(k) * np.array(ego_speed)**2
        ego_lateral_acc = np.mean(ego_histort_lateral_acc[1::])
        decision_score.lateral_acc = self.hyper_parameter.calculate_acc_score(ego_lateral_acc, self.hyper_parameter.lateral_acc if ego_lateral_acc > 0 else self.hyper_parameter.lateral_dec)

        # 5. calculate lateral jerk score
        ego_lateral_jerk = np.mean((ego_histort_lateral_acc[1::] - ego_histort_lateral_acc[0:-1])/0.1)
        decision_score.lateral_jerk = self.hyper_parameter.calculate_acc_score(ego_lateral_jerk, self.hyper_parameter.positive_jerk if ego_lateral_jerk > 0 else self.hyper_parameter.negative_jerk)

        # 6. record if the score is 0
        if decision_score.longitudinal_acc == 0.0:
            self.current_reasoning += f"your longitudinal acc is too large, which is {ego_longitudinal_acc}\n"
        if decision_score.longitudinal_jerk == 0.0:
            self.current_reasoning += f"your longitudinal jerk is too large, which is {ego_longitudinal_jerk}\n"
        if decision_score.lateral_acc == 0.0:
            self.current_reasoning += f"your lateral acc is too large, which is {ego_lateral_acc}\n"
        if decision_score.lateral_jerk == 0.0:
            self.current_reasoning += f"your lateral jerk is too large, which is {ego_lateral_jerk}\n"
        return decision_score.comfort_score()

    def judge_wait_traffic_light(self, model: ReplayModel) -> bool:
        """If the next lane is a junctionlane, and ego is 15m away from the stop line and the speed is less than 0.5 m/s, it is judged that the vehicle is waiting for a red light.

        Args:
            model (ReplayModel)

        Returns:
            bool: True if wait for red light
        """
        if len(model.sr.ego.laneIDQ) < 11:
            return False
        if model.sr.ego.laneID[0] != ":":
            roadgraph, _ = model.sr.exportScene()
            ego_availablelanes = model.sr.ego.availableLanes(model.rb)
            next_lane = roadgraph.get_available_next_lane(model.sr.ego.laneID[0], ego_availablelanes)
            next_lane = model.rb.getJunctionLane(model.sr.ego.laneID)
            if isinstance(next_lane, JunctionLane) and next_lane.currTlState != None:
                if next_lane.currTlState == "r" or next_lane.currTlState == "R":
                    lane_length = model.rb.getLane(model.sr.ego.laneID[0]).sumo_length
                    if lane_length - model.sr.ego.lanePos <= self.hyper_parameter.stop_distance and model.sr.ego.speed <= self.hyper_parameter.judge_speed:
                        return True
        return False
    
    def calculate_ttc(self, model: ReplayModel) -> float:
        """calculate the ttc score, predict the future 5s trajectory, if the trajectory will collide, the time is ttc

        Args:
            model (ReplayModel)

        Returns:
            float: minimum ttc time
        """
        ttc_list = []

        # get ego prediction trajectory
        try:
            roadgraph, _ = model.sr.exportScene()
        except Exception as e:
            return self.hyper_parameter.TTC_THRESHOLD
        ego_availablelanes = model.sr.ego.availableLanes(model.rb)
        ego_trajectory = getSVTrajectory(model.sr.ego, roadgraph, ego_availablelanes, self.hyper_parameter.TTC_THRESHOLD)

        # get other vehicle prediction trajectory
        for _, value in model.sr.vehINAoI.items():
            if value.id == model.sr.ego.id:
                continue
            veh_availablelanes = value.availableLanes(model.rb)
            vehicle_trajectory = getSVTrajectory(value, roadgraph, veh_availablelanes, self.hyper_parameter.TTC_THRESHOLD)

            # calculate ttc, if the (x, y) will collide, the time is ttc
            for index in range(0, min(len(ego_trajectory), len(vehicle_trajectory))):
                recA = Rectangle([ego_trajectory[index].x, ego_trajectory[index].y],
                                model.sr.ego.length, model.sr.ego.width, model.sr.ego.yaw)
                recB = Rectangle([vehicle_trajectory[index].x, vehicle_trajectory[index].y],
                                value.length, value.width, value.yaw)
                rc = RecCollide(recA, recB)
                if rc.isCollide():
                    ttc_list.append(ego_trajectory[index].t)
                    break
        return min(ttc_list) if len(ttc_list) > 0 else self.hyper_parameter.TTC_THRESHOLD
    
    def SaveDatainDB(self, model: ReplayModel, decision_score: Decision_Score) -> None:
        conn = sqlite3.connect(model.dataBase)
        cur = conn.cursor()
        # add evaluation data
        cur.execute(
            """INSERT INTO evaluationINFO (
                frame, traffic_light_score, comfort_score, efficiency_score, speed_limit_score, collision_score, decision_score, caution
                ) VALUES (?,?,?,?,?,?,?,?);""",
            (
                model.timeStep, decision_score.red_light, decision_score.comfort, decision_score.efficiency, decision_score.speed_limit, decision_score.safety, decision_score.score(self.hyper_parameter), self.current_reasoning
            )
        )
        conn.commit()
        conn.close()

    def getResult(self, model: ReplayModel) -> bool:
        conn = sqlite3.connect(model.dataBase)
        cur = conn.cursor()
        cur.execute("SELECT result FROM resultINFO WHERE egoID = ?;", (model.sr.ego.id,))
        result = cur.fetchone()[0]
        conn.close()
        return result
    
    def SaveResultinDB(self, model: ReplayModel) -> None:
        conn = sqlite3.connect(model.dataBase)
        cur = conn.cursor()
        # add result data
        cur.execute(
            """UPDATE resultINFO SET total_score = {},complete_percentage={}, drive_score={} where egoID = {};""".format
            (
                self.final_s, self.complete_percentage, self.decision_score.eval_score(self.hyper_parameter), model.sr.ego.id
            )
        )

        conn.commit()
        conn.close()


    def Current_Decision_Score(self, model: ReplayModel, decision_score: Decision_Score) -> Decision_Score:
        """Calculate the current decision score, include comfort, efficiency, speed limit, safety, red light

        Args:
            model (ReplayModel): class containing current frame information
            decision_score (Decision_Score): current decision score

        Returns:
            Decision_Score: current decision score
        """
        # 1. comfort -- evaluate comfortable of the driving
        decision_score.comfort = self.get_comfort_score(model.sr.ego, decision_score)
        
        # 2. pass red light check
        if self.PassRedLight(model):
            decision_score.red_light = 0.7
        else:
            decision_score.red_light = 1.0

        # 3. lower speed than other car's eval speed in the same edge in last 10 frame        
        lane_id = model.sr.ego.laneID
        if lane_id[0] == ":":
            speed_limit = model.rb.getJunctionLane(lane_id).speed_limit
        else:
            speed_limit = model.rb.getLane(lane_id).speed_limit
        ## 3.1 judge if wait for red light
        if self.judge_wait_traffic_light(model):
            decision_score.efficiency = 1.0
        ## 3.2 calculate the eval speed
        else:
            ego_history_speed = list(model.ego.speedQ)[-10::]
            ego_eval_speed = sum(ego_history_speed) /10
            decision_score.efficiency = 0.0
            vehicle_num = 0
            all_vehicle_eval_speed = 0.0
            # get next lane id
            try:
                roadgraph, _ = model.sr.exportScene()
                ego_availablelanes = model.sr.ego.availableLanes(model.rb)
                next_lane = roadgraph.get_available_next_lane(model.sr.ego.laneID, ego_availablelanes)
            except Exception as e:
                next_lane = None
            for _, value in model.sr.vehINAoI.items():
                if value.id == model.sr.ego.id:
                    continue
                if value.laneID.split("_")[0] == model.sr.ego.laneID.split("_")[0] or (next_lane != None and value.laneID == next_lane.id):
                    vehicle_history_speed = list(value.speedQ)[-10::]
                    eval_speed = sum(vehicle_history_speed)/len(vehicle_history_speed)
                    all_vehicle_eval_speed += eval_speed
                    vehicle_num += 1
            # if there is no car in same edge, need to compare with speed limit
            if vehicle_num > 0 and all_vehicle_eval_speed > 0.1:
                compare_speed = min(all_vehicle_eval_speed / vehicle_num, speed_limit)
                decision_score.efficiency = ego_eval_speed / compare_speed
            elif vehicle_num == 0:
                decision_score.efficiency = ego_eval_speed / speed_limit
            else:
                decision_score.efficiency = 1.0
            decision_score.efficiency = min(1, decision_score.efficiency)
            if decision_score.efficiency < 0.6:
                self.current_reasoning += "your speed is so low that you have a low efficiency\n"

        # 4. exceed speed limit
        decision_score.speed_limit = 0.0
        ego_speed = np.array(ego_history_speed)
        ego_speed = np.where(ego_speed > speed_limit, ego_speed - speed_limit, 0)  
        if np.count_nonzero(ego_speed) > 0:
            decision_score.speed_limit = 0.9
            self.current_reasoning += "you exceed the speed limit\n"
        else:
            decision_score.speed_limit = 1.0

        # 5. ttc: calculate the ego states and other car states in future 5s, take it as ttc(s)
        decision_score.safety = min(self.ttc_score)
        if decision_score.safety <= 0.6:
            self.current_reasoning += "you have high risk of collision with other vehicles\n"
        self.ttc_score = []
        return decision_score
    
    def Evaluate(self, model: ReplayModel) -> None:
        """Executes this function once per frame. It calculates the decision score and saves the data in the database.

        Args:
            model (ReplayModel): class containing current frame information
        """
        self.current_reasoning = ""
        
        # 1. calculate ttc: calculate the ego states and other car states in future 5s, take it as ttc(s)
        self.ttc_score.append(self.calculate_ttc(model) / self.hyper_parameter.TTC_THRESHOLD)
        
        # 2. calculate decision score each 10 frames
        if model.timeStep - self.current_time > 10 and model.timeStep % 10 == 0:
            current_decision_score = Decision_Score() # each decision's score
            self.Current_Decision_Score(model, current_decision_score)
            self.decision_score.append(current_decision_score)
            self.SaveDatainDB(model, current_decision_score)
            # update car mile
            self.CalculateDrivingMile(model)
        
        # 3. if the route is end, calculate the final score
        if model.tpEnd:
            if not self.getResult(model):
                self.decision_score.fail_result()
                self.logger.error("the result is failed")
                self.cal_route_length(model)
                self.CalculateDrivingMile(model)
                self.driving_mile += model.sr.ego.lanePos
                print(self.driving_mile, " ", self.route_length)
            else:
                self.route_length = self.driving_mile
                self.logging.info("the result is success!")

            self.SaveResultinDB(model)
            
            self.logger.info("your final score is {}".format(round(self.final_s, 3)))
            self.logger.info("your driving mile is {} m, the route length is {} m, the complete percentage is {}%".format(round(self.driving_mile, 3), round(self.route_length, 3), round(self.complete_p*100, 3)))
            self.logger.info("your driving score is {}".format(round(self.decision_score.eval_score(self.hyper_parameter), 3)))
            self.logger.info("your driving time is {} s".format((model.timeStep - self.current_time)/10))
            
        return
        
    @property
    def final_s(self) -> float:
        self.final_score = self.decision_score.eval_score(self.hyper_parameter) * self.complete_p
        return self.final_score
    
    @property
    def complete_p(self) -> float:
        self.complete_percentage = self.driving_mile / self.route_length
        return self.complete_percentage
    

def getSVTrajectory(vehicle: Vehicle, roadgraph: RoadGraph, available_lanes, T: float) -> List[State]:
    """Get the future 5s trajectory of vehicle

    Args:
        vehicle (Vehicle): predict object
        roadgraph (RoadGraph)
        available_lanes (_type_)
        T (float): predict duration (s)

    Returns:
        List[State]: state in future T times
    """
    # judge the vehicle lane and position
    prediction_trajectory = Trajectory()
    
    # if the vehicle's lane position is near the end of the lane, need the next lane to predict
    next_lane = roadgraph.get_available_next_lane(
        vehicle.laneID, available_lanes)
    
    current_lane = roadgraph.get_lane_by_id(vehicle.laneID)
    lanes = [current_lane, next_lane] if next_lane != None else [
        current_lane]
    if next_lane != None:
        next_next_lane = roadgraph.get_next_lane(next_lane.id)
        if next_next_lane != None:
            lanes.append(next_next_lane)
            next_3_lane = roadgraph.get_next_lane(next_next_lane.id)
            if next_3_lane != None:
                lanes.append(next_3_lane)
    
    for t in range(0, T*20, 1):
        t = t/20
        prediction_trajectory.states.append(State(s = vehicle.lanePos + vehicle.speed * t, t=t))
    prediction_trajectory.frenet_to_cartesian(lanes, State(s = vehicle.lanePos, s_d = vehicle.speed,laneID=vehicle.laneID, yaw=vehicle.yaw, x=vehicle.x, y=vehicle.y, vel=vehicle.speed, acc=vehicle.accel, t=0))
    
    return prediction_trajectory.states