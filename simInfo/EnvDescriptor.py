from trafficManager.common.vehicle import Behaviour
import logger, logging
from utils.roadgraph import JunctionLane, NormalLane, RoadGraph, AbstractLane
from utils.trajectory import State, Trajectory
from typing import Dict, List, Tuple, Set
import numpy as np
import json

ACTIONS_DESCRIPTION = {
    Behaviour.LCL: 'Turn-left - change lane to the left of the current lane',
    Behaviour.IDLE: 'IDLE - remain in the current lane with current speed',
    Behaviour.LCR: 'Turn-right - change lane to the right of the current lane',
    Behaviour.AC: 'Acceleration - accelerate the vehicle',
    Behaviour.DC: 'Deceleration - decelerate the vehicle'
}


class EnvDescription:
    def __init__(
            self, template_path = "./simInfo/prompt_template.json"
    ) -> None:

        self.decision = None

        self.last_decision_time = 0

        with open(template_path, 'r', encoding="utf-8") as f:
            self.des_json = json.load(f)

        self.logger = logger.setup_app_level_logger(logger_name = "prompt", file_name="prompt_debug.log")
        self.logging = logging.getLogger("prompt").getChild(__name__)

    def getCurrentLaneInfo(self, roadgraph: RoadGraph, vehicle_info: Dict) -> str:
        """get current lane info, including length, speed limit, lane type

        Args:
            roadgraph (RoadGraph): roadgraph info
            vehicle_info (Dict): all of vehicles info

        Returns:
            str: lane prompt
        """
        current_lane_describe = "### Current lane description:\n"

        ego = vehicle_info["egoCar"]
        current_lane = roadgraph.get_lane_by_id(ego["laneIDQ"][-1])

        if isinstance(current_lane, JunctionLane):
            current_lane_describe += self.des_json["basic_description"]["current_lane_scenario_description"]["junction_lane"]

        
        elif isinstance(current_lane, NormalLane):
            #----------- get lane num ------------#
            edge = current_lane.affiliated_edge
            edge_num = 0
            for lane_id in edge.lanes:
                lane = roadgraph.get_lane_by_id(lane_id)
                if lane.width > ego["width"]:
                    edge_num += 1

            #----------- get lane index ------------#
            left_lane_num = 1
            left_lane_id = current_lane.left_lane()

            while left_lane_id != None:
                left_lane = roadgraph.get_lane_by_id(left_lane_id)
                if left_lane.width > ego["width"]:
                    left_lane_num += 1
                left_lane_id = left_lane.left_lane()

            current_lane_describe += self.des_json["basic_description"]["current_lane_scenario_description"]["normal_lane"].format(edge_num = edge_num, left_lane_num = left_lane_num, lane_length = round(current_lane.course_spline.s[-1], 3))

        current_lane_describe += self.des_json["basic_description"]["traffic_info_description"].format(speed_limit = current_lane.speed_limit)

        return current_lane_describe + "\n"
    
    def getNextLaneInfo(self, roadgraph: RoadGraph, vehicle_info: Dict) -> str:
        """get next lane info
            - if the next lane is normal lane: return number of lanes
            - if the next lane is junction lane: return the traffic light info

        Args:
            roagraph (RoadGraph)
            vehicle_info (Dict)

        Returns:
            str: next lane prompt
        """
        next_lane_describe = "### Next lane description:\n"
        ego = vehicle_info["egoCar"]
        current_lane = roadgraph.get_lane_by_id(ego["laneIDQ"][-1])
        next_lane = None
        if isinstance(current_lane, NormalLane):
            next_junction_id = current_lane.affiliated_edge.to_junction
            for lane_id in ego["availableLanes"]:
                if next_junction_id in lane_id:
                    next_lane = roadgraph.get_lane_by_id(lane_id)
                    
        else:
            next_lane = roadgraph.get_available_next_lane(ego["laneIDQ"][-1], ego["availableLanes"])
        
        if next_lane == None:
            next_lane_describe += self.des_json["basic_description"]["next_lane_scenario_description"]["none"]

        elif isinstance(next_lane, NormalLane):
            #----------- get lane num ------------#
            edge = next_lane.affiliated_edge
            edge_num = 0
            for lane_id in edge.lanes:
                lane = roadgraph.get_lane_by_id(lane_id)
                if lane.width > ego["width"]:
                    edge_num += 1
            next_lane_describe += self.des_json["basic_description"]["next_lane_scenario_description"]["normal_lane"].format(edge_num = edge_num)

        elif isinstance(next_lane, JunctionLane):
            #----------- get traffic light state ------------#
            if next_lane.currTlState != None:
                tl_state = "with"
            else:
                tl_state = "without"
            dis_stop_line = round(current_lane.spline_length - ego["lanePosQ"][-1], 3)
            
            next_lane_describe += self.des_json["basic_description"]["next_lane_scenario_description"]["junction_lane"].format(tl_state = tl_state, dis_stop_line = dis_stop_line)

            if tl_state == "with":
                next_lane_describe += self.des_json["basic_description"]["traffic_light_description"].format(curr_tl_state = self.trafficLightProcess(next_lane.currTlState))

                #TODO: get the green light time
                # if self.trafficLightProcess(next_lane.currTlState) == "green" and next_lane.switchTime < 10:
                #     if self.trafficLightProcess(next_lane.nexttTlState) != "green":
                #         next_lane_describe += self.des_json["basic_description"]["traffic_light_change_des"].format(color = self.trafficLightProcess(next_lane.nexttTlState), time=next_lane.switchTime)

        return next_lane_describe + "\n"
    
    def trafficLightProcess(self, state) -> str:
        if state == "G" or state == "g":
            return "green"
        elif state == "R" or state == "r":
            return "red"
        else:
            return "yellow"
        
    def getEgoInfo(self, vehicle_info: Dict[str, Dict]) -> str:
        """get ego info, including speed, position, acceleration

        Args:
            vehicle_info (Dict[str, Dict]): all of vehicles info

        Returns:
            str: ego prompt
        """
        ego_describe = "### Your current state:\n"
        ego = vehicle_info["egoCar"]
        ego_position = str((round(ego["xQ"][-1], 3), round(ego["yQ"][-1], 3)))

        ego_describe += self.des_json["basic_description"]["ego_state_description"].format(ego_position = ego_position, ego_lane_position = round(ego["lanePosQ"][-1], 3), ego_speed = round(ego["speedQ"][-1], 3), ego_acceleration = round(ego["accelQ"][-1], 3))

        return ego_describe + "\n"
    
    def getNavigationInfo(self, roadgraph: RoadGraph, vehicles: Dict[str, Dict]) -> str:
        """get navigation info, including lane change, go straight and go through the junction

        Args:
            roadgraph (RoadGraph): information of the relationship between lanes
            vehicles (Dict[str, Dict]): all of vehicles info

        Returns:
            str: navigation prompt
        """
        nav_describe = ''
        ego = vehicles['egoCar']
        curr_lane_id: str = ego['laneIDQ'][-1]
        availableLanes: Set[str] = ego['availableLanes']
        curr_lane = roadgraph.get_lane_by_id(curr_lane_id)

        # ----------- need to change lane ------------ #
        # no need to change lane in junction or correct lane
        if not(curr_lane_id[0] == ':' or curr_lane_id in availableLanes):
            curr_lane_idx = int(curr_lane_id.split('_')[-1])
            for al in availableLanes:
                if al[0] != ':':
                    al_idx = int(al.split('_')[-1])
                    if al_idx > curr_lane_idx:
                        self.logging.info(f"Ego vehicle choose to change Left lane")
                        nav_describe += self.des_json["navigation_instruction"]["left"]

                    else:
                        self.logging.info(f"Ego vehicle choose to change Right lane")
                        nav_describe += self.des_json["navigation_instruction"]["right"]
                    break
        
        # ------------ in junction ------------ #
        elif curr_lane_id[0] == ':':
            nav_describe = self.des_json["navigation_instruction"]["junction"]
            
        # ------------ normal straight ------------ #
        else:
            nav_describe = self.des_json["navigation_instruction"]["straight"]
        
        return nav_describe

    def getNoticeInfo(self, roadgraph: RoadGraph, vehicles: Dict[str, Dict]) -> str:
        """get notice info, including distance to stop line, speed limit, other vehicles

        Args:
            roadgraph (RoadGraph)

        Returns:
            str: notice prompt
        """
        notice_description = self.des_json["intension"]["basic"]

        ego = vehicles['egoCar']
        curr_lane_id: str = ego['laneIDQ'][-1]
        curr_lane = roadgraph.get_lane_by_id(curr_lane_id)
        next_lane = roadgraph.get_available_next_lane(curr_lane_id, ego["availableLanes"])
        # 1. If the distance from the intersection stop line is less than 30 meters, make notice 
        if curr_lane_id[0] != ':' and curr_lane.spline_length - ego["lanePosQ"][-1] < 30:
            notice_description += self.des_json["intension"]["junction"]
            if next_lane.currTlState != None:
                notice_description += self.des_json["intension"]["traffic_light"]

        # 2. notice about speed
        elif ego["speedQ"][-1] < 5.0:
            notice_description += self.des_json["intension"]["low_speed"]
        
        elif ego["speedQ"][-1] > curr_lane.speed_limit * 0.9:
            notice_description += self.des_json["intension"]["high_speed"]
        
        # 3. notice about other vehicle
        if not ("There are no other vehicles" in self.getOtherVehicleInfo(roadgraph, vehicles)):
            notice_description += self.des_json["intension"]["other_vehicles"]

        return notice_description
    
    def getLastDecisionInfo(self, vehicle: Dict[str, Dict]) -> str:
        """get last decision description

        Args:
            current_decision_time (float): last time
            last_decision_time (float): current time

        Returns:
            str: last decision prompt
        """
        ego = vehicle["egoCar"]

        last_decision_describe = "### Last decision:\n"
        if self.decision != None:
            last_decision_describe += self.des_json["basic_description"]["last_decision_description"]["basic"].format(delta_time = 1, decision = self.decision)
            if ego["laneIDQ"][-1][0] != ":" and (self.decision == Behaviour.LCL or self.decision == Behaviour.LCR):
                if ego["laneIDQ"][-11] != ego["laneIDQ"][-1]:
                    last_decision_describe += self.des_json["basic_description"]["last_decision_description"]["changed_lane"]
                else:
                    last_decision_describe += self.des_json["basic_description"]["last_decision_description"]["changing_lane"].format(direction = "right" if self.decision == Behaviour.LCR else "left")

            return last_decision_describe + "\n"
        else:
            return ""

    def getAvailableActionsInfo(self, roadgraph: RoadGraph, vehicle: Dict[str, Dict]) -> str:
        """get available actions info

        Args:
            roadgraph (RoadGraph)
            vehicle (Dict[str, Dict])

        Returns:
            str: available actions prompt, including action id and action description
        """
        avaliable_action_description = 'Your available actions are: \n'
        ego = vehicle["egoCar"]
        current_lane = roadgraph.get_lane_by_id(ego["laneIDQ"][-1])
        available_actions = [Behaviour.AC, Behaviour.IDLE, Behaviour.DC, Behaviour.LCL, Behaviour.LCR]

        if ego["laneIDQ"][-1][0] == ':':
            available_actions.remove(Behaviour.LCL) if Behaviour.LCL in available_actions else None
            available_actions.remove(Behaviour.LCR) if Behaviour.LCR in available_actions else None

        else:
            if current_lane.left_lane() == None or roadgraph.get_lane_by_id(current_lane.left_lane()).width < ego["width"]:
                available_actions.remove(Behaviour.LCL) if Behaviour.LCL in available_actions else None
            if current_lane.right_lane() == None or roadgraph.get_lane_by_id(current_lane.right_lane()).width < ego["width"]:
                available_actions.remove(Behaviour.LCR) if Behaviour.LCR in available_actions else None

        if ego["speedQ"][-1] > current_lane.speed_limit:
            available_actions.remove(Behaviour.AC) if Behaviour.AC in available_actions else None
        if ego["speedQ"][-1] <= 0.1:
            available_actions.remove(Behaviour.DC) if Behaviour.DC in available_actions else None

        for action in available_actions:
            avaliable_action_description += ACTIONS_DESCRIPTION[action] + ' Action_id: ' + str(
                action.value) + '\n'
        return avaliable_action_description
    
    def getOtherVehicleInfo(self, roadgraph: RoadGraph, vehicles: Dict[str, Dict]) -> str:
        """get other vehicle info, including distance, speed, acceleration

        Args:
            vehicles (Dict): all of vehicles info

        Returns:
            str: other vehicle prompt
        """
        other_vehicle_describe = "### Nearby vehicles description:\n"
        ego = vehicles["egoCar"]
        other_vehicles = vehicles["carInAoI"]

        if len(other_vehicles) == 0:
            other_vehicle_describe += self.des_json["basic_description"]["other_vehicle_description"]["no_other_vehicle"]
            return other_vehicle_describe + "\n"

        svDescription = "There are other vehicles driving around you, and below is their basic information:\n"
        is_sv = False
        sv_list = other_vehicles[:]
        current_lane = roadgraph.get_lane_by_id(ego["laneIDQ"][-1])

        # ---------- in normal lane, need to consider the vehicles in left, right and current lane ---------- #
        if ego["laneIDQ"][-1][0] != ":":
            ego_edge = ego["laneIDQ"][-1].split('_')[0]
            # find the sv in the same edge with ego
            for sv in sv_list:
                sv_edge = sv["laneIDQ"][-1].split('_')[0]
                if sv_edge == ego_edge:
                    same_edge_sv = self.describeSVNormalLane(sv, ego, current_lane)
                    if same_edge_sv != "":
                        is_sv = True
                        svDescription += same_edge_sv
                    other_vehicles.remove(sv)
        
            # find the sv in the next lane of ego
            sv_list = other_vehicles[:]
            next_lane = roadgraph.get_available_next_lane(ego["laneIDQ"][-1], ego["availableLanes"])
            if next_lane != None:
                for sv in sv_list:
                    if sv["laneIDQ"][-1] == next_lane.id:
                        other_vehicles.remove(sv)
                        is_sv = True
                        svDescription += self.describeSVNormalLane(sv, ego, current_lane, next_lane = True)

        #------- in junction lane, need to consider the vehicles in the same lane with ego -------#
        else:
            # find the sv in the same lane with ego
            for sv in sv_list:
                if sv["laneIDQ"][-1] == ego["laneIDQ"][-1]:
                    is_sv = True
                    svDescription += self.describeSVNormalLane(sv, ego, current_lane)
                    other_vehicles.remove(sv)

            # find the sv in the next lane of ego
            sv_list = other_vehicles[:]
            next_lane = roadgraph.get_available_next_lane(ego["laneIDQ"][-1], ego["availableLanes"])
            if next_lane != None:
                for sv in sv_list:
                    if sv["laneIDQ"][-1] == next_lane.id:
                        other_vehicles.remove(sv)
                        is_sv = True
                        svDescription += self.describeSVNormalLane(sv, ego, current_lane, next_lane = True)
        

        # ---------- in junction lane, need to consider the vehicles will collide with ego ---------- #
        sv_list = other_vehicles[:]
        ego_prediction = self.getSVTrajectory(ego, roadgraph)
        for sv in sv_list:
            prediction_trajectory = self.getSVTrajectory(sv, roadgraph)

            collision_des = self.describeSVInAOI(sv, ego_prediction, prediction_trajectory)
            if collision_des != "":
                svDescription += collision_des
                is_sv = True

        if not is_sv:
            svDescription = self.des_json["basic_description"]["other_vehicle_description"]["no_other_vehicle"]
        
        other_vehicle_describe += svDescription
        
        return other_vehicle_describe + "\n"
    
    def describeSVNormalLane(self, vehicle: Dict, ego: Dict, current_lane: AbstractLane, next_lane: bool = False) -> str:
        """get the description of the vehicle in normal lane, left/right/same lane with ego, and the distance between them, also provide their speed and acceleration

        Args:
            vehicle (Dict): car info
            next_lane (bool, optional): is the sv in the next lane of the ego. Defaults to False.

        Returns:
            str: _description_
        """
        if next_lane:
            lane_relative_position = "same lane as you"
            relative_position = "ahead"
            distance = round(current_lane.spline_length - ego["lanePosQ"][-1] + vehicle["lanePosQ"][-1], 3)

        else:
            if vehicle["laneIDQ"][-1] == current_lane.id:
                # The vehicle and ego are traveling in the same lane
                lane_relative_position = "the same lane as you"
            elif vehicle["laneIDQ"][-1] == current_lane.left_lane():
                # The vehicle drives in the left lane of ego
                lane_relative_position = "your left lane"
            elif vehicle["laneIDQ"][-1] == current_lane.right_lane():
                # The vehicle drives in the right lane of ego
                lane_relative_position = "your right lane"
            else:
                return ''
        
            if vehicle["lanePosQ"][-1] - ego["lanePosQ"][-1] >= 0:
                relative_position = 'ahead'
            else:
                relative_position = 'behind'
            distance = round(abs(vehicle["lanePosQ"][-1] - ego["lanePosQ"][-1]), 3)

        sv_position = str((round(vehicle["xQ"][-1], 3), round(vehicle["yQ"][-1], 3)))

            
        sv_normal_des = self.des_json["basic_description"]["other_vehicle_description"]["surrond_vehicle_on_normal_description"].format(
                            sv_id = vehicle["id"],
                            lane_relative_position = lane_relative_position,
                            relative_position = relative_position,
                            sv_speed = round(vehicle["speedQ"][-1], 3),
                            sv_acceleration = round(vehicle["accelQ"][-1], 3),
                            sv_lane_position = round(vehicle["lanePosQ"][-1], 3),
                            sv_position = sv_position,
                            distance = distance)

        return sv_normal_des + "\n"

    def describeSVInAOI(self, vehicle: Dict, ego_prediction: List[State], prediction_state: List[State]) -> str:
        """get the description of the vehicle in junction lane, and the distance between them, also provide their speed and acceleration

        Args:
            vehicle (Dict): car info
            ego_prediction (List[State]): ego trajectory for the next 5 seconds
            prediction_state (List[State]): sv trajectory for the next 5 seconds

        Returns:
            str: vehicle in junction prompt
        """

        # Compute the intersection point of trajectories
        if prediction_state == None or ego_prediction == None or len(prediction_state) == 0 or len(ego_prediction) == 0:
            self.logging.info("the prediction state of vehicle {} is None".format(vehicle["id"]))
            return ""
        else:
            [ego_time, ego_s, sv_time, sv_s] = self.trajectory_overlap(vehicle, ego_prediction, prediction_state)
            if ego_s != None:
                sv_junction_des = self.des_json["basic_description"]["other_vehicle_description"]["surrond_vehicle_on_junction_description"].format(
                                    sv_id = vehicle["id"],
                                    sv_speed = round(vehicle["speedQ"][-1], 3),
                                    sv_acc = round(vehicle["accelQ"][-1], 3),
                                    ego_s = round(ego_s, 3),
                                    ego_time = round(ego_time, 3),
                                    sv_s = round(sv_s, 3),
                                    sv_time = round(sv_time, 3))
            
            else:
                return ""
        
        return sv_junction_des + "\n"

    def trajectory_overlap(self, sv:Dict, ego_traj: List[State], sv_traj: List[State]):
        """judge if the ego trajectory and sv trajectory will overlap in the next 5 seconds

        Args:
            ego_traj (List[State]): trajectory of ego
            sv_traj (List[State]): trajectory of SV

        Returns:
            ego_time: time for ego to arrive the collision point
            ego_s: distance between ego and collision point
            sv_time: time for SV to arrive the collision point
            sv_s: distance between SV and collision point
        """
        # Calculate the distance between each point on the two trajectories 
        ego_time, ego_s, sv_time, sv_s = None, None, None, None

        ego_xy = np.array([[state.x, state.y] for state in ego_traj]) # m * 2

        sv_xy = np.array([[state.x, state.y] for state in sv_traj]) # n * 2

        m, _ = ego_xy.shape
        n, _ = sv_xy.shape
        arr1_power = np.power(ego_xy, 2)
        arr1_power_sum = arr1_power[:, 0] + arr1_power[:, 1]
        arr1_power_sum = np.tile(arr1_power_sum, (n, 1))
        arr1_power_sum = arr1_power_sum.T
        arr2_power = np.power(sv_xy, 2)
        arr2_power_sum = arr2_power[:, 0] + arr2_power[:, 1]
        arr2_power_sum = np.tile(arr2_power_sum, (m, 1))
        dis = arr1_power_sum + arr2_power_sum - (2 * np.dot(ego_xy, sv_xy.T))
        dis = np.sqrt(dis) # m * n
        dis_statis = np.argwhere(dis < sv["width"])

        if dis_statis.size != 0:
            ego_min_index, sv_min_index = dis_statis[0]

            # If the collision occurs at the first point of ego and the first point of sv, it means that the vehicle behind ego does not need to be considered. 
            if ego_min_index == 0:
                pass
            else:
                ego_s = ego_traj[ego_min_index].s - ego_traj[0].s
                sv_s = sv_traj[sv_min_index].s - sv_traj[0].s
                ego_time = ego_traj[ego_min_index].t if ego_min_index != 0 else 0
                sv_time = sv_traj[sv_min_index].t if sv_min_index != 0 else 0
        
        return [ego_time, ego_s, sv_time, sv_s]

    def getSVTrajectory(self, vehicle: Dict, roadgraph: RoadGraph) -> List[State]:
        """get the trajectory of the vehicle in the next 5 seconds

        Args:
            vehicle (Dict): car info
            roadgraph (RoadGraph): roadgraph info

        Returns:
            List[State]: the trajectory of the vehicle in the next 5 seconds
        """
        # judge the vehicle lane and position
        # current_lane = roadgraph.get_lane_by_id(vehicle.lane_id)
        prediction_trajectory = Trajectory()
        next_lane = roadgraph.get_available_next_lane(
            vehicle["laneIDQ"][-1], vehicle["availableLanes"])
        current_lane = roadgraph.get_lane_by_id(vehicle["laneIDQ"][-1])
        pos_s, pos_d = current_lane.course_spline.cartesian_to_frenet1D(vehicle["xQ"][-1], vehicle["yQ"][-1])
        current_state = State(x=vehicle["xQ"][-1],
                       y=vehicle["yQ"][-1],
                       yaw=vehicle["yawQ"][-1],
                       s=pos_s,
                       d=pos_d,
                       s_d=vehicle["speedQ"][-1],
                       s_dd=vehicle["accelQ"][-1],
                       t=0)
        lanes = [current_lane, next_lane] if next_lane != None else [
            current_lane]
        for t in range(0, 50, 1):
            t = t/10
            prediction_trajectory.states.append(State(t = t, d = pos_d, s = vehicle['lanePosQ'][-1] + vehicle["speedQ"][-1] * t))
        prediction_trajectory.frenet_to_cartesian(lanes, current_state)
        
        return prediction_trajectory.states
    
    def getEnvPrompt(self, roadgraph: RoadGraph, vehicles: Dict[str, Dict]) -> str:
        """get the environment prompt info

        Args:
            roadgraph (RoadGraph): roadgraph info
            vehicles (Dict[str, Dict]): all of vehicles info

        Returns:
            str: prompt info
        """
        prompt = ""
        prompt += self.getCurrentLaneInfo(roadgraph, vehicles)
        prompt += self.getNextLaneInfo(roadgraph, vehicles)
        prompt += self.getEgoInfo(vehicles)
        prompt += self.getOtherVehicleInfo(roadgraph, vehicles)
        prompt += self.getLastDecisionInfo(vehicles)
        return prompt