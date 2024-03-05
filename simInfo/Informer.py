"""
- Provide VLMAgent with driving assistance information such as navigation information and current action_list 
- Provides VLMAgent with driver assistance information, such as navigation information and the current action_list
"""
from typing import Dict, Optional, Set
from utils.roadgraph import RoadGraph

class Informer:
    @staticmethod
    def getActionInfo(vehicles: Dict[str, Dict], roadgraph: RoadGraph):
        ActionInfo = '## Available Actions\n'
        ACCInfo = ' - AC: increasing your speed\n'
        DECInfo = ' - DC: slow down your speed\n'
        MVCInfo = ' - IDLE: Maintain your current speed\n'
        CLInfo = ' - LCL: Change lanes to the left\n'
        CRInfo = ' - LCR: Change lanes to the right\n'
        ego_info = vehicles['egoCar']
        curr_lane_id: str = ego_info['laneIDQ'][-1]
        curr_lane = roadgraph.get_lane_by_id(curr_lane_id)
        if curr_lane_id[0] == ':':
            # Do not change lanes when inside an intersection 
            ActionInfo += ACCInfo
            ActionInfo += DECInfo
            ActionInfo += MVCInfo
        else:
            curr_edge = curr_lane.affiliated_edge
            if curr_edge.lane_num == 1:
                # Do not change lanes when there is only one lane in the current lane 
                ActionInfo += ACCInfo
                ActionInfo += DECInfo
                ActionInfo += MVCInfo
            else:
                curr_lane_idx = curr_lane_id.split('_')
                if curr_lane_idx == '0':
                    # in the rightest lane. 
                    ActionInfo += ACCInfo
                    ActionInfo += DECInfo
                    ActionInfo += MVCInfo
                    ActionInfo += CLInfo
                elif curr_lane_idx == str(curr_edge.lane_num-1):
                    # In the leftmost lane.
                    ActionInfo += ACCInfo
                    ActionInfo += DECInfo
                    ActionInfo += MVCInfo
                    ActionInfo += CRInfo
                else:
                    ActionInfo += ACCInfo
                    ActionInfo += DECInfo
                    ActionInfo += MVCInfo
                    ActionInfo += CLInfo
                    ActionInfo += CRInfo
        return ActionInfo
    
    @staticmethod
    def getNaviInfo(vehicles: Dict[str, Dict]) -> Optional[str]:
        ego_info = vehicles['egoCar']
        curr_speed = ego_info['speedQ'][-1]
        SpeedINFO = f'Your current speed is {round(curr_speed, 2)} m/s, the speed limit is 13.89 m/s.\n'
        curr_lane_id: str = ego_info['laneIDQ'][-1]
        availableLanes: Set[str] = ego_info['availableLanes']
        NaviTitle = '## Navigation Information\n'
        if curr_lane_id[0] == ':':
            # there is no navigation info when ego in junction
            return NaviTitle + SpeedINFO + "You should just drive carefully.\n"
        else:
            if curr_lane_id in availableLanes:
                # no need to generate navigation info when ego in the correct lane
                return NaviTitle + SpeedINFO + "You are on the proper lane.\n"
            else:
                curr_lane_idx = int(curr_lane_id.split('_')[-1])
                for al in availableLanes:
                    if al[0] != ':':
                        al_idx = int(al.split('_')[-1])
                        if al_idx > curr_lane_idx:
                            NaviInfo = 'Please change lane to the left as soon as possible.\n'
                            return NaviTitle + SpeedINFO + NaviInfo
                        else:
                            NaviInfo = 'Please change lane to the right as soon as possible.\n'
                            return NaviTitle + SpeedINFO + NaviInfo
    
