'''
Author: Licheng Wen
Date: 2022-12-02 14:41:39
Description: 

Copyright (c) 2022 by PJLab, All Rights Reserved. 
'''

import numpy as np
from common.observation import Observation
from predictor.abstract_predictor import AbstractPredictor, Prediction
from trafficManager.common.vehicle import VehicleType

from utils.roadgraph import RoadGraph
from utils.trajectory import State, Trajectory


class UncontrolledPredictor(AbstractPredictor):
    def predict(
        self, observation: Observation, roadgraph: RoadGraph,
        lastseen_vehicles, through_timestep, config) -> Prediction:
        prediction = Prediction()

        for vehicle in observation.vehicles:
            if vehicle.vtype != VehicleType.OUT_OF_AOI:
                if vehicle.id in lastseen_vehicles:
                    prediction.results[vehicle] = lastseen_vehicles[vehicle.id].trajectory.states[through_timestep:]
            else:
                lane = roadgraph.get_lane_by_id(vehicle.lane_id)
                predict_t = config["MIN_T"]
                dt = config["DT"]
                s = vehicle.current_state.s
                d = vehicle.current_state.d
                s_d = vehicle.current_state.s_d

                predict_trajectory = Trajectory()
                for t in np.arange(0, predict_t, dt):
                    predict_trajectory.states.append(
                        State(t=t, d=d, s=s, s_d=s_d,))
                    s += s_d * dt
                next_lane = roadgraph.get_next_lane(lane.id)
                lanes = [lane, next_lane] if next_lane != None else [lane]
                predict_trajectory.frenet_to_cartesian(
                    lanes, vehicle.current_state)

                prediction.results[vehicle] = predict_trajectory.states
        return prediction
