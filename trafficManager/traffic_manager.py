"""
This file contains the implementation of the TrafficManager class, which manages the traffic simulation, including vehicle behavior updates,
decision making, and planning. It uses the provided roadgraph and vehicle information to generate trajectories for each vehicle in the simulation.
"""

import copy
import time
from typing import Dict, List, Union
from pynput import keyboard

from common.observation import Observation
from common.vehicle import Behaviour, Vehicle, VehicleType, create_vehicle, create_vehicle_lastseen
from trafficManager.decision_maker.mcts_decision_maker import (
    EgoDecisionMaker,
    MultiDecisionMaker,
)
from planner.ego_vehicle_planner import EgoPlanner
from planner.multi_vehicle_planner import MultiVehiclePlanner
from predictor.simple_predictor import UncontrolledPredictor
from simModel.egoTracking.model import Model
from trafficManager.decision_maker.abstract_decision_maker import AbstractEgoDecisionMaker, EgoDecision
from trafficManager.planner.abstract_planner import AbstractEgoPlanner, AbstractMultiPlanner
from trafficManager.predictor.abstract_predictor import AbstractPredictor
from utils.load_config import load_config
from utils.obstacles import StaticObstacle
from utils.roadgraph import AbstractLane, JunctionLane, NormalLane, RoadGraph
from utils import data_copy
from utils.trajectory import State, Trajectory

import logger


logging = logger.get_logger(__name__)

global KEY_INPUT
KEY_INPUT = ""

class TrafficManager:
    """
    TrafficManager is a class that manages the traffic simulation, including vehicle behavior updates,
    decision making, and planning. It uses the provided roadgraph and vehicle information to generate
    trajectories for each vehicle in the simulation.

    Attributes:
        sumo_model: The SUMO traffic simulation model.
        T: The current simulation time.
        lastseen_vehicles: A dictionary containing the last seen state of each vehicle.
        config: The configuration dictionary.
        predictor: An instance of the UncontrolledPredictor class.
        ego_decision: An instance of the EgoDecisionMaker class.
        ego_planner: An instance of the EgoPlanner class.
        multi_veh_planner: An instance of the MultiVehiclePlanner class.
    """

    def __init__(self,
                 model: Model,
                 predictor: AbstractPredictor = None,
                 ego_decision: AbstractEgoDecisionMaker = None,
                 ego_planner: AbstractEgoPlanner = None,
                 multi_decision=None,
                 multi_veh_planner: AbstractMultiPlanner = None,
                 config_file_path="./trafficManager/config.yaml"):
        self.sumo_model = model
        self.time_step = 0
        self.lastseen_vehicles = {}
        self.config = load_config(config_file_path)
        self.last_decision_time = -self.config["DECISION_INTERVAL"]
        self.mul_decisions =None
        self._set_up_keyboard_listener()

        self.predictor = predictor if predictor is not None else UncontrolledPredictor()
        self.ego_decision = ego_decision if ego_decision is not None else EgoDecisionMaker()
        self.ego_planner = ego_planner if ego_planner is not None else EgoPlanner()
        self.multi_decision = multi_decision if multi_decision is not None else MultiDecisionMaker()
        self.multi_veh_planner = multi_veh_planner if multi_veh_planner is not None else MultiVehiclePlanner()

    def _set_up_keyboard_listener(self):

        def on_press(key):
            """
            This function is used to detect the key press from the keyboard.
            When the left arrow key or 'a' is pressed, the global variable KEY_INPUT is set to 'Left'.
            When the right arrow key or 'd' is pressed, the global variable KEY_INPUT is set to 'Right'.
            """
            global KEY_INPUT
            if key == keyboard.Key.left or key == keyboard.KeyCode.from_char(
                    'a'):
                KEY_INPUT = 'Left'
            elif key == keyboard.Key.right or key == keyboard.KeyCode.from_char(
                    'd'):
                KEY_INPUT = 'Right'

        listener = keyboard.Listener(on_press=on_press)
        listener.start()  # start to listen on a separate thread

    def plan(self, T: float, roadgraph: RoadGraph,
             vehicles_info: dict) -> Dict[int, Trajectory]:
        """
        This function plans the trajectories of vehicles in a given roadgraph. 
        It takes in the total time T, the roadgraph, and the vehicles_info as parameters. 
        It first listens for keyboard input and then extracts the ego car, current vehicles, 
        and uncontrolled vehicles from the vehicles_info. 
        It then updates the behavior of the ego car and current vehicles. 
        It then constructs the observation and predicts the behavior of the uncontrolled vehicles. 
        It then makes a decision for the ego car if the ego planner is enabled. 
        It then plans the trajectories of the vehicles and updates the last seen vehicles. 
        Finally, it returns the output trajectories.
        """
        global KEY_INPUT

        start = time.time()

        current_time_step = int(T / self.config["DT"])
        through_timestep = current_time_step - self.time_step

        # Perception module
        vehicles = self.extract_vehicles(vehicles_info, roadgraph, T,
                                         through_timestep, self.sumo_model.sim_mode)
        history_tracks = self.extract_history_tracks(current_time_step,
                                                     vehicles)
        static_obs_list = self.extract_static_obstacles()
        observation = Observation(vehicles=list(vehicles.values()),
                                  history_track=history_tracks,
                                  static_obstacles=static_obs_list)

        # Prediction Module
        prediction = self.predictor.predict(observation, roadgraph,
                                            self.lastseen_vehicles,
                                            through_timestep, self.config)

        # Update Behavior
        for vehicle_id, vehicle in vehicles.items():
            # only vehicles in AoI will be controlled
            if vehicle.vtype == VehicleType.OUT_OF_AOI:
                continue
            vehicle.update_behaviour(roadgraph, KEY_INPUT)
            KEY_INPUT = ""

        # make sure ego car exists when EGO_PLANNER is used
        if self.config["EGO_PLANNER"]:
            ego_id = vehicles_info.get("egoCar")["id"]
            if ego_id is None:
                raise ValueError("Ego car is not found when EGO_PLANER is used.")

        ego_decision: EgoDecision = None
        if self.config["USE_DECISION_MAKER"] and T - self.last_decision_time >= self.config["DECISION_INTERVAL"]:
            if self.config["EGO_PLANNER"]:
                ego_decision = self.ego_decision.make_decision(
                    observation, roadgraph, prediction)
            self.mul_decisions = self.multi_decision.make_decision(
                T, observation, roadgraph, prediction, self.config)
            self.last_decision_time = T
        # Planner
        result_paths = self.multi_veh_planner.plan(observation, roadgraph,
                                                   prediction,
                                                   multi_decision=self.mul_decisions,
                                                   T=T, config=self.config)

        # an example of ego planner
        if self.config["EGO_PLANNER"]:
            ego_path = self.ego_planner.plan(vehicles[ego_id], observation,
                                             roadgraph, prediction, T,
                                             self.config, ego_decision)
            result_paths[ego_id] = ego_path

        # Update Last Seen
        output_trajectories = {}
        self.lastseen_vehicles = dict(
            (vehicle_id, vehicle)
            for vehicle_id, vehicle in vehicles.items()
            if vehicle.vtype != VehicleType.OUT_OF_AOI)
        for vehicle_id, trajectory in result_paths.items():
            self.lastseen_vehicles[vehicle_id].trajectory = trajectory
            output_trajectories[vehicle_id] = data_copy.deepcopy(trajectory)
            del output_trajectories[vehicle_id].states[0]

        # update self.T
        self.time_step = current_time_step

        logging.info(f"Current frame: {current_time_step}. One loop Time: {time.time() - start}")
        logging.info("------------------------------")

        return output_trajectories

    def extract_history_tracks(self, current_time_step: int,
                               vehicles) -> Dict[int, List[State]]:
        history_tracks = {}
        for vehicle_id in vehicles.keys():
            if vehicle_id not in self.lastseen_vehicles:
                continue
            history_tracks[vehicle_id] = self.lastseen_vehicles[
                vehicle_id].trajectory.states[self.time_step:current_time_step]

        return history_tracks

    def extract_static_obstacles(self) -> List[StaticObstacle]:
        """extract static obstacles for planning

        Returns:
            List[StaticObstacle]: static obstacles in frame
        """
        static_obs_list = []
        return static_obs_list

    def extract_vehicles(
            self, vehicles_info: Dict, roadgraph: RoadGraph, T: float,
            through_timestep: int, sim_mode: str,
    ) -> Dict[int, Vehicle]:
        """
        Extracts vehicles from the provided information and returns them as separate dictionaries.

        Args:
            vehicles_info (dict): Dictionary containing information about the vehicles.
            roadgraph (RoadGraph): Road graph of the simulation.
            lastseen_vehicles (dict): Dictionary containing the last seen vehicles.
            T (float): Current time step.
            through_timestep (int): The number of timesteps the vehicle has been through.
            sumo_model (Any): The SUMO model containing vehicle type information.

        Returns:
            Tuple[Vehicle, Dict[int, Vehicle], Dict[int, Vehicle]]: A tuple containing the ego car, current vehicles, and uncontrolled vehicles.
        """
        vehicles = {}
        ego_car = self.extract_ego_vehicle(vehicles_info, roadgraph, T,
                                           through_timestep,sim_mode)
        if ego_car is not None:
            vehicles[ego_car.id] = ego_car

        for vehicle in vehicles_info["carInAoI"]:
            if not vehicle["xQ"]:
                continue
            if vehicle["id"] in self.lastseen_vehicles  and \
                len(self.lastseen_vehicles[vehicle["id"]].trajectory.states)> through_timestep:
                last_state = self.lastseen_vehicles[
                    vehicle["id"]].trajectory.states[through_timestep]
                vehicles[vehicle["id"]] = create_vehicle_lastseen(
                    vehicle,
                    self.lastseen_vehicles[vehicle["id"]],
                    roadgraph,
                    T,
                    last_state,
                    VehicleType.IN_AOI,
                    sim_mode
                )
            else:
                vtype_info = self.sumo_model.allvTypes[vehicle["vTypeID"]]
                vehicles[vehicle["id"]] = create_vehicle(
                    vehicle, roadgraph, vtype_info, T, VehicleType.IN_AOI)

        for vehicle in vehicles_info["outOfAoI"]:
            if not vehicle["xQ"]:
                continue
            vtype_info = self.sumo_model.allvTypes[vehicle["vTypeID"]]
            if roadgraph.get_lane_by_id(vehicle["laneIDQ"][-1]) is not None:
                vehicles[vehicle["id"]] = create_vehicle(
                    vehicle, roadgraph, vtype_info, T, VehicleType.OUT_OF_AOI)

        ego_cnt = 1 if ego_car is not None else 0
        aoi_cnt = len([
            vehicle for vehicle in vehicles.values()
            if vehicle.vtype == VehicleType.IN_AOI
        ])
        sce_cnt = len([
            vehicle for vehicle in vehicles.values()
            if vehicle.vtype == VehicleType.OUT_OF_AOI
        ])
        logging.info(
            f"There's {ego_cnt} ego cars, {aoi_cnt} cars in AOI, and {sce_cnt} cars in scenario"
        )
        return vehicles

    def extract_ego_vehicle(self, vehicles_info, roadgraph, T,
                            through_timestep,sim_mode) -> Union[None, Vehicle]:
        if "egoCar" not in vehicles_info:
            return None

        ego_info = vehicles_info["egoCar"]
        if not ego_info["xQ"]:
            return None

        ego_id = ego_info["id"]
        if ego_id in self.lastseen_vehicles and \
            len(self.lastseen_vehicles[ego_id].trajectory.states)> through_timestep:
            last_state = self.lastseen_vehicles[ego_id].trajectory.states[
                through_timestep]
            ego_car = create_vehicle_lastseen(
                ego_info,
                self.lastseen_vehicles[ego_id],
                roadgraph,
                T,
                last_state,
                VehicleType.EGO,
                sim_mode
            )
        else:
            vtype_info = self.sumo_model.allvTypes[ego_info["vTypeID"]]
            ego_car = create_vehicle(ego_info, roadgraph, vtype_info, T,
                                        VehicleType.EGO)
        return ego_car
