"""
This module contains the Vehicle class and related functions for managing vehicles in a traffic simulation.

Classes:
    Behaviour (IntEnum): Enum class for vehicle behavior.
    Vehicle: Represents a vehicle in the simulation.

Functions:
    create_vehicle(vehicle_info, roadgraph: RoadGraph, T, vtype_info, vtype) -> Vehicle:
        Creates a new Vehicle instance based on the provided information.
    create_vehicle_lastseen(vehicle_info, lastseen_vehicle, roadgraph: RoadGraph, T, through_timestep, vtype) -> Vehicle:
        Creates a Vehicle instance based on the last seen vehicle information.
    extract_vehicles(vehicles_info: dict, roadgraph: RoadGraph, lastseen_vehicles: dict, T: float, through_timestep: int, sumo_model) -> Tuple[Vehicle, Dict[int, Vehicle], Dict[int, Vehicle]]:
        Extracts vehicles from the provided information and returns them as separate dictionaries.
"""
from copy import copy, deepcopy
from enum import IntEnum, Enum
from typing import Any, Dict, Set, Tuple

from trafficManager.common.coord_conversion import cartesian_to_frenet2D
from utils.roadgraph import AbstractLane, JunctionLane, NormalLane, RoadGraph
from utils.trajectory import State

import logger

logging = logger.get_logger(__name__)


class Behaviour(IntEnum):
    KL = 0
    AC = 1
    DC = 2
    LCL = 3
    LCR = 4
    STOP = 5
    IN_JUNCTION = 6
    OVERTAKE = 7
    IDLE = 8
    OTHER = 100

class VehicleType(str, Enum):
    EGO = "Ego_Car"
    IN_AOI = "Car_In_AoI"
    OUT_OF_AOI = "Car_out_of_AoI"


class Vehicle:

    def __init__(self,
                 vehicle_id: int,
                 init_state: State = State(),
                 lane_id: int = -1,
                 target_speed: float = 0.0,
                 behaviour: Behaviour = Behaviour.KL,
                 vtype: VehicleType = VehicleType.OUT_OF_AOI,
                 length: float = 5.0,
                 width: float = 2.0,
                 max_accel: float = 3.0,
                 max_decel: float = -3.0,
                 max_speed: float = 50.0,
                 available_lanes: Dict[int, Any] = {}) -> None:
        """
        Initialize a Vehicle instance.

        Args:
            vehicle_id (int): Vehicle ID.
            init_state (State, optional): Initial state of the vehicle. Defaults to State().
            lane_id (int, optional): Lane ID of the vehicle. Defaults to -1.
            target_speed (float, optional): Target speed of the vehicle. Defaults to 0.0.
            behaviour (Behaviour, optional): Behaviour of the vehicle. Defaults to Behaviour.KL.
            vtype (str, optional): Vehicle type. Defaults to "outOfAoI".
            length (float, optional): Length of the vehicle. Defaults to 5.0.
            width (float, optional): Width of the vehicle. Defaults to 2.0.
            max_accel (float, optional): Maximum acceleration of the vehicle. Defaults to 3.0.
            max_decel (float, optional): Maximum deceleration of the vehicle. Defaults to -3.0.
            max_speed (float, optional): Maximum speed of the vehicle. Defaults to 50.0.
            available_lanes (Dict[int, Any], optional): Available lanes for the vehicle. Defaults to {}.
        """
        self.id = vehicle_id
        self._current_state = init_state
        self.lane_id = lane_id
        self.behaviour = behaviour
        self.target_speed = target_speed
        self.vtype = vtype
        self.length = length
        self.width = width
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.max_speed = max_speed
        self.available_lanes = available_lanes

    @property
    def current_state(self) -> State:
        """
        Get the current state of the vehicle.

        Returns:
            State: The current state of the vehicle.
        """
        return self._current_state

    @current_state.setter
    def current_state(self, state: State) -> None:
        """
        Set the current state of the vehicle.

        Args:
            state (State): The new state of the vehicle.
        """
        self._current_state = state

    def get_state_in_lane(self, lane) -> State:
        course_spline = lane.course_spline

        rs = course_spline.find_nearest_rs(self.current_state.x,
                                           self.current_state.y)

        rx, ry = course_spline.calc_position(rs)
        ryaw = course_spline.calc_yaw(rs)
        rkappa = course_spline.calc_curvature(rs)

        s, s_d, d, d_d = cartesian_to_frenet2D(rs, rx, ry, ryaw, rkappa,
                                               self.current_state)
        return State(s=s, s_d=s_d, d=d, d_d=d_d,
                     x=self.current_state.x,
                     y=self.current_state.y,
                     yaw=self.current_state.yaw,
                     vel=self.current_state.vel,
                     acc=self.current_state.acc)

    def change_to_lane(self, lane: AbstractLane) -> None:
        """
        Change the vehicle to the next lane.
        """
        self.lane_id = lane.id
        self.current_state = self.get_state_in_lane(lane)
        self.behaviour = Behaviour.KL

    def __repr__(self) -> str:
        """
        Get the string representation of the vehicle.

        Returns:
            str: The string representation of the vehicle.
        """
        return f"Vehicle(id={self.id}, lane_id={self.lane_id}, "\
               f"current_state=(s={self.current_state.s}, d={self.current_state.d}, "\
               f"s_d={self.current_state.s_d}, s_dd={self.current_state.s_dd}))"

    def update_behavior_with_manual_input(self, manual_input: str,
                                          current_lane: AbstractLane):
        """use keyboard to send lane change command to ego car.

        Args:
            manual_input (str): the left turn or right turn command
            current_lane (AbstractLane): the lane that ego car lies on
        """
        if self.vtype != VehicleType.EGO:
            # currently, manual input only works on ego car
            return
        if self.behaviour != Behaviour.KL:
            # lane change behavior works only when ego is in lane keeping state.
            return
        if manual_input == 'Left' and current_lane.left_lane(
        ) in self.available_lanes:
            self.behaviour = Behaviour.LCL
            logging.info(f"Key command Vehicle {self.id} to change Left lane")

        elif manual_input == 'Right' and current_lane.right_lane(
        ) in self.available_lanes:
            self.behaviour = Behaviour.LCR
            logging.warning(
                f"Key command Vehicle {self.id} to change Right lane")

    def update_behaviour(self, roadgraph: RoadGraph) -> None:
        """Update the behaviour of a vehicle.

        Args:
            roadgraph (RoadGraph): The roadgraph containing the lanes the vehicle is traveling on.
        """
        current_lane = roadgraph.get_lane_by_id(self.lane_id)
        logging.debug(
            f"Vehicle {self.id} is in lane {self.lane_id}, "
            f"In available_lanes? {current_lane.id in self.available_lanes}")

        # Lane change behavior
        if isinstance(current_lane, NormalLane):
            if self.behaviour == Behaviour.LCL:
                left_lane_id = current_lane.left_lane()
                left_lane = roadgraph.get_lane_by_id(left_lane_id)
                state = self.get_state_in_lane(left_lane)
                if state.d > -left_lane.width / 2:
                    self.change_to_lane(left_lane)

            elif self.behaviour == Behaviour.LCR:
                right_lane_id = current_lane.right_lane()
                right_lane = roadgraph.get_lane_by_id(right_lane_id)
                state = self.get_state_in_lane(right_lane)
                if state.d < right_lane.width / 2:
                    self.change_to_lane(right_lane)

            elif self.behaviour == Behaviour.IN_JUNCTION:
                self.behaviour = Behaviour.KL
            elif current_lane.id not in self.available_lanes:
                logging.debug(
                    f"Vehicle {self.id} need lane-change, "
                    f"since {self.lane_id} not in available_lanes {self.available_lanes}"
                )

                if self.behaviour == Behaviour.KL:
                    # find left available lanes
                    lane = current_lane
                    while lane.left_lane() is not None:
                        lane_id = lane.left_lane()
                        if lane_id in self.available_lanes:
                            self.behaviour = Behaviour.LCL
                            logging.info(
                                f"Vehicle {self.id} choose to change Left lane")
                            break
                        lane = roadgraph.get_lane_by_id(lane_id)
                if self.behaviour == Behaviour.KL:
                    # find right available lanes
                    lane = current_lane
                    while lane.right_lane() is not None:
                        lane_id = lane.right_lane()
                        if lane_id in self.available_lanes:
                            self.behaviour = Behaviour.LCR
                            logging.info(
                                f"Vehicle {self.id} choose to change Right lane"
                            )
                            break
                        lane = roadgraph.get_lane_by_id(lane_id)
                if self.behaviour == Behaviour.KL:
                    # can not reach to available lanes
                    logging.error(
                        f"Vehicle {self.id} cannot change to available lanes, "
                        f"current lane {self.lane_id}, available lanes {self.available_lanes}"
                    )

        # in junction behaviour
        if self.current_state.s > current_lane.course_spline.s[-1] - 0.2:
            if isinstance(current_lane, NormalLane):
                next_lane = roadgraph.get_available_next_lane(
                    current_lane.id, self.available_lanes)
                try:
                    self.lane_id = next_lane.id
                except AttributeError as e:
                    print(self.id)
                    logging.error(f"Vehicle {self.id} cannot switch to the next lane, the reason is {e}")
                self.current_state = self.get_state_in_lane(next_lane)
                current_lane = next_lane
            elif isinstance(current_lane, JunctionLane):
                next_lane_id = current_lane.next_lane_id
                next_lane = roadgraph.get_lane_by_id(next_lane_id)
                self.lane_id = next_lane.id
                self.current_state = self.get_state_in_lane(next_lane)
                current_lane = next_lane
            else:
                logging.error(
                    f"Vehicle {self.id} Lane {self.lane_id}  is unknown lane type {type(current_lane)}"
                )

            if isinstance(current_lane, JunctionLane):  # in junction
                self.behaviour = Behaviour.IN_JUNCTION
                logging.info(f"Vehicle {self.id} is in {self.behaviour}")
            else:  # out junction
                self.behaviour = Behaviour.KL


def create_vehicle(vehicle_info: Dict, roadgraph: RoadGraph, vtype_info: Any,
                   T,
                   vtype: VehicleType) -> Vehicle:
    """
    Creates a new Vehicle instance based on the provided information.

    Args:
        vehicle_info (Dict): Information about the vehicle.
        roadgraph (RoadGraph): Road graph of the simulation.
        T (float): Current time step.
        vtype_info (Any): Vehicle type information.
        vtype (str): Vehicle type.

    Returns:
        Vehicle: A new Vehicle instance.
    """
    available_lanes = vehicle_info["availableLanes"]
    lane_id = vehicle_info["laneIDQ"][-1]
    lane_pos = vehicle_info["lanePosQ"][-1]
    pos_x = vehicle_info["xQ"][-1]
    pos_y = vehicle_info["yQ"][-1]
    yaw = vehicle_info["yawQ"][-1]
    speed = vehicle_info["speedQ"][-1]
    # acc = vehicle_info["accelQ"].pop()
    acc = 0

    lane_id, pos_s, pos_d = find_lane_position(lane_id, roadgraph,
                                               available_lanes, lane_pos, pos_x,
                                               pos_y)

    init_state = State(x=pos_x,
                       y=pos_y,
                       yaw=yaw,
                       s=pos_s,
                       d=pos_d,
                       s_d=speed,
                       s_dd=acc,
                       t=T)
    return Vehicle(
        vehicle_id=vehicle_info["id"],
        init_state=init_state,
        lane_id=lane_id,
        target_speed=30.0 / 3.6,
        behaviour=Behaviour.KL,
        vtype=vtype,
        length=vtype_info.length,
        width=vtype_info.width,
        max_accel=vtype_info.maxAccel,
        max_decel=-vtype_info.maxDecel,
        max_speed=vtype_info.maxSpeed,
        available_lanes=available_lanes,
    )


def find_lane_position(lane_id: str, roadgraph: RoadGraph,
                       available_lanes: Set[str],
                       lane_pos: float, pos_x: float,
                       pos_y: float) -> Tuple[str, float, float]:
    """Given the map information and cartesian coordinate, 
       find corresponding frenet coordinates 

    Args:
        lane_id (str): initial guess of lane that cartesian coordinate lies on
        roadgraph (RoadGraph): map information
        available_lanes (Set[str]): possible lanes
        lane_pos (float): initial guess of s coordinate
        pos_x (float): cartesian x coordinate
        pos_y (float): cartesian y coordinate

    Returns:
        Tuple[str, float, float]: A tuple (lane_id, s coordinate, d coordinate)
    """
    lane = roadgraph.get_lane_by_id(lane_id)

    # lan_pos &lane_id is wrong in internal-junction lane
    # https://sumo.dlr.de/docs/Networks/SUMO_Road_Networks.html#internal_junctions
    if lane is None or (isinstance(lane, JunctionLane) and
                        lane.id not in available_lanes):
        for available_lane_id in available_lanes:
            lane = roadgraph.get_lane_by_id(available_lane_id)
            if not isinstance(lane, JunctionLane):
                continue

            pos_s, pos_d = lane.course_spline.cartesian_to_frenet1D(
                pos_x, pos_y)

            if abs(pos_d) < 2.0:
                return available_lane_id, pos_s, pos_d
    else:
        pos_s, pos_d = lane.course_spline.cartesian_to_frenet1D(pos_x, pos_y)
        return lane_id, pos_s, pos_d
    return None, None, None


def create_vehicle_lastseen(vehicle_info: Dict, lastseen_vehicle: Vehicle,
                            roadgraph: RoadGraph, T: float, last_state: State,
                            vtype: VehicleType, sim_mode: str) -> Vehicle:
    """
    Creates a Vehicle instance based on the last seen vehicle information.

    Args:
        vehicle_info (Dict): Information about the vehicle.
        lastseen_vehicle (Vehicle): The last seen vehicle instance.
        roadgraph (RoadGraph): Road graph of the simulation.
        T (float): Current time step.
        through_timestep (int): The number of timesteps the vehicle has been through.
        vtype (str): Vehicle type.

    Returns:
        Vehicle: A new Vehicle instance with updated information.
    """
    vehicle = copy(lastseen_vehicle)
    vehicle.current_state = last_state
    vehicle.current_state.t = T
    vehicle.current_state.x = vehicle_info["xQ"][-1]
    vehicle.current_state.y = vehicle_info["yQ"][-1]
    vehicle.vtype = vtype
    vehicle.available_lanes = vehicle_info["availableLanes"]
    lane_id = vehicle_info["laneIDQ"][-1]
    # in some junctions, sumo will not find any lane_id for the vehicle
    while lane_id == "":
        logging.debug("lane_id is empty")
        lane_id = vehicle_info["laneIDQ"].pop()
        if lane_id != "":
            lane_id = roadgraph.get_available_next_lane(
                lane_id, vehicle_info["availableLanes"]).id

    # don't care about lane_id while in junction
    if sim_mode == 'InterReplay' or all(isinstance(roadgraph.get_lane_by_id(lane), NormalLane)
                                        for lane in (vehicle.lane_id, lane_id)):
        # fixme: really need?
        if lane_id != vehicle.lane_id and \
                vehicle.behaviour in (Behaviour.LCL, Behaviour.LCR):
            logging.warning(f"Vehicle {vehicle.id} have changed"
                            f"lane from {vehicle.lane_id} to {lane_id}")
            vehicle.behaviour = Behaviour.KL

        lanepos = vehicle_info["lanePosQ"][-1]
        vehicle.lane_id = lane_id
        lane = roadgraph.get_lane_by_id(lane_id)
        x = vehicle_info["xQ"][-1]
        y = vehicle_info["yQ"][-1]
        try:
            s, d = lane.course_spline.cartesian_to_frenet1D(x, y)
        except TypeError:
            logging.error("Vehicle line 185:", vehicle_info["lanePosQ"],
                          lanepos, lane.course_spline.s[-1])
            exit(1)
        vehicle.current_state.x = x
        vehicle.current_state.y = y
        vehicle.current_state.s = s
        vehicle.current_state.d = d

    return vehicle


def get_lane_id(vehicle_info, roadgraph):
    lane_id = vehicle_info["laneIDQ"].pop()
    # in some junctions, sumo will not find any lane_id for the vehicle
    while lane_id == "":
        logging.debug("lane_id is empty")
        lane_id = vehicle_info["laneIDQ"].pop()
        if lane_id != "":
            lane_id = roadgraph.get_available_next_lane(
                lane_id, vehicle_info["availableLanes"]).id

    return lane_id
