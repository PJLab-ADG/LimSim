from dataclasses import dataclass, field
import sqlite3
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class EvaluationState:
    """A simple state class to represent vehicle state
    """
    x: float
    y: float
    yaw: float
    speed: float

@dataclass
class EvaluationModel:
    """A simple vehicle model class to represent internal parameter of a vehicle
    """
    width: float
    length: float
    height: float = 0.0


@dataclass
class EvaluationVehicle:
    """A simple vehicle class used to check collision
    """
    vehicle_id: int
    model: EvaluationModel
    # time stamp to state
    states: Dict[int, EvaluationState] = field(default_factory=dict)


def separate_axis_theorem(vertices_a: np.ndarray,
                          vertices_b: np.ndarray) -> bool:
    """Check if two polygon overlaps, polygons are represented by its vertices

    Args:
        vertices1 (np.ndarray): of shape (n, 2), vertices of polygon a
        vertices2 (np.ndarray): of shape (m, 2), vertices of polygon b

    Returns:
        bool: True if two polygons overlap

    Reference:
        https://github.com/Chanuk-Yang/Deep_Continuous_Fusion_for_Multi-Sensor_3D_Object_Detection

        http://programmerart.weebly.com/separating-axis-theorem.html
    """
    # find all possible axes
    n, m = vertices_a.shape[0], vertices_b.shape[0]
    edges_a: List[np.ndarray] = [
        vertices_a[(i + 1) % n] - vertices_a[i] for i in range(n)
    ]
    edges_b: List[np.ndarray] = [
        vertices_a[(i + 1) % m] - vertices_a[i] for i in range(m)
    ]
    edges: List[np.ndarray] = edges_a + edges_b
    axes: List[np.ndarray] = [np.array([edge[1], -edge[0]]) for edge in edges]
    # normalize
    axes: List[np.ndarray] = [
        axis / np.linalg.norm(axis, ord=2) for axis in axes
    ]

    # check for collision with separate axis theorem
    for axis in axes:
        projection_a: np.ndarray = np.array(
            [np.dot(vertex, axis) for vertex in vertices_a])
        projection_b: np.ndarray = np.array(
            [np.dot(vertex, axis) for vertex in vertices_b])

        # if two line segments do not overlaps, then two polygons do not overlap
        if np.min(projection_a) > np.max(projection_b) or \
           np.min(projection_b) > np.max(projection_a):
            return False

    return True


@dataclass
class Rectangle:
    """A simple rectangle representation

                  length
    w ----------------------------
    i -                          -
    d -           center         -
    t -                          -
    h ----------------------------
    yaw is in radians and in anti-clockwise direction
    """
    center: np.ndarray
    width: float
    length: float
    yaw: float

    def __repr__(self) -> str:
        return f"center={self.center}, width={self.width}, length={self.length}, yaw={self.yaw}"

    def corners(self) -> np.ndarray:
        """Get the coordinates of corners of a rectangle

        Returns:
            List[np.ndarray]: an array of shape (4, 2)
            each of them represents the coordinate of a corner
        """
        points = np.array([[self.length, self.width], [
            -self.length, self.width
        ], [-self.length, -self.width], [self.length, -self.width]]) / 2
        rotation = np.array([[np.cos(self.yaw), -np.sin(self.yaw)],
                             [np.sin(self.yaw),
                              np.cos(self.yaw)]])
        return self.center + np.array(
            [rotation.dot(point) for point in points])

    def in_collision(self, other_rectangle: 'Rectangle') -> bool:
        """check if two rectangles intersects (in collision)

        Args:
            other_rectangle (Rectangle): another rectangle

        Returns:
            bool: True if two rectangle intersects, False otherwise
        """
        self_corners = self.corners()
        other_corners = other_rectangle.corners()

        # first use AABB to filter impossible cases
        self_x_max, self_y_max = np.max(self_corners, axis=0)
        self_x_min, self_y_min = np.min(self_corners, axis=0)
        other_x_max, other_y_max = np.max(other_corners, axis=0)
        other_x_min, other_y_min = np.min(other_corners, axis=0)

        if self_x_max < other_x_min or self_x_min > other_x_max or \
           self_y_max < other_y_min or self_y_min > other_y_max:
            return False

        # use separate axis theorem check in-collision or not
        return separate_axis_theorem(self_corners, other_corners)


def get_data_frame(database_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """extract dataframe from SQL databse

    Args:
        database_name (str): path to database

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: scenario information and vehicle information
    """
    connector = sqlite3.connect(database_name)
    frame_df = pd.read_sql_query('''SELECT * FROM frameINFO''', connector)
    vehicle_df = pd.read_sql_query('''SELECT * FROM vehicleINFO''', connector)
    connector.close()

    return frame_df, vehicle_df


def extract_vehicles(
    frame_df: pd.DataFrame, vehicle_df: pd.DataFrame
) -> Tuple[EvaluationVehicle, List[EvaluationVehicle]]:
    """extract vehicles information from scenario information and vehicle model information
       only ego vehicle and vehicles in AoI will be considered.

    Args:
        frame_df (pd.DataFrame): data frame that stores scenario information
        vehicle_df (pd.DataFrame): data frame that stores vehicle model information

    Returns:
        Tuple[EvaluationVehicle, List[EvaluationVehicle]]: 
        the ego vehicle information and vehicle information on AoI
    """
    vehicles: List[EvaluationVehicle] = []
    # get vehicle model info
    for index, row in vehicle_df.iterrows():
        vehicles.append(
            EvaluationVehicle(vehicle_id=row["vid"],
                              model=EvaluationModel(width=row["width"],
                                                    length=row["length"])))

    # get states information of vehicle
    ego_index: int = 0
    for index, vehicle in enumerate(vehicles):
        related_df = frame_df[(frame_df["vid"] == vehicle.vehicle_id)
                              & (frame_df["vtag"] != "outOfAoI")]
        if related_df.empty:
            continue
        vehicle.states = {
            row["frame"]: EvaluationState(*row[["x", "y", "yaw", "speed"]])
            for _, row in related_df.iterrows()
        }

        if all(related_df["vtag"] == "ego"):
            ego_index = index

    # split vehicles to ego and others
    ego_vehicle = vehicles[ego_index]
    other_vehicles = [
        vehicle for vehicle in vehicles if vehicle != ego_vehicle
    ]
    return ego_vehicle, other_vehicles


def relative_angle(ego_state: EvaluationState,
                   other_state: EvaluationState) -> float:
    """Get the relative angle between ego state and the other state

    Args:
        ego_state (EvaluationState): state of ego
        other_state (EvaluationState): state of other vehicle

    Returns:
        float: the angle (in radians) between ego and other vehicle
    """
    vec_a = np.array([np.cos(ego_state.yaw), np.sin(ego_state.yaw)])
    vec_b = np.array([other_state.x, other_state.y]) - \
            np.array([ego_state.x, ego_state.y])

    return np.arccos((np.dot(vec_a, vec_b)) /
                     (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)))


def extract_vehicles_excluding_impossible_vehicles(
    ego_vehicle: EvaluationVehicle,
    other_vehicles: List[EvaluationVehicle],
) -> List[List[EvaluationVehicle]]:
    """Only vehicles in AoI will be considered, new Vehicle instances are 
       created to reduce memory use

    Args:
        ego_vehicle (EvaluationVehicle): ego
        other_vehicles (List[EvaluationVehicle]): original vehicles information

    Returns:
        List[List[EvaluationVehicle]]: A list of time-related vehicles information
        its length is equal to length of states of ego, ego vehicle contains one state
    """
    result: List[List[EvaluationVehicle]] = []
    for timestamp in ego_vehicle.states.keys():
        vehicles = [
            EvaluationVehicle(vehicle_id=vehicle.vehicle_id,
                              states={timestamp: vehicle.states[timestamp]},
                              model=vehicle.model)
            for vehicle in other_vehicles
            if timestamp in vehicle.states.keys()
        ]
        result.append(np.array(vehicles))

    return result


def get_long_box_center(state: EvaluationState,
                        time_horizon: float) -> np.ndarray:
    """Assume constant speed and orientation, find center of long box generated by
       trajectory

    Args:
        state (EvaluationState): current state
        time_horizon (float): time in seconds

    Returns:
        np.ndarray: center of long box generated by assumption
    """
    orientation = np.array([np.cos(state.yaw), np.sin(state.yaw)])
    return np.array([state.x, state.y]) + time_horizon / 2 * state.speed * orientation


def compute_time_to_collision_by_state(ego_vehicle: EvaluationVehicle,
                                       other_vehicles: List[EvaluationVehicle],
                                       delta_t: float,
                                       threshold: float) -> np.ndarray:
    """compute time-to-collision for all states of ego

    Args:
        ego_vehicle (EvaluationVehicle): ego

        other_vehicles (List[EvaluationVehicle]): other vehicles in AoI

        delta_t (float): frequency

        threshold (float): time horizon in consideration

    Returns:
        np.ndarray: (n, 2)
        an array of same length as ego states, indicates time-to-collision in seconds


    Reference:
        https://github.com/motional/nuplan-devkit
    """
    # 1. exclude vehicles that is not possible to collide with ego
    vehicles = extract_vehicles_excluding_impossible_vehicles(
        ego_vehicle, other_vehicles)

    result: np.ndarray = np.ones((len(ego_vehicle.states), 2)) * threshold
    result[:, 0] = np.fromiter(ego_vehicle.states.keys(), dtype=float)
    for index, (timestamp, ego_state) in enumerate(ego_vehicle.states.items()):
        # no vehicles detected
        if vehicles[index].size == 0:
            result[index, 1] = threshold
            continue

        ego_position = np.array([ego_state.x, ego_state.y])
        ego_delta_position = ego_state.speed * delta_t * np.array(
            [np.cos(ego_state.yaw), np.sin(ego_state.yaw)])

        ego_box = Rectangle(center=ego_position,
                            width=ego_vehicle.model.width,
                            length=ego_vehicle.model.length,
                            yaw=ego_state.yaw)
        # use a big, long box to represent the trajectory of ego (const speed and yaw)
        # its center is the median of start position and end position
        ego_long_box = Rectangle(
            center=get_long_box_center(ego_state, threshold),
            width=ego_vehicle.model.width,
            length=ego_vehicle.model.length + ego_state.speed * threshold,
            yaw=ego_state.yaw)

        # 2. use nuplan way to create Big Boxes to replace other vehicles
        vehicle_long_boxes = [
            Rectangle(center=get_long_box_center(vehicle.states[timestamp],
                                                 threshold),
                      width=vehicle.model.width,
                      length=vehicle.model.length +
                      vehicle.states[timestamp].speed * threshold,
                      yaw=vehicle.states[timestamp].yaw)
            for vehicle in vehicles[index]
        ]

        # 3. find possible collisions
        possible_vehicles_mask = np.where([
            ego_long_box.in_collision(vehicle_long_box)
            for vehicle_long_box in vehicle_long_boxes
        ])[0]

        # no possible collision vehicles
        if possible_vehicles_mask.size == 0:
            result[index, 1] = threshold
            continue

        vehicle_boxes = [
            Rectangle(center=np.array(
                [vehicle.states[timestamp].x, vehicle.states[timestamp].y]),
                      width=vehicle.model.width,
                      length=vehicle.model.length,
                      yaw=vehicle.states[timestamp].yaw)
            for vehicle in vehicles[index][possible_vehicles_mask]
        ]

        vehicle_delta_pos = [
            vehicle.states[timestamp].speed * delta_t * np.array([
                np.cos(vehicle.states[timestamp].yaw),
                np.sin(vehicle.states[timestamp].yaw)
            ]) for vehicle in vehicles[index][possible_vehicles_mask]
        ]

        # 3. check collision by incrementing time step
        ttc_found: bool = False
        for t in np.arange(0, threshold, delta_t):
            ego_box.center += ego_delta_position
            for i, vehicle_box in enumerate(vehicle_boxes):
                vehicle_box.center += vehicle_delta_pos[i]
                if ego_box.in_collision(vehicle_box):
                    result[index, 1] = t
                    ttc_found = True
                    break

            if ttc_found:
                break

    return result


def compute_time_to_collision(database_name: str) -> np.ndarray:
    """compute time-to-collision information from a given scenario database

    Args:
        database_name (str): a SQL database contains scenario information

    Returns:
        np.ndarray: (n, 2)
        an array of same length as ego states, indicates time-to-collision in seconds
    """
    # constant
    delta_t = 0.1  # (s)
    threshold = 20.0  # (s)

    frame_df, vehicle_df = get_data_frame(database_name)
    ego_vehicle, other_vehicles = extract_vehicles(frame_df, vehicle_df)
    result = compute_time_to_collision_by_state(ego_vehicle, other_vehicles,
                                                delta_t, threshold)

    return result
