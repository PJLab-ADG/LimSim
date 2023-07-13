"""
A module for managing observations in a traffic simulation environment.

This module contains the Observation class, which represents an observation of the traffic environment.
It includes information about vehicles, their historical state trajectories, and obstacles in the environment.
"""
from typing import List, Dict
from vehicle import Vehicle
from utils.obstacles import StaticObstacle
from utils.trajectory import State


class Observation:
    """
    A class to represent an observation of the traffic environment.
    
    Attributes:
        vehicles (List[Vehicle]): 
            A list of Vehicle objects in the environment.
        history_track (Dict[int, List[State]]): 
            A dictionary mapping vehicle IDs to their historical state trajectories.
        obstacle (List[List[State]]): 
            A list of lists containing Static obstacles in the environment
    """

    def __init__(self,
                 vehicles: List[Vehicle] = None,
                 history_track: Dict[int, List[State]] = None,
                 static_obstacles: List[StaticObstacle] = None) -> None:
        self.vehicles: List[Vehicle] = vehicles if vehicles is not None else []
        self.history_track: Dict[int,List[State]] = history_track if history_track is not None else {}
        self.obstacles: List[StaticObstacle] = static_obstacles if static_obstacles is not None else []
