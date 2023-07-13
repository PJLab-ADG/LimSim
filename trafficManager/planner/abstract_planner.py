'''
Author: Licheng Wen
Date: 2022-11-17 17:28:44
Description: 
Copyright (c) 2022 by PJLab, All Rights Reserved. 
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from common.observation import Observation
from common.vehicle import Vehicle
from decision_maker.abstract_decision_maker import EgoDecision, MultiDecision
from predictor.abstract_predictor import Prediction

from utils.roadgraph import RoadGraph
from utils.trajectory import State, Trajectory


class AbstractEgoPlanner(ABC):
    @abstractmethod
    def plan(
        self,
        observation: Observation,
        road_graph: RoadGraph,
        prediction: Prediction = None,
        ego_decision: EgoDecision = None,
    ) -> Trajectory:
        pass


class AbstractMultiPlanner(ABC):
    @abstractmethod
    def plan(
        self,
        observation: Observation,
        road_graph: RoadGraph,
        prediction: Prediction = None,
        multi_decision: MultiDecision = None,
    ) -> Dict[Vehicle, Trajectory]:
        pass
