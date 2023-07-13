'''
Author: Licheng Wen
Date: 2022-11-17 15:25:35
Description: 
Copyright (c) 2022 by PJLab, All Rights Reserved. 
'''
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from common.observation import Observation
from common.vehicle import Vehicle

from utils.roadgraph import RoadGraph
from utils.trajectory import State


@dataclass(frozen=True)
class Prediction:
    results:  Dict[Vehicle, List[State]] = field(default_factory=dict)


class AbstractPredictor(ABC):
    @abstractmethod
    def predict(
        self, observation: Observation, road_graph: RoadGraph
    ) -> Prediction:
        pass
