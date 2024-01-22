import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union
from PIL import Image

import numpy as np
from rich import print


def resizeImage(
    img: np.ndarray, width: int, height: int
) -> np.ndarray:
    img = Image.fromarray(img)
    img_resized = img.resize((width, height))
    np_img_resized = np.array(img_resized)
    return np_img_resized


@dataclass
class ERD:
    # edge render data
    id: str
    num_lanes: int

@dataclass
class LRD:
    # lane render data
    id: str
    left_bound: List[float] = field(default_factory=list)
    right_bound: List[float] = field(default_factory=list)

    @property
    def edge(self):
        return self.id.rsplit('_', 1)[0]


@dataclass
class JLRD:
    # junction lane render data
    id: str
    center_line: List[float] = field(default_factory=list)
    currTlState: str = None


@dataclass
class RGRD:
    # roadgraph render data
    lanes: Dict[str, LRD] = field(default_factory=dict)
    junction_lanes: Dict[str, JLRD] = field(default_factory=dict)
    edges: Dict[str, ERD] = field(default_factory=dict)

    def get_lane_by_id(self, lid: str) -> Union[LRD, JLRD]:
        if lid[0] == ':':
            return self.junction_lanes[lid]
        else:
            return self.lanes[lid]

    def get_edge_by_id(self, eid: str) -> ERD:
        return self.edges[eid]


@dataclass
class VRD:
    # vehicle render data
    id: str
    x: float
    y: float
    yaw: float
    deArea: float
    length: float
    width: float
    trajectoryXQ: List[float] = field(default_factory=list)
    trajectoryYQ: List[float] = field(default_factory=list)


@dataclass
class CameraImages:
    ORI_CAM_FRONT: np.ndarray
    ORI_CAM_FRONT_RIGHT: np.ndarray
    ORI_CAM_FRONT_LEFT: np.ndarray
    ORI_CAM_BACK_LEFT: np.ndarray
    ORI_CAM_BACK: np.ndarray
    ORI_CAM_BACK_RIGHT: np.ndarray

    def resizeImage(self, width: int, height: int) -> None:
        self.CAM_FRONT = resizeImage(
            self.ORI_CAM_FRONT, width, height)
        self.CAM_FRONT_RIGHT = resizeImage(
            self.ORI_CAM_FRONT_RIGHT, width, height)
        self.CAM_FRONT_LEFT = resizeImage(
            self.ORI_CAM_FRONT_LEFT, width, height)
        self.CAM_BACK_LEFT = resizeImage(
            self.ORI_CAM_BACK_LEFT, width, height)
        self.CAM_BACK = resizeImage(
            self.ORI_CAM_BACK, width, height)
        self.CAM_BACK_RIGHT = resizeImage(
            self.ORI_CAM_BACK_RIGHT, width, height)

@dataclass
class QuestionAndAnswer:
    description: str = ''
    navigation: str = ''
    actions: str = ''   # available actions
    few_shots: str = ''
    response: str = ''
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    choose_action: int = 0


class RenderQueue:
    def __init__(self, max_size: int) -> None:
        self.queue = multiprocessing.Manager().list()
        self.max_size = max_size

    def put(self, item: Tuple[RGRD, Dict[str, List[VRD]]]):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(item)

    def get(self) -> Tuple[RGRD, Dict[str, List[VRD]]]:
        return self.queue[-1] if self.queue else None
    

class ImageQueue:
    def __init__(self, max_size: int) -> None:
        self.queue = multiprocessing.Manager().list()
        self.max_size = max_size

    def put(self, item: CameraImages):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(item)

    def get(self, start_frame: int, steps: int=1) -> List[CameraImages]:
        """
        Return a list of CameraImages objects from the queue.

        Args:
            start_frame (int): The number of frames to retrieve from the queue.
            steps (int): The number of steps to skip between each frame.

        Returns:
            List[CameraImages]: A list of CameraImages objects from the queue. If the queue is empty, return None.
        """
        return self.queue[-start_frame::steps]


class QAQueue:
    def __init__(self, max_size: int) -> None:
        self.queue = multiprocessing.Manager().list()
        self.max_size = max_size

    def put(self, item: QuestionAndAnswer):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(item)

    def get(self) -> QuestionAndAnswer:
        return self.queue[-1] if self.queue else None
