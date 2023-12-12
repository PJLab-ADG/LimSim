import multiprocessing
from typing import Tuple, Dict
from utils.roadgraph import RoadGraph

class RenderDataQueue:
    def __init__(self, max_size: int) -> None:
        self.queue = multiprocessing.Manager().list()
        self.max_size = max_size

    def put(self, item: Tuple[Dict, RoadGraph]):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(item)

    def get(self) -> Tuple[Dict, RoadGraph]:
        return self.queue[-1] if self.queue else None
