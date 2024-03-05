import dearpygui.dearpygui as dpg
from typing import Tuple


class CoordTF:
    # Ego is always in the center of the window
    def __init__(self, realSize: float, windowTag: str) -> None:
        self.realSize: float = realSize
        self.drawCenter: float = self.realSize / 2
        self.dpgDrawSize: float = float(dpg.get_item_height(windowTag)) - 30
        self.offset = (0, 0)

    @property
    def zoomScale(self) -> float:
        return self.dpgDrawSize / self.realSize

    def dpgCoord(
            self, x: float, y: float, ex: float, ey: float) -> Tuple[float]:
        relx, rely = x - ex, y - ey
        return (
            self.zoomScale * (self.drawCenter + relx + self.offset[0]),
            self.zoomScale * (self.drawCenter - rely + self.offset[1])
        )


class MapCoordTF:
    def __init__(
        self, leftBottom: Tuple[float],
        topRight: Tuple[float],
        windowTag: str
    ) -> None:
        self.netLeftBottom = leftBottom
        self.netTopRight = topRight
        self.netHeight = topRight[1] - leftBottom[1]
        self.netWidth = topRight[0] - leftBottom[0]
        self.netCenter = (
            (topRight[0] + leftBottom[0]) / 2,
            (topRight[1] + leftBottom[1]) / 2
        )
        self.dpgSize = dpg.get_item_height(windowTag) - 35

        self.zoomScale = self.dpgSize / max(self.netHeight, self.netWidth)
        self.dpgCenter = (self.dpgSize / 2) / self.zoomScale

    def dpgCoord(self, x: float, y: float) -> Tuple[float]:
        # relativex = x - self.netLeftBottom[0]
        # relativey = y - self.netLeftBottom[1]
        relativex = x - self.netCenter[0]
        relativey = y - self.netCenter[1]
        return (
            (self.dpgCenter + relativex) * self.zoomScale,
            (self.dpgCenter - relativey) * self.zoomScale
        )


def deduceEdge(laneID: str) -> str:
    slist = laneID.split('_')
    del (slist[-1])
    return '_'.join(slist)


class vehType:
    def __init__(self, id: str) -> None:
        self.id = id
        self.maxAccel = None
        self.maxDecel = None
        self.maxSpeed = None
        self.length = None
        self.width = None
        self.vclass = None

    def __str__(self) -> str:
        return 'ID: {},vClass: {}, maxAccel: {}, maxDecel: {:.2f}, maxSpeed: {:.2f}, length: {:.2f}, width: {:.2f}'.format(
            self.id, self.vclass, self.maxAccel, self.maxDecel, self.maxSpeed, self.length, self.width
        )
