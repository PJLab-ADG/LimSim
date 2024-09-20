import dearpygui.dearpygui as dpg
from typing import Tuple


class CoordTF:
    def __init__(self, realSize: float, windowTag: str) -> None:
        self.realSize: float = realSize
        self.drawCenter: float = self.realSize / 2
        self.windowTag = windowTag
        self.dpgDrawSize: float = float(dpg.get_item_height(self.windowTag)) - 30
        self.offset = (0, 0)

    @property
    def zoomScale(self) -> float:
        return self.dpgDrawSize / self.realSize

    def dpgDrawSizeInit(self):
        self.dpgDrawSize = float(dpg.get_item_height(self.windowTag)) - 30

    def dpgCoord(self, x: float, y: float, ex: float, ey: float) -> Tuple[float]:
        relx, rely = x - ex, y - ey
        return (
            self.zoomScale * (self.drawCenter + relx + self.offset[0]),  # dpg_x
            self.zoomScale * (self.drawCenter - rely + self.offset[1]),  # dpg_y
        )

    def dpgCoordInverse(
        self, dpg_x: float, dpg_y: float, ex: float, ey: float
    ) -> Tuple[float]:
        relx = dpg_x / self.zoomScale - self.drawCenter - self.offset[0]
        rely = self.drawCenter - dpg_y / self.zoomScale + self.offset[1]
        x = ex + relx
        y = ey + rely
        return (x, y)


class MapCoordTF:
    def __init__(
        self, leftBottom: Tuple[float], topRight: Tuple[float], windowTag: str
    ) -> None:
        self.netLeftBottom = leftBottom
        self.netTopRight = topRight
        self.netHeight = topRight[1] - leftBottom[1]
        self.netWidth = topRight[0] - leftBottom[0]
        self.netCenter = (
            (topRight[0] + leftBottom[0]) / 2,
            (topRight[1] + leftBottom[1]) / 2,
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
            (self.dpgCenter + relativex) * self.zoomScale,  # dpg_x
            (self.dpgCenter - relativey) * self.zoomScale,  # dpg_y
        )
