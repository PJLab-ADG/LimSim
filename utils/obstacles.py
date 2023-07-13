from abc import ABC
from enum import IntEnum
import dearpygui.dearpygui as dpg
import numpy as np

from trafficManager.common.coord_conversion import cartesian_to_frenet2D
from trajectory import State, Trajectory
from simBase import CoordTF
from separate_axis_theorem import separate_axis_theorem


class Shape(ABC):
    """
    The base class for all shapes.
    """
    pass


class Rectangle(Shape):
    """
    A class representing a rectangle shape.
    """

    def __init__(self, length: float, width: float, yaw: float = 0.0) -> None:
        super().__init__()
        self._length: float = length
        self._width: float = width
        self._yaw: float = yaw

    @property
    def length(self) -> float:
        return self._length

    @property
    def width(self) -> float:
        return self._width

    @property
    def yaw(self) -> float:
        return self._yaw

    def get_vertexes(self, center: np.ndarray) -> np.ndarray:
        points = np.array([[self.length, self.width],
                           [-self.length, self.width],
                           [-self.length, -self.width],
                           [self.length, -self.width]]) / 2
        rotation = np.array([[np.cos(self.yaw), -np.sin(self.yaw)],
                             [np.sin(self.yaw), np.cos(self.yaw)]])
        return center + np.array([rotation.dot(point) for point in points])

    def in_collision(self, self_center: np.ndarray,
                     other_rectangle: 'Rectangle',
                     other_center: np.ndarray) -> bool:
        """check if two rectangles intersects (in collision)

        Args:
            other_rectangle (Rectangle): another rectangle

        Returns:
            bool: True if two rectangle intersects, False otherwise
        """
        self_vertexes = self.get_vertexes(self_center)
        other_vertexes = other_rectangle.get_vertexes(other_center)

        # first use AABB to filter impossible cases
        self_x_max, self_y_max = np.max(self_vertexes, axis=0)
        self_x_min, self_y_min = np.min(self_vertexes, axis=0)
        other_x_max, other_y_max = np.max(other_vertexes, axis=0)
        other_x_min, other_y_min = np.min(other_vertexes, axis=0)

        if self_x_max < other_x_min or self_x_min > other_x_max or \
           self_y_max < other_y_min or self_y_min > other_y_max:
            return False

        # use separate axis theorem check in-collision or not
        return separate_axis_theorem(self_vertexes, other_vertexes)

    def plotSelf(self, node: dpg.node, center: np.ndarray, ex: float,
                 ey: float):
        """
        Plots the rectangle shape.

        Args:
            node: The node to plot the shape on.
            ex: The x-coordinate of the observer.
            ey: The y-coordinate of the observer.
        """
        vertexes = self.get_vertexes(center)
        relativeVex = [[vertex[0] - ex, vertex[1] - ey] for vertex in vertexes]
        drawVex = [[
            CoordTF.zoomScale * (CoordTF.drawCenter + rev[0]),
            CoordTF.zoomScale * (CoordTF.drawCenter - rev[1])
        ] for rev in relativeVex]
        drawVex.append(drawVex[0])
        dpg.draw_polygon(drawVex,
                         color=(235, 47, 6),
                         fill=(229, 80, 57, 20),
                         parent=node)


class Circle(Shape):
    """
    A class representing a circle shape.
    """

    def __init__(self, radius: float) -> None:
        super().__init__()
        self._radius: float = radius

    @property
    def radius(self) -> float:
        return self._radius

    def plotSelf(self, node: dpg.node, center: np.ndarray, ex: float,
                 ey: float, ctf: CoordTF):
        """
        Plots the circle shape.

        Args:
            node: The node to plot the shape on.
            ex: The x-coordinate of the observer.
            ey: The y-coordinate of the observer.
        """
        cx, cy = ctf.dpgCoord(center[0], center[1], ex, ey)
        dpg.draw_circle((cx, cy),
                        CoordTF.zoomScale * self.radius,
                        color=(235, 47, 6),
                        fill=(229, 80, 57, 20),
                        parent=node)


class ObsType(IntEnum):
    """
    An enumeration representing the type of obstacle.
    """
    PARKED_VEHICLE = 0
    CONSTRUCTION_ZONE = 1
    PEDESTRIAN = 2
    BICYCLE = 3
    CAR = 4
    OTHER = 100


class Obstacle(ABC):
    """
    A class representing an obstacle.
    """

    def __init__(self,
                 obstacle_id: str,
                 shape: Shape,
                 obstacle_type: ObsType,
                 current_state: State,
                 lane_id: str,
                 edge: str = "") -> None:
        super().__init__()

        self._obstacle_id = obstacle_id
        self._shape: Shape = shape
        self._obstacle_type: ObsType = obstacle_type
        self._current_state: State = current_state
        self._lane_id: str = lane_id
        self._affiliated_edge: str = edge

    @property
    def type(self) -> ObsType:
        return self._obstacle_type

    @property
    def current_state(self) -> State:
        return self._current_state

    @property
    def shape(self) -> Shape:
        return self._shape

    @property
    def lane_id(self) -> str:
        return self._lane_id
    
    def update_frenet_coord_in_lane(self, lane) -> State:
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


    @classmethod
    def collision_check(cls, obs1, obs2) -> bool:
        """
        Checks if two obstacles collide.

        Args:
            obs1: The first obstacle.
            obs2: The second obstacle.

        Returns:
            True if the obstacles collide, False otherwise.
        """
        pass


class StaticObstacle(Obstacle):
    """
    A class representing a static obstacle.
    """

    def __init__(self,
                 obstacle_id: str,
                 shape: Shape,
                 obstacle_type: ObsType,
                 current_state: State,
                 lane_id: str,
                 edge: str = "") -> None:
        super().__init__(obstacle_id, shape, obstacle_type, current_state,
                         lane_id, edge)


class DynamicObstacle(Obstacle):
    """
    A class representing a dynamic obstacle.
    """

    def __init__(self,
                 obstacle_id: str,
                 shape: Shape,
                 obstacle_type: ObsType,
                 current_state: State,
                 lane_id: str,
                 future_trajectory: Trajectory = None,
                 edge: str = "") -> None:
        super().__init__(obstacle_id, shape, obstacle_type, current_state,
                         lane_id, edge)

        self._future_trajectory = future_trajectory if future_trajectory is not None else Trajectory()
        


    @property
    def future_trajectory(self) -> Trajectory:
        return self._future_trajectory
