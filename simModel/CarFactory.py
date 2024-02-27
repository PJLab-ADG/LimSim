from __future__ import annotations

from collections import deque
from math import cos, pi, sin
from collections import defaultdict

import dearpygui.dearpygui as dpg
from rich import print
import numpy as np
import traci
from traci import TraCIException

from simModel.NetworkBuild import NetworkBuild, Rebuild
from simModel.DataQueue import VRD
from utils.simBase import CoordTF, deduceEdge
from utils.trajectory import Trajectory


class Vehicle:
    def __init__(self, id: str) -> None:
        self.id = id
        # store the last 10[s] x position for scenario rebuild
        # x, y: position(two doubles) of the named vehicle (center) within the last step
        self.xQ = deque(maxlen=100)
        self.yQ = deque(maxlen=100)
        self.yawQ = deque(maxlen=100)
        self.speedQ = deque(maxlen=100)
        self.accelQ = deque(maxlen=100)
        self.laneIDQ = deque(maxlen=100)
        # lanePos: The position of the vehicle along the lane (the distance
        # from the center of the car to the start of the lane in [m])
        self.lanePosQ = deque(maxlen=100)
        self.routeIdxQ = deque(maxlen=100)
        self.routes: list[str] = None
        self.LLRSet: set[str] = None
        self.LLRDict: dict[str, dict[str, set[str]]] = None
        self.LCRDict: dict[str, str] = None
        self.length: float = 5.0   # SUMO default value
        self.width: float = 1.8   # SUMO default value
        self.maxAccel: float = 3.0   # SUMO default value
        self.maxDecel: float = 4.5   # SUMO default value
        self.maxSpeed: float = 13.89
        self.vTypeID: str = None
        self._iscontroled: bool = 0
        # the follow three parameters are used to obtain available lanes
        self.lookForward: float = 150
        self.noChange: float = 5.0
        self.plannedTrajectory: Trajectory = None
        self.dbTrajectory: Trajectory = None

    # LLR: lane-level route
    def getLaneLevelRoute(self, nb: NetworkBuild) -> tuple[set, dict]:
        LLRSet: set[str] = set()
        LLRDict: dict[str, dict[str, set[str]]] = {}
        for i in range(len(self.routes)-1):
            eid = self.routes[i]
            LLRDict[eid] = {}
            edgeIns = nb.getEdge(eid)
            LLRSet = LLRSet | edgeIns.lanes
            # LLRDict[eid]['edgeLanes'] = edgeIns.lanes
            LLRDict[eid]['edgeLanes'] = set()
            for el in edgeIns.lanes:
                elIns = nb.getLane(el)
                if elIns.width >= self.width:
                    LLRDict[eid]['edgeLanes'].add(el)
            nextEid = self.routes[i+1]
            changeLanes = edgeIns.next_edge_info[nextEid]
            LLRDict[eid]['changeLanes'] = set()
            for cl in changeLanes:
                clIns = nb.getLane(cl)
                if clIns.width >= self.width:
                    LLRDict[eid]['changeLanes'].add(cl)
            nextEdge = nb.getEdge(nextEid)
            changeJuncLanes = set()
            for targetLane in nextEdge.lanes:
                for cl in changeLanes:
                    clIns = nb.getLane(cl)
                    try:
                        changeJuncLanes.add(clIns.next_lanes[targetLane][0])
                    except KeyError:
                        pass

            LLRSet = LLRSet | changeJuncLanes
            LLRDict[eid]['junctionLanes'] = changeJuncLanes

        lastEdgeID = self.routes[-1]
        lastEdgeIns = nb.getEdge(lastEdgeID)
        LLRSet = LLRSet | lastEdgeIns.lanes
        LLRDict[lastEdgeID] = {
            'edgeLanes': lastEdgeIns.lanes
        }

        # Lane corresponded route
        LCRDict: dict[str, list[int]] = defaultdict(list)
        for i in range(len(self.routes)):
            eid = self.routes[i]
            edge = nb.getEdge(eid)
            for lid in edge.lanes:
                LCRDict[lid].append(i)
            nextJunction = nb.getJunction(edge.to_junction)
            for jlid in nextJunction.JunctionLanes:
                LCRDict[jlid].append(i)

        return LLRSet, LLRDict, LCRDict

    @property
    def iscontroled(self):
        return self._iscontroled

    @property
    def yaw(self):
        if self.yawQ:
            return self.yawQ[-1]
        else:
            raise TypeError(f'Vehicle {self.id}: Please call Model.updateVeh() at first.')

    # return center x of the vehicle
    @property
    def x(self):
        if self.xQ:
            return self.xQ[-1]
        else:
            raise TypeError(f'Vehicle {self.id}: Please call Model.updateVeh() at first.')

    # return center y of the vehicle
    @property
    def y(self):
        if self.yQ:
            return self.yQ[-1]
        else:
            raise TypeError(f'Vehicle {self.id}: Please call Model.updateVeh() at first.')

    @property
    def speed(self):
        if self.speedQ:
            return self.speedQ[-1]
        else:
            raise TypeError(f'Vehicle {self.id}: Please call Model.updateVeh() at first.')

    @property
    def accel(self):
        if self.accelQ:
            return self.accelQ[-1]
        else:
            raise TypeError(f'Vehicle {self.id}: Please call Model.updateVeh() at first.')

    @property
    def laneID(self) -> str:
        if self.laneIDQ:
            return self.laneIDQ[-1]
        else:
            raise TypeError(f'Vehicle {self.id}: Please call Model.updateVeh() at first.')

    @property
    def lanePos(self) -> float:
        if self.lanePosQ:
            return self.lanePosQ[-1]
        else:
            raise TypeError(f'Vehicle {self.id}: Please call Model.updateVeh() at first.')

    @property
    def edgeID(self) -> str:
        if self.routeIdxQ:
            currIdx = self.routeIdxQ[-1]
            currEdge = self.routes[currIdx]
            if currIdx < len(self.routes) - 1:
                searchLanes = self.LLRDict[currEdge]['edgeLanes'] | self.LLRDict[currEdge]['junctionLanes']
            else:
                searchLanes = self.LLRDict[currEdge]['edgeLanes']
            if self.laneID in searchLanes:
                return currEdge
            else:
                if currIdx !=0:
                    return self.routes[currIdx-1]
                else:
                    return currEdge
        else:
            raise TypeError(f'Vehicle {self.id}: Please call Model.updateVeh() at first.')

    @property
    def nextEdgeID(self) -> str:
        currIdx = self.routeIdxQ[-1]
        currEdge = self.routes[currIdx]
        if currIdx < len(self.routes) - 1:
            searchLanes = self.LLRDict[currEdge]['edgeLanes'] | self.LLRDict[currEdge]['junctionLanes']
        else:
            searchLanes = self.LLRDict[currEdge]['edgeLanes']
        if self.laneID in searchLanes:
            if currIdx < len(self.routes) - 1:
                return self.routes[currIdx+1]
            else:
                return 'Destination edge'
        else:
            return self.routes[currIdx]

    def arriveDestination(self, nb: NetworkBuild | Rebuild) -> bool:
        nextEdge = self.nextEdgeID
        if nextEdge == 'Destination edge':
            try:
                desLength = nb.getLane(self.laneID).sumo_length
            except AttributeError:
                return False
            if self.lanePos > desLength - 10:
                return True
            else:
                return False
        else:
            return False

    def availableLanes(self, nb: NetworkBuild):
        if ':' in self.laneID:
            if self.nextEdgeID == 'Destination edge':
                return self.LLRDict[self.edgeID]['edgeLanes']
            else:
                output = set()
                output = output | self.LLRDict[self.edgeID]['changeLanes']
                output = output | self.LLRDict[self.edgeID]['junctionLanes']
                output = output | self.LLRDict[self.nextEdgeID]['edgeLanes']
                return output
        else:
            if self.nextEdgeID == 'Destination edge':
                return self.LLRDict[self.edgeID]['edgeLanes']
            laneLength = nb.getLane(self.laneID).sumo_length
            if laneLength < self.lookForward:
                # if self.lanePos < 5:
                #     return self.LLRDict[self.edgeID]['edgeLanes']
                # else:
                output = set()
                output = output | self.LLRDict[self.edgeID]['changeLanes']
                output = output | self.LLRDict[self.edgeID]['junctionLanes']
                return output
            remainDis = laneLength - self.lanePos
            if remainDis > max(laneLength / 3, self.lookForward):
                return self.LLRDict[self.edgeID]['edgeLanes']
            else:
                output = set()
                output = output | self.LLRDict[self.edgeID]['changeLanes']
                output = output | self.LLRDict[self.edgeID]['junctionLanes']
                return output

    # entry control mode and control vehicles
    # used for real-time simulation mode.
    def controlSelf(
        self, centerx: float, centery: float,
        yaw: float, speed: float, accel: float
    ):
        x = centerx + (self.length / 2) * cos(yaw)
        y = centery + (self.length / 2) * sin(yaw)
        # x, y = centerx, centery
        angle = (pi / 2 - yaw) * 180 / pi
        if self._iscontroled:
            traci.vehicle.moveToXY(self.id, '', -1, x, y,
                                   angle=angle, keepRoute=2)
            traci.vehicle.setSpeed(self.id, speed)
            if accel >= 0:
                traci.vehicle.setAccel(self.id, accel)
                traci.vehicle.setDecel(self.id, self.maxDecel)
            else:
                traci.vehicle.setAccel(self.id, self.maxAccel)
                traci.vehicle.setDecel(self.id, -accel)
        else:
            traci.vehicle.setLaneChangeMode(self.id, 0)
            traci.vehicle.setSpeedMode(self.id, 0)
            traci.vehicle.moveToXY(self.id, '', -1, x, y,
                                   angle=angle, keepRoute=2)
            traci.vehicle.setSpeed(self.id, speed)
            if accel >= 0:
                traci.vehicle.setAccel(self.id, accel)
                traci.vehicle.setDecel(self.id, self.maxDecel)
            else:
                traci.vehicle.setAccel(self.id, self.maxAccel)
                traci.vehicle.setDecel(self.id, -accel)
            self._iscontroled = 1

    # exit control mode and set self.iscontroled = 0
    def exitControlMode(self):
        if self._iscontroled:
            try:
                traci.vehicle.setLaneChangeMode(self.id, 0b101010101010)
                traci.vehicle.setSpeedMode(self.id, 0b010111)
                traci.vehicle.setSpeed(self.id, 20)
            except TraCIException:
                pass
            self._iscontroled = 0

    def replayUpdate(self):
        # if plannedTrajectory and dbTrajectory are both empty, return 'Failure',
        # else, return 'Success'.
        if self.plannedTrajectory and self.plannedTrajectory.xQueue:
            x, y, yaw, speed, accel, laneID, lanePos, _ = \
                self.plannedTrajectory.pop_last_state_r()
            self._iscontroled = 1
        elif self.dbTrajectory and self.dbTrajectory.xQueue:
            x, y, yaw, speed, accel, laneID, lanePos, routeIdx = \
                self.dbTrajectory.pop_last_state_r()
        else:
            return 'Failure'
        self.xQ.append(x)
        self.yQ.append(y)
        self.yawQ.append(yaw)
        self.speedQ.append(speed)
        self.accelQ.append(accel)
        self.laneIDQ.append(laneID)
        self.lanePosQ.append(lanePos)
        if ':' not in laneID:
            edge = deduceEdge(laneID)
            self.routeIdxQ.append(self.routes.index(edge))
        else:
            if self.routeIdxQ:
                self.routeIdxQ.append(self.routeIdxQ[-1])
            else:
                self.routeIdxQ.append(routeIdx)
        return 'Success'

    def export2Dict(self, nb: NetworkBuild | Rebuild) -> dict:
        self.LLRSet, self.LLRDict, self.LCRDict = self.getLaneLevelRoute(nb)
        return {
            'id': self.id, 'vTypeID': self.vTypeID,
            'xQ': self.xQ, 'yQ': self.yQ, 'yawQ': self.yawQ,
            'speedQ': self.speedQ, 'accelQ': self.accelQ,
            'laneIDQ': self.laneIDQ, 'lanePosQ': self.lanePosQ,
            'availableLanes': self.availableLanes(nb),
            'routeIdxQ': self.routeIdxQ, 'width': self.width
        }
    
    def exportVRD(self) -> VRD:
        if self.plannedTrajectory and self.plannedTrajectory.xQueue:
            return VRD(
                self.id, self.x, self.y, self.yaw, None,
                self.length, self.width,
                self.plannedTrajectory.xQueue,
                self.plannedTrajectory.yQueue
            )
        elif self.dbTrajectory and self.dbTrajectory.xQueue:
            return VRD(
                self.id, self.x, self.y, self.yaw, None,
                self.length, self.width,
                self.dbTrajectory.xQueue,
                self.dbTrajectory.yQueue
            )
        else:
            return VRD(
                self.id, self.x, self.y, self.yaw, None,
                self.length, self.width,
                None, None
            )

    def plotSelf(self, vtag: str, node: dpg.node, ex: float, ey: float, ctf: CoordTF):
        if not self.xQ:
            raise TypeError('Please call Model.updateVeh() at first.')
        rotateMat = np.array(
            [
                [cos(self.yaw), -sin(self.yaw)],
                [sin(self.yaw), cos(self.yaw)]
            ]
        )
        vertexes = [
            np.array([[self.length/2], [self.width/2]]),
            np.array([[self.length/2], [-self.width/2]]),
            np.array([[-self.length/2], [-self.width/2]]),
            np.array([[-self.length/2], [self.width/2]])
        ]
        rotVertexes = [np.dot(rotateMat, vex) for vex in vertexes]
        relativeVex = [[self.x+rv[0]-ex, self.y+rv[1]-ey]
                       for rv in rotVertexes]
        drawVex = [
            [
                ctf.zoomScale*(ctf.drawCenter+rev[0]+ctf.offset[0]),
                ctf.zoomScale*(ctf.drawCenter-rev[1]+ctf.offset[1])
            ] for rev in relativeVex
        ]
        if vtag == 'ego':
            vcolor = (211, 84, 0)
        elif vtag == 'AoI':
            vcolor = (41, 128, 185)
        else:
            vcolor = (99, 110, 114)
        dpg.draw_polygon(drawVex, color=vcolor, fill=vcolor, parent=node)
        dpg.draw_text(
            ctf.dpgCoord(self.x, self.y, ex, ey),
            self.id,
            color=(0, 0, 0),
            size=20,
            parent=node
        )

    def plotTrajectory(self, node: dpg.node, ex: float, ey: float, ctf: CoordTF):
        if self.plannedTrajectory and self.plannedTrajectory.xQueue:
            tps = [
                ctf.dpgCoord(
                    self.plannedTrajectory.xQueue[i],
                    self.plannedTrajectory.yQueue[i],
                    ex, ey
                ) for i in range(len(self.plannedTrajectory.xQueue))
            ]
            dpg.draw_polyline(tps, color=(205, 132, 241),
                              parent=node, thickness=2)

    def plotDBTrajectory(self, node: dpg.node, ex: float, ey: float, ctf: CoordTF):
        if self.dbTrajectory and self.dbTrajectory.xQueue:
            tps = [
                ctf.dpgCoord(
                    self.dbTrajectory.xQueue[i],
                    self.dbTrajectory.yQueue[i],
                    ex, ey
                ) for i in range(len(self.dbTrajectory.xQueue))
            ]
            dpg.draw_polyline(tps, color=(225, 112, 85),
                              parent=node, thickness=2)

    # append yaw of the car
    def yawAppend(self, angle: float):
        self.yawQ.append((90 - angle) * (pi / 180))

    # append the center x
    def xAppend(self, x: float):
        self.xQ.append(x - self.length / 2 * cos(self.yaw))

    # append the center y
    def yAppend(self, y: float):
        self.yQ.append(y - self.length / 2 * sin(self.yaw))

    # append the lanepos of the center of the car
    def lanePosAppend(self, lanePos: float):
        self.lanePosQ.append(lanePos - self.length / 2)

    def laneAppend(self, nb: NetworkBuild):
        traciLaneID = traci.vehicle.getLaneID(self.id)
        traciLanePos = traci.vehicle.getLanePosition(self.id)
        routeIndex = self.routeIdxQ[-1]
        if routeIndex >= 1:
            currEdge = self.routes[routeIndex]
            lastEdge = self.routes[routeIndex-1]
            edgeLanes = self.LLRDict[currEdge]['edgeLanes'] | \
                self.LLRDict[lastEdge]['edgeLanes']
            try:
                currJunctionLanes = self.LLRDict[currEdge]['junctionLanes']
            except KeyError:
                currJunctionLanes = set()
            lastJunctionLanes = self.LLRDict[lastEdge]['junctionLanes']
            junctionLanes = currJunctionLanes | lastJunctionLanes
        else:
            currEdge = self.routes[routeIndex]
            edgeLanes = self.LLRDict[currEdge]['edgeLanes']
            try:
                junctionLanes = self.LLRDict[currEdge]['junctionLanes']
            except:
                junctionLanes = set()
        searchLanes = edgeLanes | junctionLanes
        if traciLaneID in searchLanes:
            self.laneIDQ.append(traciLaneID)
            self.lanePosQ.append(traciLanePos - self.length / 2)
        else:
            for lid in searchLanes:
                if ':' in lid:
                    laneINS = nb.getJunctionLane(lid)
                else:
                    laneINS = nb.getLane(lid)
                s, d = laneINS.course_spline.cartesian_to_frenet1D(
                    self.x, self.y)
                if abs(d) < 2.0:
                    self.laneIDQ.append(lid)
                    self.lanePosQ.append(s)

    def routeIdxAppend(self, laneID: str):
        curIndexList = self.LCRDict[laneID]
        if self.routeIdxQ:
            lastIndex = self.routeIdxQ[-1]
            for curIndex in curIndexList:
                if curIndex - lastIndex == 0 or curIndex - lastIndex == 1:
                    self.routeIdxQ.append(curIndex)
                    return
        else:
            self.routeIdxQ.append(traci.vehicle.getRouteIndex(self.id))

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, self.__class__):
            return self.id == __o.id
        else:
            raise TypeError('Only class:Vehicle can be added into this set!')

    def __str__(self) -> str:
        return 'ID: {}, x: {:.5f}, y: {:.5f}, yaw: {:.5f}, speed: {:.5f}, accel: {:.5f}, vType: {}'.format(
            self.id, self.x, self.y,
            self.yaw, self.speed, self.accel, self.vTypeID
        )


class egoCar(Vehicle):
    def __init__(
        self, id: str, deArea: float = 50, sceMargin: float = 20
    ) -> None:
        super().__init__(id)
        # detection area
        self.deArea = deArea
        self.sceMargin = sceMargin

    def exportVRD(self) -> VRD:
        if self.plannedTrajectory and self.plannedTrajectory.xQueue:
            return VRD(
                self.id, self.x, self.y, self.yaw, None,
                self.length, self.width,
                self.plannedTrajectory.xQueue,
                self.plannedTrajectory.yQueue
            )
        elif self.dbTrajectory and self.dbTrajectory.xQueue:
            return VRD(
                self.id, self.x, self.y, self.yaw, None,
                self.length, self.width,
                self.dbTrajectory.xQueue,
                self.dbTrajectory.yQueue
            )
        else:
            return VRD(
                self.id, self.x, self.y, self.yaw, None,
                self.length, self.width,
                None, None
            )

    def plotdeArea(self, node: dpg.node, ex: float, ey: float, ctf: CoordTF):
        cx, cy = ctf.dpgCoord(self.x, self.y, ex, ey)
        dpg.draw_circle(
            (cx, cy),
            ctf.zoomScale*self.deArea,
            thickness=0,
            fill=(243, 156, 18, 20),
            parent=node
        )
