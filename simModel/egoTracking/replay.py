from __future__ import annotations
import dearpygui.dearpygui as dpg
import sqlite3
from rich import print
import time
import numpy as np
from datetime import datetime
from typing import List
from math import sin, cos, pi

from simModel.common.networkBuild import Rebuild
from simModel.common.carFactory import Vehicle, egoCar
from simModel.common.gui import GUI
from simModel.egoTracking.movingScene import SceneReplay
from utils.trajectory import Trajectory, State
from utils.simBase import MapCoordTF
from evaluation.evaluation import RealTimeEvaluation


class ReplayModel:
    '''
        dataBase: Replay database, please note that the database files for ego tracking and fixed scene are not common;
    '''

    def __init__(self, dataBase: str, startFrame: int = None) -> None:
        print(
            '[green bold]Model initialized at {}.[/green bold]'.format(
                datetime.now().strftime('%H:%M:%S.%f')[:-3]
            )
        )
        self.dataBase = dataBase
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()

        # minTimeStep
        cur.execute("""SELECT MAX(frame) FROM frameINFO;""")
        maxTimeStep = cur.fetchone()[0] - 200
        if maxTimeStep < 0:
            maxTimeStep = 0
        cur.execute("""SELECT MIN(frame) FROM frameINFO;""")
        minTimeStep = cur.fetchone()[0]
        if startFrame:
            if startFrame > maxTimeStep:
                print(
                    '[yellow]The start frame is too large, and is reassigned to[/yellow] %i.'
                    % maxTimeStep
                )
                self.timeStep = maxTimeStep
            elif startFrame < minTimeStep:
                print(
                    '[yellow]The start frame is too small, and is reassigned to[/yellow] %i.'
                    % minTimeStep
                )
                self.timeStep = minTimeStep
            else:
                self.timeStep = startFrame
        else:
            self.timeStep = minTimeStep

        self.rb = Rebuild(dataBase)
        self.rb.getData()
        self.rb.buildTopology()

        cur.execute("""SELECT * FROM simINFO;""")
        simINFO = cur.fetchone()
        _, localPosx, localPosy, radius, egoID, strBoundary, _, _ = simINFO
        if egoID:
            self.egoID = egoID
            self.ego = self.initVeh(egoID, self.timeStep)
            netBoundaryList = strBoundary.split(' ')
            self.netBoundary: list[list[float]] = [
                list(map(float, p.split(','))) for p in netBoundaryList
            ]
        else:
            raise TypeError('Please select the appropriate database file.')

        cur.close()
        conn.close()

        self.sr = SceneReplay(self.rb, self.ego)

        self.evaluation = RealTimeEvaluation(dt=0.1)

        self.gui = GUI('replay-ego')
        self.gui.start()
        self.drawMapBG()
        self.drawRadarBG()
        self.frameIncrement = 0

        self.tpEnd = 0

    def _evaluation_transform_coordinate(self, points: List[float],
                                         scale: float) -> List[List[float]]:
        dpgHeight = dpg.get_item_height('sEvaluation') - 30
        dpgWidth = dpg.get_item_width('sEvaluation') - 20
        centerx = dpgWidth / 2
        centery = dpgHeight / 2

        transformed_points = []
        for j in range(5):
            transformed_points.append([
                centerx + scale * points[j] * cos(pi / 10 + 2 * pi * j / 5),
                dpgHeight -
                (centery + scale * points[j] * sin(pi / 10 + 2 * pi * j / 5))
            ])

        return transformed_points

    def drawRadarBG(self):
        bgNode = dpg.add_draw_node(parent='radarBackground')
        # eliminate the bias
        dpgHeight = dpg.get_item_height('sEvaluation') - 30
        dpgWidth = dpg.get_item_width('sEvaluation') - 20
        centerx = dpgWidth / 2
        centery = dpgHeight / 2
        for i in range(4):
            dpg.draw_circle(center=[centerx, centery],
                            radius=30 * (i + 1),
                            color=(223, 230, 233),
                            parent=bgNode)

        radarLabels = [
            "Offset", "Discomfort", "Collision", "Orientation", "Consumption"
        ]
        offset = np.array([[-0.3, 0.2], [-2.2, 0.3], [-2.3, 0.2], [-2.8, 0.5],
                           [-0.1, 0.5]]) * 30

        axis_points = self._evaluation_transform_coordinate([4, 4, 4, 4, 4],
                                                            scale=30)
        text_points = self._evaluation_transform_coordinate([1, 1, 1, 1, 1],
                                                            scale=140)
        for j in range(5):
            dpg.draw_line(
                [centerx, centery],
                axis_points[j],
                color=(223, 230, 233),
                parent=bgNode,
            )

            dpg.draw_text([
                text_points[j][0] + offset[j][0],
                text_points[j][1] - offset[j][1]
            ],
                          text=radarLabels[j],
                          size=20,
                          parent=bgNode)

    def drawMapBG(self):
        # left-bottom: x1, y1
        # top-right: x2, y2
        ((x1, y1), (x2, y2)) = self.netBoundary
        self.mapCoordTF = MapCoordTF((x1, y1), (x2, y2), 'macroMap')
        mNode = dpg.add_draw_node(parent='mapBackground')
        for jid in self.rb.junctions.keys():
            self.rb.plotMapJunction(jid, mNode, self.mapCoordTF)

        self.gui.drawMainWindowWhiteBG((x1, y1), (x2, y2))

    def dbTrajectory(self, vehid: str, currFrame: int) -> Trajectory:
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(
            """SELECT frame, x, y, yaw, speed, accel, laneID, lanePos, routeIdx FROM frameINFO
            WHERE vid = "{}" AND frame >= {} AND frame < {};""".format(
                vehid, currFrame, currFrame + 50
            )
        )
        frameData = cur.fetchall()
        if frameData:
            # if the trajectory is segmented in time, only the
            # data of the first segment will be taken.
            validSeq = [frameData[0]]
            for i in range(len(frameData) - 1):
                if frameData[i + 1][0] - frameData[i][0] == 1:
                    validSeq.append(frameData[i + 1])

            tState = []
            for vs in validSeq:
                state = State(
                    x=vs[1],
                    y=vs[2],
                    yaw=vs[3],
                    vel=vs[4],
                    acc=vs[5],
                    laneID=vs[6],
                    s=vs[7],
                    routeIdx=vs[8],
                )
                tState.append(state)
            dbTrajectory = Trajectory(states=tState)
        else:
            self.sr.outOfRange.add(vehid)
            return

        cur.close()
        conn.close()
        return dbTrajectory

    def dbVType(self, vid: str):
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(
            """SELECT length, width,
            maxAccel, maxDecel, maxSpeed, vTypeID, routes FROM vehicleINFO
            WHERE vid = '%s';"""
            % vid
        )

        vType = cur.fetchall()

        cur.close()
        conn.close()
        return vType[0]

    def initVeh(self, vid: str, currFrame: int) -> Vehicle | egoCar:
        dbTrajectory = self.dbTrajectory(vid, currFrame)
        if vid == self.egoID:
            veh = egoCar(vid)
        else:
            veh = Vehicle(vid)
        veh.dbTrajectory = dbTrajectory

        vType = self.dbVType(vid)
        length, width, maxAccel, maxDecel, maxSpeed, vTypeID, routes = vType
        veh.length = length
        veh.width = width
        veh.maxAccel = maxAccel
        veh.maxDecel = maxDecel
        veh.maxSpeed = maxSpeed
        veh.vTypeID = vTypeID
        veh.routes = routes.split(' ')

        return veh

    def setDBTrajectory(self, veh: Vehicle | egoCar):
        dbTrajectory = self.dbTrajectory(veh.id, self.timeStep)
        if dbTrajectory:
            veh.dbTrajectory = dbTrajectory

    def updateVeh(self, veh: Vehicle | egoCar):
        self.setDBTrajectory(veh)
        if veh.dbTrajectory and veh.dbTrajectory.xQueue:
            (x, y, yaw, speed, accel, laneID, lanePos,
             routeIdx) = veh.dbTrajectory.pop_last_state_r()
            veh.xQ.append(x)
            veh.yQ.append(y)
            veh.yawQ.append(yaw)
            veh.speedQ.append(speed)
            veh.accelQ.append(accel)
            if veh.dbTrajectory.laneIDQueue:
                veh.laneIDQ.append(laneID)
                veh.lanePosQ.append(lanePos)
                veh.routeIdxQ.append(routeIdx)

    def plotVState(self):
        if self.ego.speedQ:
            laneID = self.ego.laneID
            if ':' in laneID:
                lane = self.rb.getJunctionLane(laneID)
            else:
                lane = self.rb.getLane(laneID)
            laneMaxSpeed = lane.speed_limit
            dpg.set_axis_limits('v_y_axis', 0, laneMaxSpeed)
            if len(self.ego.speedQ) >= 50:
                vx = list(range(-49, 1))
                vy = list(self.ego.speedQ)[-50:]
            else:
                vy = list(self.ego.speedQ)
                vx = list(range(-len(vy) + 1, 1))
            dpg.set_value('v_series_tag', [vx, vy])

        if self.ego.accelQ:
            if len(self.ego.accelQ) >= 50:
                ax = list(range(-49, 1))
                ay = list(self.ego.accelQ)[-50:]
            else:
                ay = list(self.ego.accelQ)
                ax = list(range(-len(ay) + 1, 1))
            dpg.set_value('a_series_tag', [ax, ay])

        if self.ego.dbTrajectory:
            if self.ego.dbTrajectory.velQueue:
                vfy = list(self.ego.dbTrajectory.velQueue)
                vfx = list(range(1, len(vfy) + 1))
                dpg.set_value('v_series_tag_future', [vfx, vfy])
            if self.ego.dbTrajectory.accQueue:
                afy = list(self.ego.dbTrajectory.accQueue)
                afx = list(range(1, len(afy) + 1))
                dpg.set_value('a_series_tag_future', [afx, afy])

    def drawSce(self):
        node = dpg.add_draw_node(parent="Canvas")
        ex, ey = self.ego.x, self.ego.y

        self.sr.plotScene(node, ex, ey, self.gui.ctf)

        self.ego.plotdeArea(node, ex, ey, self.gui.ctf)
        self.ego.plotSelf('ego', node, ex, ey, self.gui.ctf)
        self.ego.plotDBTrajectory(node, ex, ey, self.gui.ctf)
        if self.sr.vehINAoI:
            for vAoI in self.sr.vehINAoI.values():
                vAoI.plotSelf('AoI', node, ex, ey, self.gui.ctf)
                vAoI.plotDBTrajectory(node, ex, ey, self.gui.ctf)
        if self.sr.outOfAoI:
            for vSce in self.sr.outOfAoI.values():
                vSce.plotSelf('Sce', node, ex, ey, self.gui.ctf)

        mvNode = dpg.add_draw_node(parent='movingScene')
        mvCenterx, mvCentery = self.mapCoordTF.dpgCoord(ex, ey)
        dpg.draw_circle((mvCenterx, mvCentery),
                        self.ego.deArea * self.mapCoordTF.zoomScale,
                        thickness=0,
                        fill=(243, 156, 18, 60),
                        parent=mvNode)

        infoNode = dpg.add_draw_node(parent='simInfo')
        dpg.draw_text((5, 5),
                      f'Replay {self.dataBase}',
                      color=(75, 207, 250),
                      size=20,
                      parent=infoNode)
        dpg.draw_text((5, 25),
                      'Time step: %.2f s.' % (self.timeStep / 10),
                      color=(85, 230, 193),
                      size=20,
                      parent=infoNode)
        dpg.draw_text((5, 45),
                      'Current lane: %s' % self.ego.laneID,
                      color=(249, 202, 36),
                      size=20,
                      parent=infoNode)
        dpg.draw_text((5, 65),
                      'Lane position: %.5f' % self.ego.lanePos,
                      color=(249, 202, 36),
                      size=20,
                      parent=infoNode)

        radarNode = dpg.add_draw_node(parent='radarPlot')

        points = self.evaluation.output_result()

        transformed_points = self._evaluation_transform_coordinate(points,
                                                                   scale=30)
        transformed_points.append(transformed_points[0])

        dpg.draw_polygon(transformed_points,
                         color=(75, 207, 250, 100),
                         fill=(75, 207, 250, 100),
                         thickness=5,
                         parent=radarNode)

    def getNextFrameVehs(self):
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(
            """SELECT DISTINCT vid FROM frameINFO
            WHERE frame = {};""".format(
                self.timeStep
            )
        )
        nextFrameVehs = cur.fetchall()
        nextFrameVehs = [query[0] for query in nextFrameVehs]
        cur.close()
        conn.close()
        return nextFrameVehs

    def update_evluation_data(self):
        current_lane = self.rb.getLane(self.ego.laneID)
        if current_lane is None:
            current_lane = self.rb.getJunctionLane(self.ego.laneID)
        agents = [
            agent for agent in self.sr.vehINAoI.values()
            if agent.laneID == current_lane.id
        ]
        self.evaluation.update_data(self.ego, current_lane, agents)

    def getSce(self):
        nextFrameVehs = self.getNextFrameVehs()
        if nextFrameVehs:
            self.updateVeh(self.ego)
            for veh in self.sr.currVehicles.values():
                self.updateVeh(veh)
            newVehs = nextFrameVehs - self.sr.currVehicles.keys()
            for nv in newVehs:
                if nv != self.egoID:
                    veh = self.initVeh(nv, self.timeStep)
                    self.updateVeh(veh)
                    self.sr.currVehicles[nv] = veh
            self.sr.updateSurroudVeh()

            self.update_evluation_data()
        else:
            if not self.tpEnd:
                print('The ego car has reached the destination.')
                self.tpEnd = 1
                self.gui.is_running = 0

    def render(self):
        self.gui.update_inertial_zoom()
        dpg.delete_item('Canvas', children_only=True)
        dpg.delete_item("movingScene", children_only=True)
        dpg.delete_item("simInfo", children_only=True)
        dpg.delete_item("radarPlot", children_only=True)
        self.sr.updateScene(self.dataBase, self.timeStep)
        self.drawSce()
        self.plotVState()
        dpg.render_dearpygui_frame()
        if self.gui.replayDelay:
            time.sleep(self.gui.replayDelay)

    @ property
    def canGetNextSce(self) -> int:
        if self.gui.is_running:
            return 1  # just move steps
        else:
            if self.frameIncrement < self.gui.frameIncre:
                return 2  # move only one step
            else:
                return 0

    def moveStep(self):
        if self.canGetNextSce == 1:
            self.timeStep += 1
            self.getSce()
        elif self.canGetNextSce == 2:
            self.timeStep += 1
            self.frameIncrement += 1
            self.getSce()

        if not dpg.is_dearpygui_running():
            self.tpEnd = 1
        self.render()
