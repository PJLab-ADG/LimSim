from __future__ import annotations
import dearpygui.dearpygui as dpg
import sqlite3
from rich import print
import time
import numpy as np
from datetime import datetime
import os
import threading
from queue import Queue

from simModel.common.gui import GUI
from simModel.common.networkBuild import Rebuild
from simModel.common.carFactory import Vehicle, egoCar
from simModel.egoTracking.movingScene import SceneReplay
from utils.trajectory import Trajectory, State, Rectangle, RecCollide
from utils.simBase import vehType, MapCoordTF
from evaluation.evaluation import RealTimeEvaluation
from typing import List
from math import sin, cos, pi


class InterReplayModel:
    def __init__(self,
                 dataBase: str,
                 dataBase2: str = None,
                 startFrame: int = None,
                 simNote: str = ''
                 ) -> None:
        print('[green bold]Model initialized at {}.[/green bold]'.format(
            datetime.now().strftime('%H:%M:%S.%f')[:-3]
        ))
        self.sim_mode: str = 'InterReplay'
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
                    '[yellow]The start frame is too large, and is reassigned to[/yellow] %i.' % maxTimeStep)
                self.timeStep = maxTimeStep
            elif startFrame < minTimeStep:
                print(
                    '[yellow]The start frame is too small, and is reassigned to[/yellow] %i.' % minTimeStep)
                self.timeStep = minTimeStep
            else:
                self.timeStep = startFrame
        else:
            self.timeStep = minTimeStep

        self.startFrame = self.timeStep

        # tpEnd marks whether the trajectory planning is end,
        # when the ego car leaves the network, tpEnd turns into 1.
        self.tpEnd = 0

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

        if dataBase2:
            self.dataBase2 = dataBase2
        else:
            self.dataBase2 = datetime.strftime(
                datetime.now(), '%Y-%m-%d_%H-%M-%S') + '_egoTracking_ir' + '.db'

        self.createDatabase2()
        self.simDescriptionCommit(simNote)
        self.databaseMigration()
        self.dataQue = Queue()
        self.createTimer()

        self.sr = SceneReplay(self.rb, self.ego)

        self.evaluation = RealTimeEvaluation(dt=0.1)

        self.gui = GUI('real-time-ego')
        self.gui.start()
        self.drawMapBG()
        self.drawRadarBG()
        self.frameIncre = 0

        self.allvTypes = {}
        self.getAllvType()

    def createDatabase2(self):
        # if database exist then delete it
        if os.path.exists(self.dataBase2):
            os.remove(self.dataBase2)
        conn = sqlite3.connect(self.dataBase2)
        cur = conn.cursor()

        cur.execute('''CREATE TABLE IF NOT EXISTS simINFO(
                        startTime TIMESTAMP PRIMARY KEY,
                        localPosx REAL,
                        localPosy REAL,
                        radius REAL,
                        egoID TEXT,
                        netBoundary TEXT,
                        description TEXT,
                        note TEXT);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS frameINFO(
                            frame INT NOT NULL,
                            vid TEXT NOT NULL,
                            vtag TEXT NOT NULL,
                            x REAL NOT NULL,
                            y REAL NOT NULL,
                            yaw REAL NOT NULL,
                            speed REAL NOT NULL,
                            accel REAL NOT NULL,
                            laneID TEXT NOT NULL,
                            lanePos REAL NOT NULL,
                            routeIdx INT NOT NULL,
                            PRIMARY KEY (frame, vid));''')

        conn.commit()
        cur.close()
        conn.close()

    def simDescriptionCommit(self, simNote: str):
        netBoundary = '{},{} {},{}'.format(
            self.netBoundary[0][0], self.netBoundary[0][1],
            self.netBoundary[1][0], self.netBoundary[1][1]
        )
        currTime = datetime.now()
        insertQuery = '''INSERT INTO simINFO VALUES (?, ?, ?, ?, ?, ?, ?, ?);'''
        conn = sqlite3.connect(self.dataBase2)
        cur = conn.cursor()
        cur.execute(
            insertQuery,
            (currTime, None, None, None,
             self.ego.id, netBoundary, 'ego track interactive replay', simNote))

        conn.commit()
        cur.close()
        conn.close()

    def databaseMigration(self):
        tables = [
            'vehicleINFO',
            'edgeINFO',
            'laneINFO',
            'junctionLaneINFO',
            'junctionINFO',
            'tlLogicINFO',
            'connectionINFO',
            'trafficLightStates',
            'circleObsINFO',
            'rectangleObsINFO',
            'geohashINFO',
            'evaluationINFO'
        ]
        for t in tables:
            os.system('sqlite3 {} ".dump {}" | sqlite3 {}'.format(
                self.dataBase, t, self.dataBase2
            ))
        print('[green bold]Database migration finished.[/green bold]')

    def createTimer(self):
        if not self.tpEnd or not self.dataQue.empty():
            t = threading.Timer(1, self.dataStore)
            t.daemon = True
            t.start()

    def dataStore(self):
        # stime = time.time()
        cnt = 0
        conn = sqlite3.connect(self.dataBase2, check_same_thread=False)
        cur = conn.cursor()
        while cnt < 1000 and not self.dataQue.empty():
            tableName, data = self.dataQue.get()
            sql = 'INSERT INTO %s VALUES ' % tableName + \
                '(' + '?,'*(len(data)-1) + '?' + ')'
            try:
                cur.execute(sql, data)
            except Exception as e:
                print(sql, data)
                raise e
            cnt += 1

        conn.commit()
        cur.close()
        conn.close()

        self.createTimer()

    def putFrameInfo(self, vid: str, vtag: str, veh: Vehicle):
        self.dataQue.put(
            ('frameINFO',
             (self.timeStep, vid, vtag, veh.x, veh.y, veh.yaw, veh.speed,
              veh.accel, veh.laneID, veh.lanePos, veh.routeIdxQ[-1])))

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

    def getAllvType(self):
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(
            """SELECT DISTINCT vTypeID, length, width, maxAccel, maxDecel, maxSpeed
        from vehicleINFO;""")

        dbvTypes = cur.fetchall()
        for dbvType in dbvTypes:
            vTypeIns = vehType(dbvType[0])
            vTypeIns.length = dbvType[1]
            vTypeIns.width = dbvType[2]
            vTypeIns.maxAccel = dbvType[3]
            vTypeIns.maxDecel = dbvType[4]
            vTypeIns.maxSpeed = dbvType[5]
            self.allvTypes[dbvType[0]] = vTypeIns

        cur.close()
        conn.close()

    def dbTrajectory(self, vehid: str, currFrame: int) -> dict:
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(
            """SELECT frame, x, y, yaw, speed, accel, laneID, lanePos, routeIdx FROM frameINFO 
            WHERE vid = '{}' AND frame >= {} AND frame < {};""".format(
                vehid, currFrame, currFrame + 50
            )
        )
        frameData = cur.fetchall()
        if frameData:
            # if the trajectory is segmented in time, only the data of the first
            # segment will be taken.
            validSeq = [frameData[0]]
            for i in range(len(frameData)-1):
                if frameData[i+1][0] - frameData[i][0] == 1:
                    validSeq.append(frameData[i+1])

            tState = []
            for vs in validSeq:
                state = State(
                    x=vs[1], y=vs[2], yaw=vs[3], vel=vs[4],
                    acc=vs[5], laneID=vs[6], s=vs[7], routeIdx=vs[8]
                )
                tState.append(state)
            dbTrajectory = Trajectory(states=tState)
        else:
            if vehid not in self.sr.vehINAoI.keys():
                self.sr.outOfRange.add(vehid)
            return

        cur.close()
        conn.close()
        return dbTrajectory

    def dbVType(self, vid: str):
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute("""SELECT length, width, 
            maxAccel, maxDecel, maxSpeed, vTypeID, routes FROM vehicleINFO 
            WHERE vid = '%s';""" % vid)

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
        veh.LLRSet, veh.LLRDict, veh.LCRDict = veh.getLaneLevelRoute(self.rb)

        return veh

    def isInvolved(self, veh: Vehicle, currVehs: dict[str, Vehicle]) -> bool:
        # if the vehicle's dbTrajectory is too short, limsim will take over it
        # until the vehicle drive out of the AoI, avoiding the vhicles's suddenly fading.
        if len(veh.dbTrajectory.states) <= 10:
            return True
        else:
            if self.VTCollisionCheck(veh, self.ego):
                return True
            for cv in currVehs.values():
                if self.VTCollisionCheck(veh, cv):
                    return True
            return False

    def setTrajectories(self, trajectories: dict[str, Trajectory]):
        if self.ego.id in trajectories.keys():
            self.ego.plannedTrajectory = trajectories[self.egoID]
        dbTrajectory = self.dbTrajectory(self.egoID, self.timeStep)
        self.ego.dbTrajectory = dbTrajectory
        for vid, vins in self.sr.vehINAoI.items():

            dbTrajectory = self.dbTrajectory(vid, self.timeStep)
            if vid in trajectories.keys():
                if not vins.dbTrajectory:
                    vins.plannedTrajectory = trajectories[vid]
                    vins.dbTrajectory = None
                else:
                    if self.isInvolved(vins, self.sr.vehINAoI):
                        test_involve = 1
                        # set planned trajectory
                        vins.plannedTrajectory = trajectories[vid]
                        # if vehicle is controled by the planner,
                        # we won't update the dbTrajectory.
                        vins.dbTrajectory = None
                    else:
                        vins.dbTrajectory = dbTrajectory
            else:
                # set db trajectory
                if not vins.iscontroled:
                    # if the dbTrajectory if None, the follow process will
                    # destroy the vehicle.
                    vins.dbTrajectory = dbTrajectory
                else:
                    # if the vehicle is controlled already, the dbTrajectory
                    # will not update anymore. after it runs out of its
                    # plannedTrajectory, it will be deleted.
                    continue

    def updateVeh(self, veh: Vehicle | egoCar):
        message = veh.replayUpdate()
        if message == 'Failure':
            self.sr.outOfRange.add(veh.id)

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
                vx = list(range(-len(vy)+1, 1))
            dpg.set_value('v_series_tag', [vx, vy])

        if self.ego.accelQ:
            if len(self.ego.accelQ) >= 50:
                ax = list(range(-49, 1))
                ay = list(self.ego.accelQ)[-50:]
            else:
                ay = list(self.ego.accelQ)
                ax = list(range(-len(ay)+1, 1))
            dpg.set_value('a_series_tag', [ax, ay])

        if self.ego.plannedTrajectory:
            if self.ego.plannedTrajectory.velQueue:
                vfy = list(self.ego.plannedTrajectory.velQueue)
                vfy = list(self.ego.plannedTrajectory.velQueue)
                vfx = list(range(1, len(vfy)+1))
                dpg.set_value('v_series_tag_future', [vfx, vfy])
            if self.ego.plannedTrajectory.accQueue:
                afy = list(self.ego.plannedTrajectory.accQueue)
                afx = list(range(1, len(afy)+1))
                dpg.set_value('a_series_tag_future', [afx, afy])
        else:
            if self.ego.dbTrajectory:
                if self.ego.dbTrajectory.velQueue:
                    vfy = list(self.ego.dbTrajectory.velQueue)
                    vfx = list(range(1, len(vfy)+1))
                    dpg.set_value('v_series_tag_future', [vfx, vfy])
                if self.ego.dbTrajectory.accQueue:
                    afy = list(self.ego.dbTrajectory.accQueue)
                    afx = list(range(1, len(afy)+1))
                    dpg.set_value('a_series_tag_future', [afx, afy])

    def drawSce(self):
        node = dpg.add_draw_node(parent="Canvas")
        ex, ey = self.ego.x, self.ego.y

        self.sr.plotScene(node, ex, ey, self.gui.ctf)

        self.ego.plotdeArea(node, ex, ey, self.gui.ctf)
        self.ego.plotSelf('ego', node, ex, ey, self.gui.ctf)
        if self.ego.plannedTrajectory:
            self.ego.plotTrajectory(node, ex, ey, self.gui.ctf)
            self.ego.plotDBTrajectory(node, ex, ey, self.gui.ctf)
        self.putFrameInfo(self.ego.id, 'ego', self.ego)
        # else:
        #     self.ego.plotDBTrajectory(node, ex, ey)
        if self.sr.vehINAoI:
            for vAoI in self.sr.vehINAoI.values():
                vAoI.plotSelf('AoI', node, ex, ey, self.gui.ctf)
                self.putFrameInfo(vAoI.id, 'AoI', vAoI)
                if vAoI.plannedTrajectory:
                    vAoI.plotTrajectory(node, ex, ey, self.gui.ctf)
                    continue
                if vAoI.dbTrajectory:
                    vAoI.plotDBTrajectory(node, ex, ey, self.gui.ctf)
                #     vAoI.plotDBTrajectory(node, ex, ey)
                # else:
                #     vAoI.plotDBTrajectory(node, ex, ey)
        if self.sr.outOfAoI:
            for vSce in self.sr.outOfAoI.values():
                vSce.plotSelf('Sce', node, ex, ey, self.gui.ctf)
                self.putFrameInfo(vSce.id, 'outOfAoI', vSce)
                # vSce.plotTrajectory(node, ex, ey)

        mvNode = dpg.add_draw_node(parent='movingScene')
        mvCenterx, mvCentery = self.mapCoordTF.dpgCoord(ex, ey)
        dpg.draw_circle((mvCenterx, mvCentery),
                        self.ego.deArea * self.mapCoordTF.zoomScale,
                        color=(37, 204, 247, 20),
                        fill=(37, 204, 247, 200),
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

    def getNextFrameVehs(self):
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute("""SELECT DISTINCT vid FROM frameINFO 
            WHERE frame = {};""".format(self.timeStep))
        nextFrameVehs = cur.fetchall()
        nextFrameVehs = [query[0] for query in nextFrameVehs]
        cur.close()
        conn.close()
        return nextFrameVehs

    # vehicle trajectory collision check
    # seprate axis theorem
    def VTCollisionCheck(self, veh1: Vehicle, veh2: Vehicle) -> bool:
        tjA = veh1.dbTrajectory
        if veh2.plannedTrajectory:
            tjB = veh2.plannedTrajectory
        else:
            # when veh2 doesn't have planned trajectory, it will drive according
            # to the database, so veh1 and veh2 won't collide.
            return False
        duration = min(len(tjA.states), len(tjB.states))
        for i in range(0, duration, 3):
            stateA = tjA.states[i]
            stateB = tjB.states[i]
            recA = Rectangle([stateA.x, stateA.y],
                             veh1.length, veh1.width, stateA.yaw)
            recB = Rectangle([stateB.x, stateB.y],
                             veh2.length, veh2.width, stateB.yaw)
            rc = RecCollide(recA, recB)
            if rc.isCollide():
                return True
        return False

    # new vehicles
    def newVehCollisionCheck(
            self, veh: Vehicle, currVehs: dict[str, Vehicle]) -> bool:
        for cv in currVehs.values():
            if self.VTCollisionCheck(veh, cv):
                return True
        return False

    def getSce(self):
        nextFrameVehs = self.getNextFrameVehs()
        if nextFrameVehs:
            dpg.delete_item("Canvas", children_only=True)
            dpg.delete_item("movingScene", children_only=True)
            dpg.delete_item("simInfo", children_only=True)
            dpg.delete_item("radarPlot", children_only=True)
            self.updateVeh(self.ego)
            if self.ego.arriveDestination(self.rb):
                self.tpEnd = 1
            for veh in self.sr.currVehicles.values():
                self.updateVeh(veh)
            newVehs = nextFrameVehs - self.sr.currVehicles.keys()
            for nv in newVehs:
                if nv != self.egoID:
                    veh = self.initVeh(nv, self.timeStep)
                    # if veh.id == '107':
                    #     print(veh.dbTrajectory)
                    self.updateVeh(veh)
                    if not self.newVehCollisionCheck(veh, self.sr.vehINAoI):
                        self.sr.currVehicles[nv] = veh
                    # self.sr.currVehicles[nv] = veh
            self.sr.updateSurroudVeh()

        else:
            if not self.tpEnd:
                print('The ego car has reached the destination.')
                self.tpEnd = 1
                self.gui.is_running = 0

    def exportSce(self):
        return self.sr.exportScene()

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

    def render(self):
        self.gui.update_inertial_zoom()
        dpg.delete_item('Canvas', children_only=True)
        self.sr.updateScene(self.dataBase, self.timeStep)
        self.drawSce()
        self.plotVState()
        dpg.render_dearpygui_frame()
        if self.gui.replayDelay:
            time.sleep(self.gui.replayDelay)

    def moveStep(self):
        if self.gui.is_running:
            self.timeStep += 1
            self.getSce()

        if not dpg.is_dearpygui_running():
            self.tpEnd = 1
        self.render()
