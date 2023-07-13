from __future__ import annotations
import dearpygui.dearpygui as dpg
import sqlite3
from rich import print
import time
from datetime import datetime

from simModel.common.networkBuild import Rebuild
from simModel.common.carFactory import Vehicle, DummyVehicle
from simModel.common.gui import GUI
from simModel.fixedScene.localScene import LocalSceneReplay
from utils.trajectory import Trajectory, State


class ReplayModel:
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

        self.startFrame = self.timeStep

        self.rb = Rebuild(dataBase)
        self.rb.getData()
        self.rb.buildTopology()

        # get initialized ego car with origin position
        cur.execute("""SELECT * FROM simINFO;""")
        simINFO = cur.fetchone()
        _, localPosx, localPosy, radius, egoID, _, _, _ = simINFO
        if radius:
            self.dv = DummyVehicle(localPosx, localPosy, radius)
        else:
            raise TypeError('Please select the appropriate database file.')

        cur.close()
        conn.close()

        self.lsr = LocalSceneReplay(self.rb, self.dv)
        self.gui = GUI('replay-local')
        self.gui.start()
        self.gui.drawMainWindowWhiteBG(
            (self.dv.x-1000, self.dv.y-1000),
            (self.dv.x+1000, self.dv.y+1000)
        )
        self.frameIncrement = 0

        self.tpEnd = 0

    def dbTrajectory(self, vehid: str, currFrame: int) -> Trajectory:
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
            self.lsr.outOfRange.add(vehid)
            return

        cur.close()
        conn.close()
        return dbTrajectory

    def dbVType(self, vid: str):
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(
            """SELECT length, width, maxAccel, maxDecel, maxSpeed, vTypeID, 
            routes FROM vehicleINFO WHERE vid = '%s';"""
            % vid
        )

        vType = cur.fetchall()

        cur.close()
        conn.close()
        return vType[0]

    def initVeh(self, vid: str, currFrame: int) -> Vehicle:
        dbTrajectory = self.dbTrajectory(vid, currFrame)
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

    def setDBTrajectory(self, veh: Vehicle):
        dbTrajectory = self.dbTrajectory(veh.id, self.timeStep)
        if dbTrajectory:
            veh.dbTrajectory = dbTrajectory

    def updateVeh(self, veh: Vehicle):
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

    def drawSce(self):
        node = dpg.add_draw_node(parent="Canvas")
        cx, cy = self.dv.x, self.dv.y

        self.lsr.plotScene(node, cx, cy, self.gui.ctf)
        self.dv.plotArea(node, cx, cy, self.gui.ctf)
        if self.lsr.vehINAoI:
            for vAoI in self.lsr.vehINAoI.values():
                vAoI.plotSelf('AoI', node, cx, cy, self.gui.ctf)
                vAoI.plotDBTrajectory(node, cx, cy, self.gui.ctf)
        if self.lsr.outOfAoI:
            for vSce in self.lsr.outOfAoI.values():
                vSce.plotSelf('Sce', node, cx, cy, self.gui.ctf)

        dpg.draw_text(
            (10, 20),
            'Replay {}'.format(self.dataBase),
            color=(75, 207, 250),
            size=20,
            parent=node,
        )

        dpg.draw_text(
            (10, 50),
            'Time step: %.2f s.' % (self.timeStep / 10),
            color=(85, 230, 193),
            size=20,
            parent=node,
        )

    def getNextFrameVehs(self):
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(
            """SELECT DISTINCT vid FROM frameINFO
            WHERE frame={};""".format(self.timeStep)
        )
        nexFrameVehs = cur.fetchall()
        nexFrameVehs = [query[0] for query in nexFrameVehs]
        cur.close()
        conn.close()
        return nexFrameVehs

    def getSce(self):
        nextFrameVehs = self.getNextFrameVehs()
        if nextFrameVehs:
            for veh in self.lsr.currVehicles.values():
                self.updateVeh(veh)
            newVehs = nextFrameVehs - self.lsr.currVehicles.keys()
            for nv in newVehs:
                veh = self.initVeh(nv, self.timeStep)
                self.updateVeh(veh)
                self.lsr.currVehicles[nv] = veh
            self.lsr.updateSurroundVeh()
        else:
            if not self.tpEnd:
                print('Replay finished.')
                self.tpEnd = 1
                self.gui.is_running = 0

    def render(self):
        self.gui.update_inertial_zoom()
        dpg.delete_item("Canvas", children_only=True)
        self.lsr.updateScene(self.dataBase, self.timeStep)
        self.drawSce()
        dpg.render_dearpygui_frame()
        if self.gui.replayDelay:
            time.sleep(self.gui.replayDelay)

    @property
    def canGetNextSce(self) -> int:
        if self.gui.is_running:
            return 1
        else:
            if self.frameIncrement < self.gui.frameIncre:
                return 2
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
