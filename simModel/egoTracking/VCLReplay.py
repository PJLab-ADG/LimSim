from __future__ import annotations
import io
import json
import dearpygui.dearpygui as dpg
import sqlite3
from rich import print
import time
import numpy as np
from datetime import datetime
from typing import List
from math import sin, cos, pi
from PIL import Image
import base64

from simModel.common.networkBuild import Rebuild
from simModel.common.carFactory import Vehicle, egoCar
from simModel.common.VCLGUI_R import GUI
from simModel.egoTracking.movingScene import SceneReplay
from utils.trajectory import Trajectory, State

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
        _, localPosx, localPosy, localRadius, egoID, strBoundary, _, _ = simINFO
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

        self.gui = GUI('replay-ego')
        self.gui.start()
        self.gui.drawMainWindowWhiteBG(self.netBoundary[0], self.netBoundary[1])
        self.frameIncrement = 0

        self.tpEnd = 0

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
        else:
            if not self.tpEnd:
                print('The ego car has reached the destination.')
                self.tpEnd = 1
                self.gui.is_running = 0

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


    def showCameraImage(self):
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(
            """SELECT frontView, surroundingView 
            FROM imageINFO where frame={};""".format(self.timeStep)
        )
        data = cur.fetchone()
        conn.close()
        if data:
            frontView, surView = data
            if frontView:
                base64_data = base64.b64decode(frontView)
                image = Image.open(io.BytesIO(base64_data))
                image = np.array(image)
                image = image[:,:,[2,1,0,3]].copy() 
                self.gui.showImage(image)
            else:
                return
        else:
            return
        

    def showPromptsResponse(self):
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(
            """SELECT information, response 
            FROM promptsINFO where frame={};""".format(self.timeStep)
        )
        data = cur.fetchone()
        conn.close()
        if data:
            information, response = data
            self.gui.showInformation(information)
            fullResponse = json.loads(response)
            self.gui.showResponse(fullResponse['choices'][0]['message']['content'])
        else:
            return

    def render(self):
        self.gui.update_inertial_zoom()
        dpg.delete_item("Canvas", children_only=True)
        dpg.delete_item("informationCanvas", children_only=True)
        dpg.delete_item('responseCanvas', children_only=True)
        self.sr.updateScene(self.dataBase, self.timeStep)
        self.drawSce()
        self.showCameraImage()
        self.showPromptsResponse()
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
