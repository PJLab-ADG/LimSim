from __future__ import annotations
import dearpygui.dearpygui as dpg
import sqlite3
from rich import print
import time
import io
import cv2
import pickle
import numpy as np
from datetime import datetime
from typing import List
from math import sin, cos, pi

from simModel.NetworkBuild import Rebuild
from simModel.CarFactory import Vehicle, egoCar
from simModel.MovingScene import SceneReplay
from utils.trajectory import Trajectory, State
from simModel.DataQueue import (
    LRD, ERD, JLRD, RGRD, VRD, CameraImages, QuestionAndAnswer
)

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
        _, egoID, netBoundary = simINFO
        if egoID:
            self.egoID = egoID
            self.ego = self.initVeh(egoID, self.timeStep)
            netBoundaryList = netBoundary.split(' ')
            self.netBoundary: list[list[float]] = [
                list(map(float, p.split(','))) for p in netBoundaryList
            ]
        else:
            raise TypeError('Please select the appropriate database file.')

        cur.close()
        conn.close()

        self.sr = SceneReplay(self.rb, self.ego)

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
        if self.ego.id in nextFrameVehs:
            self.updateVeh(self.ego)
            for veh in self.sr.currVehicles.values():
                self.updateVeh(veh)
            newVehs = nextFrameVehs - self.sr.currVehicles.keys()
            for nv in newVehs:
                if nv != self.egoID:
                    veh = self.initVeh(nv, self.timeStep)
                    self.updateVeh(veh)
                    self.sr.currVehicles[nv] = veh
            self.sr.updateScene(self.dataBase, self.timeStep)
            self.sr.updateSurroudVeh()
        else:
            if not self.tpEnd:
                print('[green]The ego car has reached the destination.[/green]')
                self.tpEnd = 1

    def runStep(self):
        if not self.tpEnd:
            self.timeStep += 1
            self.getSce()
        else:
            return
        
    def exportRenderData(self):
        try:
            roadgraphRenderData, VRDDict = self.sr.exportRenderData()
            return roadgraphRenderData, VRDDict
        except TypeError:
            return None, None
        
    def exportImageData(self) -> CameraImages:
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM imageINFO WHERE frame = {};""".format(
                self.timeStep
            )
        )
        data = cur.fetchone()
        conn.close()
        if data:
            return CameraImages(
                pickle.loads(data[1]), pickle.loads(data[2]),
                pickle.loads(data[3]), pickle.loads(data[4]),
                pickle.loads(data[5]), pickle.loads(data[6]),
            )
        else:
            return None
    
    def exportQAData(self) -> QuestionAndAnswer:
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM QAINFO WHERE frame = {};".format(
                self.timeStep
            )
        )
        data = cur.fetchone()
        conn.close()
        if data:
            return QuestionAndAnswer(
                data[1], data[2], data[3], data[4],
                data[5], data[6], data[7], data[8]
            )
        else:
            return None
        
