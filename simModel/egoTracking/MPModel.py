import os
import sqlite3
import threading
import time
from typing import List, Dict
import xml.etree.ElementTree as ET
from datetime import datetime
from queue import Queue
from math import sin, cos, pi
from simModel.common.RenderDataQueue import RenderDataQueue

import dearpygui.dearpygui as dpg
import numpy as np
import traci
from rich import print
from traci import TraCIException

from simModel.common.carFactory import Vehicle, egoCar
from simModel.egoTracking.movingScene import MovingScene
from simModel.common.networkBuild import NetworkBuild
from utils.trajectory import State, Trajectory
from utils.simBase import MapCoordTF, vehType
from utils.dbBridge import DBBridge
from utils.roadgraph import RoadGraph

from evaluation.evaluation import RealTimeEvaluation

class Model:
    '''
        egoID: id of ego car;
        netFile: network files, e.g. `example.net.xml`;
        rouFile: route files, e.g. `example.rou.xml`. if you have 
                vehicle-type file as an input, define this parameter as 
                `examplevTypes.rou.xml,example.rou.xml`;
        obseFile: obstacle files, e.g. `example.obs.xml`;
        dataBase: the name of the database, e.g. `example.db`. if it is not 
                specified, it will be named with the current timestamp.
        SUMOGUI: boolean variable, used to determine whether to display the SUMO 
                graphical interface;
        simNote: the simulation note information, which can be any information you 
                wish to record. For example, the version of your trajectory 
                planning algorithm, or the user name of this simulation.
    '''
    def __init__(
        self, egoID: str,
        netFile: str, rouFile: str,
        RDQ: RenderDataQueue,
        dataBase: str = None,
        SUMOGUI: int = 0, simNote: str = None,
    ) -> None:
        print('[green bold]Model initialized at {}.[/green bold]'.format(
            datetime.now().strftime('%H:%M:%S.%f')[:-3]))
        self.netFile = netFile
        self.rouFile = rouFile
        self.RDQ = RDQ   # RenderDataQueue
        self.SUMOGUI = SUMOGUI
        self.sim_mode: str = 'RealTime'
        self.timeStep = 0
        # tpStart marks whether the trajectory planning is started,
        # when the ego car appears in the network, tpStart turns into 1.
        self.tpStart = 0
        # tpEnd marks whether the trajectory planning is end,
        # when the ego car leaves the network, tpEnd turns into 1.
        self.tpEnd = 0

        self.ego = egoCar(egoID)

        if dataBase:
            self.dataBase = dataBase
        else:
            self.dataBase = datetime.strftime(
                datetime.now(), '%Y-%m-%d_%H-%M-%S') + '_egoTracking' + '.db'

        if os.path.exists(self.dataBase):
            os.remove(self.dataBase)
        self.dbBridge = DBBridge(self.dataBase)
        self.dbBridge.createTable()
        self.simDescriptionCommit(simNote)
        self.sqliteQue = Queue()
        self.createTimer()

        self.nb = NetworkBuild(self.dataBase, netFile)
        self.nb.getData()
        self.nb.buildTopology()

        self.ms = MovingScene(self.nb, self.ego)

        self.allvTypes = None

        self.evaluation = RealTimeEvaluation(dt=0.1)

    def simDescriptionCommit(self, simNote: str):
        currTime = datetime.now()
        insertQuery = '''INSERT INTO simINFO VALUES (?, ?, ?, ?, ?, ?, ?, ?);'''
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(
            insertQuery,
            (currTime, None, None, None, self.ego.id, '', 'ego track', simNote))

        conn.commit()
        cur.close()
        conn.close()

    def createTimer(self):
        if not self.tpEnd or not self.sqliteQue.empty():
            t = threading.Timer(1, self.dataStore)
            t.daemon = True
            t.start()

    def dataStore(self):
        # stime = time.time()
        cnt = 0
        conn = sqlite3.connect(self.dataBase, check_same_thread=False)
        cur = conn.cursor()
        while cnt < 1000 and not self.sqliteQue.empty():
            tableName, data = self.sqliteQue.get()
            sql = 'INSERT INTO %s VALUES ' % tableName + \
                '(' + '?,'*(len(data)-1) + '?' + ')'
            try:
                cur.execute(sql, data)
            except sqlite3.IntegrityError:
                pass
            cnt += 1

        conn.commit()
        cur.close()
        conn.close()

        self.createTimer()

    # DEFAULT_VEHTYPE
    def getAllvTypeID(self) -> list:
        allvTypesID = []
        if ',' in self.rouFile:
            vTypeFile = self.rouFile.split(',')[0]
            elementTree = ET.parse(vTypeFile)
            root = elementTree.getroot()
            for child in root:
                if child.tag == 'vType':
                    vtid = child.attrib['id']
                    allvTypesID.append(vtid)
        else:
            elementTree = ET.parse(self.rouFile)
            root = elementTree.getroot()
            for child in root:
                if child.tag == 'vType':
                    vtid = child.attrib['id']
                    allvTypesID.append(vtid)

        return allvTypesID

    def start(self):
        traci.start([
            'sumo-gui' if self.SUMOGUI else 'sumo',
            '-n', self.netFile,
            '-r', self.rouFile,
            '--step-length', '0.1',
            '--lateral-resolution', '10',
            '--start', '--quit-on-end',
            '-W', '--collision.action', 'remove'
        ])

        ((x1, y1), (x2, y2)) = traci.simulation.getNetBoundary()
        netBoundary = f"{x1},{y1} {x2},{y2}"
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(f"""UPDATE simINFO SET netBoundary = '{netBoundary}';""")
        conn.commit()
        conn.close()

        allvTypeID = self.getAllvTypeID()
        allvTypes = {}
        if allvTypeID:
            for vtid in allvTypeID:
                vtins = vehType(vtid)
                vtins.maxAccel = traci.vehicletype.getAccel(vtid)
                vtins.maxDecel = traci.vehicletype.getDecel(vtid)
                vtins.maxSpeed = traci.vehicletype.getMaxSpeed(vtid)
                vtins.length = traci.vehicletype.getLength(vtid)
                vtins.width = traci.vehicletype.getWidth(vtid)
                vtins.vclass = traci.vehicletype.getVehicleClass(vtid)
                allvTypes[vtid] = vtins
        else:
            vtid = 'DEFAULT_VEHTYPE'
            vtins = vehType(vtid)
            vtins.maxAccel = traci.vehicletype.getAccel(vtid)
            vtins.maxDecel = traci.vehicletype.getDecel(vtid)
            vtins.maxSpeed = traci.vehicletype.getMaxSpeed(vtid)
            vtins.length = traci.vehicletype.getLength(vtid)
            vtins.width = traci.vehicletype.getWidth(vtid)
            vtins.vclass = traci.vehicletype.getVehicleClass(vtid)
            allvTypes[vtid] = vtins
            self.allvTypes = allvTypes
        self.allvTypes = allvTypes

    def putFrameInfo(self, vid: str, vtag: str, veh: Vehicle):
        self.sqliteQue.put(
            (
                'frameINFO',
                (
                    self.timeStep, vid, vtag, veh.x, veh.y, veh.yaw, veh.speed,
                    veh.accel, veh.laneID, veh.lanePos, veh.routeIdxQ[-1]
                )
            )
        )

    def putVehicleInfo(self, vid: str, vtins: vehType, routes: str):
        self.sqliteQue.put(
            (
                'vehicleINFO', 
                (
                    vid, vtins.length, vtins.width, vtins.maxAccel,
                    vtins.maxDecel, vtins.maxSpeed, vtins.id, routes
                )
            )
        )

    def putVehicleINFO(self):
        self.putFrameInfo(self.ego.id, 'ego', self.ego)
        if self.ms.vehINAoI:
            for v1 in self.ms.vehINAoI.values():
                self.putFrameInfo(v1.id, 'AoI', v1)
        if self.ms.outOfAoI:
            for v2 in self.ms.outOfAoI.values():
                self.putFrameInfo(v2.id, 'outOfAoI', v2)

    def getvTypeIns(self, vtid: str) -> vehType:
        return self.allvTypes[vtid]

    def getVehInfo(self, veh: Vehicle):
        vid = veh.id
        if veh.vTypeID:
            max_decel = veh.maxDecel
        else:
            vtypeid: str = traci.vehicle.getTypeID(vid)
            if '@' in vtypeid:
                vtypeid = vtypeid.split('@')[0]
            vtins = self.getvTypeIns(vtypeid)
            veh.maxAccel = vtins.maxAccel
            veh.maxDecel = vtins.maxDecel
            veh.length = vtins.length
            veh.width = vtins.width
            veh.maxSpeed = vtins.maxSpeed
            # veh.targetCruiseSpeed = random.random()
            veh.vTypeID = vtypeid
            veh.routes = traci.vehicle.getRoute(vid)
            veh.LLRSet, veh.LLRDict, veh.LCRDict = veh.getLaneLevelRoute(
                self.nb)

            routes = ' '.join(veh.routes)
            self.putVehicleInfo(vid, vtins, routes)
            max_decel = veh.maxDecel
        veh.yawAppend(traci.vehicle.getAngle(vid))
        x, y = traci.vehicle.getPosition(vid)
        veh.xAppend(x)
        veh.yAppend(y)
        veh.speedQ.append(traci.vehicle.getSpeed(vid))
        if max_decel == traci.vehicle.getDecel(vid):
            accel = traci.vehicle.getAccel(vid)
        else:
            accel = -traci.vehicle.getDecel(vid)
        veh.accelQ.append(accel)
        laneID = traci.vehicle.getLaneID(vid)
        veh.routeIdxAppend(laneID)
        veh.laneAppend(self.nb)

    def vehMoveStep(self, veh: Vehicle):
        # control vehicles after update its data
        # control happens next timestep
        if veh.plannedTrajectory and veh.plannedTrajectory.xQueue:
            centerx, centery, yaw, speed, accel = veh.plannedTrajectory.pop_last_state(
            )
            try:
                veh.controlSelf(centerx, centery, yaw, speed, accel)
            except:
                return
        else:
            veh.exitControlMode()

    def updateVeh(self):
        self.vehMoveStep(self.ego)
        if self.ms.currVehicles:
            for v in self.ms.currVehicles.values():
                self.vehMoveStep(v)

    def setTrajectories(self, trajectories: Dict[str, Trajectory]):
        for k, v in trajectories.items():
            if k == self.ego.id:
                self.ego.plannedTrajectory = v
            else:
                veh = self.ms.currVehicles[k]
                veh.plannedTrajectory = v

    def getSce(self):
        if self.ego.id in traci.vehicle.getIDList():
            self.tpStart = 1
            self.ms.updateScene(self.sqliteQue, self.timeStep)
            self.ms.updateSurroudVeh()

            self.getVehInfo(self.ego)
            if self.ms.currVehicles:
                for v in self.ms.currVehicles.values():
                    self.getVehInfo(v)

            self.putVehicleINFO()
        else:
            if self.tpStart:
                print('[cyan]The ego car has reached the destination.[/cyan]')
                self.tpEnd = 1

        if self.tpStart:
            if self.ego.arriveDestination(self.nb):
                self.tpEnd = 1
                print('[cyan]The ego car has reached the destination.[/cyan]')

    def putRenderData(self):
        if self.tpStart:
            roadgraphRenderData, VRDDict = self.ms.exportRenderData()
            print('VRDDict exported: ', VRDDict['carInAoI'])
            self.RDQ.put((roadgraphRenderData, VRDDict))

    def exportSce(self):
        if self.tpStart:
            return self.ms.exportScene()
        else:
            return None, None

    def moveStep(self):
        traci.simulationStep()
        self.timeStep += 1
        if self.ego.id in traci.vehicle.getIDList():
            self.getSce()
            self.putRenderData()
            if not self.tpStart:
                self.tpStart = 1
            

    def destroy(self):
        # stop the saveThread.
        time.sleep(1.1)
        traci.close()
