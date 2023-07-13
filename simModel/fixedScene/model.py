import os
import sqlite3
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from queue import Queue

import dearpygui.dearpygui as dpg
import traci

from simModel.common.carFactory import Vehicle, DummyVehicle
from simModel.common.gui import GUI
from simModel.common.networkBuild import NetworkBuild
from simModel.fixedScene.localScene import LocalScene
from utils.trajectory import Trajectory
from utils.simBase import vehType


class Model:
    '''
        localPos: position of the local area, e.g. (1003.34, 998.63);
        radius: radius of the local are, e.g. 50;
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

    def __init__(self,
                 localPos: tuple[float],
                 radius: float,
                 netFile: str,
                 rouFile: str,
                 obsFile: str = None,
                 dataBase: str = None,
                 SUMOGUI: bool = 1,
                 simNote: str = None
                 ) -> None:
        self.netFile = netFile
        self.rouFile = rouFile
        self.obsFile = obsFile
        self.SUMOGUI = SUMOGUI
        self.sim_mode: str = 'RealTime'
        self.timeStep = 0

        self.dv = DummyVehicle(localPos[0], localPos[1], radius)

        if dataBase:
            self.dataBase = dataBase
        else:
            self.dataBase = datetime.strftime(
                datetime.now(), '%Y-%m-%d_%H-%M-%S'
            ) + '_fixedScene' + '.db'

        self.createDatabase()
        self.simDescriptionCommit(simNote)
        self.dataQue = Queue()
        self.createTimer()

        self.nb = NetworkBuild(self.dataBase, netFile, obsFile)
        self.nb.getData()
        self.nb.buildTopology()
        self.ls = LocalScene(self.nb, self.dv)

        self.allvTypes = None

        self.gui = GUI('real-time-local')

    def createDatabase(self):
        # if database exist then delete it
        if os.path.exists(self.dataBase):
            os.remove(self.dataBase)
        conn = sqlite3.connect(self.dataBase)
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

        cur.execute('''CREATE TABLE IF NOT EXISTS vehicleINFO(
                            vid TEXT PRIMARY KEY,
                            length REAL NOT NULL,
                            width REAL NOT NULL,
                            maxAccel REAL,
                            maxDecel REAL,
                            maxSpeed REAL,
                            vTypeID TEXT NOT NULL,
                            routes TEXT NOT NULL);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS edgeINFO(
                            id TEXT RRIMARY KEY,
                            laneNumber INT NOT NULL,
                            from_junction TEXT,
                            to_junction TEXT);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS laneINFO(
                            id TEXT PRIMARY KEY,
                            rawShape TEXT,
                            width REAL,
                            maxSpeed REAL,
                            edgeID TEXT,
                            length REAL);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS junctionLaneINFO(
                            id TEXT PRIMARY KEY,
                            width REAL,
                            maxSpeed REAL,
                            length REAL,
                            tlLogicID TEXT,
                            tlsIndex INT);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS junctionINFO(
                            id TEXT PRIMARY KEY,
                            rawShape TEXT);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS tlLogicINFO(
                            id TEXT PRIMARY KEY,
                            tlType TEXT,
                            preDefPhases TEXT)''')

        cur.execute('''CREATE TABLE IF NOT EXISTS connectionINFO(
                            fromLane TEXT NOT NULL,
                            toLane TEXT NOT NULL,
                            direction TEXT,
                            via TEXT,
                            PRIMARY KEY (fromLane, toLane));''')

        cur.execute('''CREATE TABLE IF NOT EXISTS trafficLightStates(
                            frame INT NOT NULL,
                            id TEXT NOT NULL,
                            currPhase TEXT,
                            nextPhase TEXT,
                            switchTime REAL,
                            PRIMARY KEY (frame, id));''')

        cur.execute('''CREATE TABLE IF NOT EXISTS circleObsINFO(
                            id TEXT PRIMARY KEY,
                            edgeID TEXT NOT NULL,
                            centerx REAL NOT NULL,
                            centery REAL NOT NULL,
                            radius REAL NOT NULL);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS rectangleObsINFO(
                            id TEXT PRIMARY KEY,
                            edgeID TEXT NOT NULL,
                            centerx REAL NOT NULL,
                            centery REAL NOT NULL,
                            length REAL NOT NULL,
                            width REAL NOT NULL,
                            yaw REAL NOT NULL);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS geohashINFO(
                            ghx INT NOT NULL,
                            ghy INT NOT NULL,
                            edges TEXT,
                            junctions TEXT,
                            PRIMARY KEY (ghx, ghy));''')

        conn.commit()
        cur.close()
        conn.close()

    def simDescriptionCommit(self, simNote: str):
        currTime = datetime.now()
        insertQuery = '''INSERT INTO simINFO VALUES (?, ?, ?, ?, ?, ?, ?, ?);'''
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(
            insertQuery,
            (currTime, self.dv.x, self.dv.y, self.dv.radius, '', '',
             'local area', simNote)
        )

        conn.commit()
        cur.close()
        conn.close()

    def createTimer(self):
        t = threading.Timer(1, self.dataStore)
        t.daemon = True
        t.start()

    def dataStore(self):
        cnt = 0
        conn = sqlite3.connect(self.dataBase, check_same_thread=False)
        cur = conn.cursor()
        while cnt < 1000 and not self.dataQue.empty():
            tableName, data = self.dataQue.get()
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
        traci.start(
            [
            'sumo-gui' if self.SUMOGUI else 'sumo',
            '-n',
            self.netFile,
            '-r',
            self.rouFile,
            '--step-length',
            '0.1',
            '--lateral-resolution',
            '10',
            '--start',
            '--quit-on-end',
            '-W',
            '--collision.action',
            'remove',
            ]
        )

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

        self.gui.start()
        self.gui.drawMainWindowWhiteBG(
            (self.dv.x-100, self.dv.y-100), 
            (self.dv.x+100, self.dv.y+100)
            )

    def putFrameInfo(self, vid: str, vtag: str, veh: Vehicle):
        self.dataQue.put(
            ('frameINFO', (
                self.timeStep, vid, vtag, veh.x, veh.y, veh.yaw,
                veh.speed, veh.accel, veh.laneID, veh.lanePos, veh.routeIdxQ[-1]
            )
            )
        )

    def putVehicleInfo(self, vid: str, vtins: vehType, routes: str):
        self.dataQue.put(
            ('vehicleINFO', (
                vid, vtins.length, vtins.width, vtins.maxAccel,
                vtins.maxDecel, vtins.maxSpeed, vtins.id, routes
            )
            )
        )

    def drawScene(self):
        # ex, ey refers to the center position of the local area
        ex, ey = self.dv.x, self.dv.y
        node = dpg.add_draw_node(parent="Canvas")
        self.ls.plotScene(node, ex, ey, self.gui.ctf)
        self.dv.plotArea(node, ex, ey, self.gui.ctf)
        if self.ls.vehINAoI:
            for v1 in self.ls.vehINAoI.values():
                v1.plotSelf('AoI', node, ex, ey, self.gui.ctf)
                v1.plotTrajectory(node, ex, ey, self.gui.ctf)
                self.putFrameInfo(v1.id, 'AoI', v1)
        if self.ls.outOfAoI:
            for v2 in self.ls.outOfAoI.values():
                v2.plotSelf('outOfAoI', node, ex, ey, self.gui.ctf)
                v2.plotTrajectory(node, ex, ey, self.gui.ctf)
                self.putFrameInfo(v2.id, 'outOfAoI', v2)

        dpg.draw_text(
            (10, 20),
            'Real time simulation local area.',
            color=(0, 0, 0),
            size=20,
            parent=node
        )
        dpg.draw_text(
            (10, 50),
            'Time step: %.2f s.' % (self.timeStep/10),
            color=(0, 0, 0),
            size=20,
            parent=node)

    def getvTypeIns(self, vtid: str) -> vehType:
        return self.allvTypes[vtid]

    def getVehInfo(self, veh: Vehicle):
        vid = veh.id
        if veh.vTypeID:
            max_decel = veh.maxDecel
        else:
            vtypeid = traci.vehicle.getTypeID(vid)
            if '@' in vtypeid:
                vtypeid = vtypeid.split('@')[0]
            vtins = self.getvTypeIns(vtypeid)
            veh.maxAccel = vtins.maxAccel
            veh.maxDecel = vtins.maxDecel
            veh.length = vtins.length
            veh.width = vtins.width
            veh.maxSpeed = vtins.maxSpeed
            veh.vTypeID = vtypeid
            veh.routes = traci.vehicle.getRoute(vid)
            veh.LLRSet, veh.LLRDict, veh.LCRDict = veh.getLaneLevelRoute(self.nb)

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
            centerx, centery, yaw, speed, accel = veh.plannedTrajectory.pop_last_state()
            try:
                veh.controlSelf(centerx,  centery, yaw, speed, accel)
            except:
                return
        else:
            veh.exitControlMode()

    def updateVeh(self):
        if self.ls.currVehicles:
            for v in self.ls.currVehicles.values():
                self.vehMoveStep(v)

    def setTrajectories(self, trajectories: dict[str, Trajectory]):
        for k, v in trajectories.items():
            veh = self.ls.currVehicles[k]
            veh.plannedTrajectory = v

    def getSce(self):
        dpg.delete_item("Canvas", children_only=True)
        self.ls.updateScene(self.dataQue, self.timeStep)
        self.ls.updateSurroundVeh()

        if self.ls.currVehicles:
            for v in self.ls.currVehicles.values():
                self.getVehInfo(v)

        self.drawScene()

    def exportSce(self):
        return self.ls.exportScene()

    def render(self):
        self.gui.update_inertial_zoom()
        self.getSce()
        dpg.render_dearpygui_frame()

    @property
    def simEnd(self):
        if traci.simulation.getMinExpectedNumber():
            return False
        else:
            return True

    def moveStep(self):
        if self.gui.is_running:
            traci.simulationStep()
            self.timeStep += 1
            self.render()

    def destroy(self):
        time.sleep(1.1)
        traci.close()
        self.gui.destroy()
