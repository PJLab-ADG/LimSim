import os
import sqlite3
import threading
import time
from typing import List, Dict
import xml.etree.ElementTree as ET
from datetime import datetime
from queue import Queue
from math import sin, cos, pi

import dearpygui.dearpygui as dpg
import numpy as np
import traci
from rich import print
from traci import TraCIException

from simModel.common.carFactory import Vehicle, egoCar
from simModel.common.gui import GUI
from simModel.egoTracking.movingScene import MovingScene
from simModel.common.networkBuild import NetworkBuild
from utils.trajectory import State, Trajectory
from utils.simBase import MapCoordTF, vehType

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

    def __init__(self,
                 egoID: str,
                 netFile: str,
                 rouFile: str,
                 obsFile: str = None,
                 dataBase: str = None,
                 SUMOGUI: int = 0,
                 simNote: str = None,
                 carla_cosim: bool = False) -> None:
        print('[green bold]Model initialized at {}.[/green bold]'.format(
            datetime.now().strftime('%H:%M:%S.%f')[:-3]))
        self.netFile = netFile
        self.rouFile = rouFile
        self.obsFile = obsFile
        self.SUMOGUI = SUMOGUI
        self.sim_mode: str = 'RealTime'
        self.timeStep = 0
        # tpStart marks whether the trajectory planning is started,
        # when the ego car appears in the network, tpStart turns into 1.
        self.tpStart = 0
        # tpEnd marks whether the trajectory planning is end,
        # when the ego car leaves the network, tpEnd turns into 1.
        self.tpEnd = 0
        # need carla cosimulation
        self.carla_cosim = carla_cosim

        self.ego = egoCar(egoID)

        if dataBase:
            self.dataBase = dataBase
        else:
            self.dataBase = datetime.strftime(
                datetime.now(), '%Y-%m-%d_%H-%M-%S') + '_egoTracking' + '.db'

        self.createDatabase()
        self.simDescriptionCommit(simNote)
        self.dataQue = Queue()
        self.createTimer()

        self.nb = NetworkBuild(self.dataBase, netFile, obsFile)
        self.nb.getData()
        self.nb.buildTopology()

        self.ms = MovingScene(self.nb, self.ego)

        self.allvTypes = None

        self.gui = GUI('real-time-ego')

        self.evaluation = RealTimeEvaluation(dt=0.1)

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

        cur.execute('''CREATE TABLE IF NOT EXISTS evaluationINFO(
                    frame INT PRIMARY KEY,
                    offset REAL,
                    discomfort REAL,
                    collision REAL,
                    orientation REAL,
                    consumption REAL);''')

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
            (currTime, None, None, None, self.ego.id, '', 'ego track', simNote))

        conn.commit()
        cur.close()
        conn.close()

    def createTimer(self):
        if not self.tpEnd or not self.dataQue.empty():
            t = threading.Timer(1, self.dataStore)
            t.daemon = True
            t.start()

    def dataStore(self):
        # stime = time.time()
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

        if self.carla_cosim:
            num_clients = "2"
        else:
            num_clients = "1"
        traci.start([
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
            "--num-clients",
            num_clients,
        ], port=8813)
        traci.setOrder(1)

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

    def plotVState(self):
        if self.ego.speedQ:
            currLane = traci.vehicle.getLaneID(self.ego.id)
            if ':' not in currLane:
                try:
                    laneMaxSpeed = traci.lane.getMaxSpeed(currLane)
                except TraCIException:
                    laneMaxSpeed = 15
            else:
                laneMaxSpeed = 15
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

        if self.ego.plannedTrajectory:
            if self.ego.plannedTrajectory.velQueue:
                vfy = list(self.ego.plannedTrajectory.velQueue)
                vfx = list(range(1, len(vfy) + 1))
                dpg.set_value('v_series_tag_future', [vfx, vfy])
            if self.ego.plannedTrajectory.accQueue:
                afy = list(self.ego.plannedTrajectory.accQueue)
                afx = list(range(1, len(afy) + 1))
                dpg.set_value('a_series_tag_future', [afx, afy])

    def putFrameInfo(self, vid: str, vtag: str, veh: Vehicle):
        self.dataQue.put(
            ('frameINFO',
             (self.timeStep, vid, vtag, veh.x, veh.y, veh.yaw, veh.speed,
              veh.accel, veh.laneID, veh.lanePos, veh.routeIdxQ[-1])))

    def putVehicleInfo(self, vid: str, vtins: vehType, routes: str):
        self.dataQue.put(
            ('vehicleINFO', (vid, vtins.length, vtins.width, vtins.maxAccel,
                             vtins.maxDecel, vtins.maxSpeed, vtins.id, routes)))

    def putEvaluationInfo(self, points: np.ndarray):
        self.dataQue.put(
            ('evaluationINFO', tuple([self.timeStep] + points.tolist())))

    def drawScene(self):
        ex, ey = self.ego.x, self.ego.y
        node = dpg.add_draw_node(parent="Canvas")
        self.ms.plotScene(node, ex, ey, self.gui.ctf)
        self.ego.plotSelf('ego', node, ex, ey, self.gui.ctf)
        self.ego.plotdeArea(node, ex, ey, self.gui.ctf)
        self.ego.plotTrajectory(node, ex, ey, self.gui.ctf)
        self.putFrameInfo(self.ego.id, 'ego', self.ego)
        if self.ms.vehINAoI:
            for v1 in self.ms.vehINAoI.values():
                v1.plotSelf('AoI', node, ex, ey, self.gui.ctf)
                v1.plotTrajectory(node, ex, ey, self.gui.ctf)
                self.putFrameInfo(v1.id, 'AoI', v1)
        if self.ms.outOfAoI:
            for v2 in self.ms.outOfAoI.values():
                v2.plotSelf('outOfAoI', node, ex, ey, self.gui.ctf)
                v2.plotTrajectory(node, ex, ey, self.gui.ctf)
                self.putFrameInfo(v2.id, 'outOfAoI', v2)

        mvNode = dpg.add_draw_node(parent='movingScene')
        mvCenterx, mvCentery = self.mapCoordTF.dpgCoord(ex, ey)
        dpg.draw_circle((mvCenterx, mvCentery),
                        self.ego.deArea * self.mapCoordTF.zoomScale,
                        thickness=0,
                        fill=(243, 156, 18, 60),
                        parent=mvNode)

        infoNode = dpg.add_draw_node(parent='simInfo')
        dpg.draw_text((5, 5),
                      'Real time simulation ego tracking.',
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
        self.putEvaluationInfo(self.evaluation.result)

        transformed_points = self._evaluation_transform_coordinate(points,
                                                                   scale=30)
        transformed_points.append(transformed_points[0])

        radarNode = dpg.add_draw_node(parent='radarPlot')
        dpg.draw_polygon(transformed_points,
                         color=(75, 207, 250, 100),
                         fill=(75, 207, 250, 100),
                         thickness=5,
                         parent=radarNode)

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

    def update_evluation_data(self):
        current_lane = self.nb.getLane(self.ego.laneID)
        if current_lane is None:
            current_lane = self.nb.getJunctionLane(self.ego.laneID)
        agents = list(self.ms.vehINAoI.values())
        self.evaluation.update_data(self.ego, current_lane, agents)

    def getSce(self):
        if self.ego.id in traci.vehicle.getIDList():
            self.tpStart = 1
            dpg.delete_item("Canvas", children_only=True)
            dpg.delete_item("movingScene", children_only=True)
            dpg.delete_item("simInfo", children_only=True)
            dpg.delete_item("radarPlot", children_only=True)
            self.ms.updateScene(self.dataQue, self.timeStep)
            self.ms.updateSurroudVeh()

            self.getVehInfo(self.ego)
            if self.ms.currVehicles:
                for v in self.ms.currVehicles.values():
                    self.getVehInfo(v)

            self.update_evluation_data()

            self.drawScene()
            self.plotVState()
        else:
            if self.tpStart:
                print('[cyan]The ego car has reached the destination.[/cyan]')
                self.tpEnd = 1

        if self.tpStart:
            if self.ego.arriveDestination(self.nb):
                self.tpEnd = 1
                print('[cyan]The ego car has reached the destination.[/cyan]')

    def exportSce(self):
        if self.tpStart:
            return self.ms.exportScene()
        else:
            return None, None

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
            "offset", "discomfort", "collision", "orientation",
            "consumption"
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
        ((x1, y1), (x2, y2)) = traci.simulation.getNetBoundary()
        netBoundary = f"{x1},{y1} {x2},{y2}"
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(f"""UPDATE simINFO SET netBoundary = '{netBoundary}';""")
        conn.commit()
        conn.close()
        self.mapCoordTF = MapCoordTF((x1, y1), (x2, y2), 'macroMap')
        mNode = dpg.add_draw_node(parent='mapBackground')
        for jid in self.nb.junctions.keys():
            self.nb.plotMapJunction(jid, mNode, self.mapCoordTF)

        self.gui.drawMainWindowWhiteBG((x1-100, y1-100), (x2+100, y2+100))

    def render(self):
        self.gui.update_inertial_zoom()
        self.getSce()
        dpg.render_dearpygui_frame()

    def moveStep(self):
        if self.gui.is_running:
            traci.simulationStep()
            self.timeStep += 1
        if not dpg.is_dearpygui_running():
            self.tpEnd = 1
        if self.ego.id in traci.vehicle.getIDList():
            if not self.tpStart:
                self.gui.start()
                self.drawRadarBG()
                self.drawMapBG()
                self.tpStart = 1
            self.render()

    def destroy(self):
        # stop the saveThread.
        time.sleep(1.1)
        traci.close()
        self.gui.destroy()
