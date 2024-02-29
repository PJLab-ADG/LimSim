import os
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List

import numpy as np
import traci
import pickle
from PIL import Image
from rich import print

from simModel.CarFactory import Vehicle, egoCar
from simModel.DataQueue import (
    CameraImages, ImageQueue, QAQueue, QuestionAndAnswer, RenderQueue,
)
from simModel.DBBridge import DBBridge
from simModel.MovingScene import MovingScene
from simModel.NetworkBuild import NetworkBuild
from utils.simBase import vehType
from utils.trajectory import Trajectory


class SettingErro(Exception):
    def __init__(self, errorInfo: str) -> None:
        super().__init__(self)
        self.errorInfo = errorInfo

    def __str__(self) -> str:
        return self.errorInfo

def resizeImage(img: np.ndarray, width: int, height: int) -> bytes:
    img = Image.fromarray(img)
    img_resized = img.resize((width, height))
    np_img_resized = np.array(img_resized)
    print(type(np_img_resized))
    return np_img_resized.tobytes()

class Model:
    def __init__(
        self, egoID: str,
        netFile: str, rouFile: str, cfgFile: str,
        dataBase: str = None,
        SUMOGUI: bool = False, 
        CARLACosim: bool = True,
        carla_host: str = '127.0.0.1',
        carla_port: int = 2000,
        tls_manager: str = 'sumo'
    ) -> None:
        print('[green bold]Model initialized at {}.[/green bold]'.format(
            datetime.now().strftime('%H:%M:%S.%f')[:-3]))
        self.netFile = netFile
        self.rouFile = rouFile
        self.cfgFile = cfgFile
        self.SUMOGUI = SUMOGUI
        self.CARLACosim = CARLACosim
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.tls_manager = tls_manager
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
        self.simDescriptionCommit()
        self.renderQueue = RenderQueue(5)
        self.imageQueue = ImageQueue(50)
        self.QAQ = QAQueue(5)

        self.nb = NetworkBuild(self.dataBase, netFile)
        self.nb.getData()
        self.nb.buildTopology()

        self.ms = MovingScene(self.nb, self.ego)

        self.allvTypes = None

        self.netBoundary = None

    def simDescriptionCommit(self):
        currTime = datetime.now()
        insertQuery = '''INSERT INTO simINFO VALUES (?, ?, ?);'''
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(
            insertQuery,
            (currTime, self.ego.id, '')
        )

        conn.commit()
        cur.close()
        conn.close()

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
            '-W'
        ])

        ((x1, y1), (x2, y2)) = traci.simulation.getNetBoundary()
        self.netBoundary = ((x1, y1), (x2, y2))
        netBoundaryStr = f"{x1},{y1} {x2},{y2}"
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute(f"""UPDATE simINFO SET netBoundary = '{netBoundaryStr}';""")
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

        if self.CARLACosim:
            from sumo_integration.run_synchronization import getSynchronization
            self.carlaSync = getSynchronization(
                sumo_cfg_file=self.cfgFile,
                carla_host=self.carla_host,
                carla_port=self.carla_port,
                ego_id=self.ego.id,
                step_length=0.1,
                tls_manager=self.tls_manager,
                sync_vehicle_color=True,
                sync_vehicle_lights=True
            )
        else:
            return

    def commitFrameInfo(self, vid: str, vtag: str, veh: Vehicle):
        self.dbBridge.putData(
            'frameINFO',
            (
                self.timeStep, vid, vtag, veh.x, veh.y, veh.yaw, veh.speed,
                veh.accel, veh.laneID, veh.lanePos, veh.routeIdxQ[-1]
            )
        )


    def commitVehicleInfo(self, vid: str, vtins: vehType, routes: str):
        self.dbBridge.putData(
            'vehicleINFO', 
            (
                vid, vtins.length, vtins.width, vtins.maxAccel,
                vtins.maxDecel, vtins.maxSpeed, vtins.id, routes
            )
        )

    def putVehicleINFO(self):
        self.commitFrameInfo(self.ego.id, 'ego', self.ego)
        if self.ms.vehINAoI:
            for v1 in self.ms.vehINAoI.values():
                self.commitFrameInfo(v1.id, 'AoI', v1)
        if self.ms.outOfAoI:
            for v2 in self.ms.outOfAoI.values():
                try:
                    self.commitFrameInfo(v2.id, 'outOfAoI', v2)
                except TypeError:
                    return

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
            self.commitVehicleInfo(vid, vtins, routes)
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
            self.ms.updateScene(self.dbBridge, self.timeStep)
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
            self.renderQueue.put((roadgraphRenderData, VRDDict))

    def exportSce(self):
        if self.tpStart:
            return self.ms.exportScene()
        else:
            return None, None
        
    def putCARLAImage(self):
        if self.CARLACosim:
            carla_ego = self.carlaSync.getEgo()
            if carla_ego:
                self.carlaSync.moveSpectator(carla_ego)
                self.carlaSync.setCameras(carla_ego)
                ci = self.carlaSync.getCameraImages()
                if ci:
                    ci.resizeImage(560, 315)
                    self.imageQueue.put(ci)
                    self.dbBridge.putData(
                        'imageINFO',
                        (
                            self.timeStep, 
                            sqlite3.Binary(pickle.dumps(ci.CAM_FRONT)),
                            sqlite3.Binary(pickle.dumps(ci.CAM_FRONT_RIGHT)),
                            sqlite3.Binary(pickle.dumps(ci.CAM_FRONT_LEFT)),
                            sqlite3.Binary(pickle.dumps(ci.CAM_BACK_LEFT)),
                            sqlite3.Binary(pickle.dumps(ci.CAM_BACK)),
                            sqlite3.Binary(pickle.dumps(ci.CAM_BACK_RIGHT))
                        )
                    )
        else:
            return
        
    def getCARLAImage(
            self, start_frame: int, steps: int=1
        ) -> List[CameraImages]:
        return self.imageQueue.get(start_frame, steps)
        
    def putQA(self, QA: QuestionAndAnswer):
        self.QAQ.put(QA)
        self.dbBridge.putData(
            'QAINFO',
            (
                self.timeStep, QA.description, QA.navigation,
                QA.actions, QA.few_shots, QA.response,
                QA.prompt_tokens, QA.completion_tokens, QA.total_tokens, QA.total_time,  QA.choose_action
            )
        )

    def moveStep(self):
        traci.simulationStep()
        if self.CARLACosim:
            self.carlaSync.tick()
        self.timeStep += 1
        if self.ego.id in traci.vehicle.getIDList():
            self.getSce()
            self.putRenderData()
            self.putCARLAImage()
            if not self.tpStart:
                self.tpStart = 1
            
    def destroy(self):
        traci.close()
        self.dbBridge.close()
        if self.CARLACosim:
            self.carlaSync.close()
