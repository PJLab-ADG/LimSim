from __future__ import annotations
import dearpygui.dearpygui as dpg
import sqlite3
from rich import print
import time
from datetime import datetime
from queue import Queue
import threading
import os

from simModel.common.gui import GUI
from simModel.common.networkBuild import Rebuild
from simModel.common.carFactory import Vehicle, DummyVehicle
from simModel.fixedScene.localScene import LocalSceneReplay
from utils.trajectory import Trajectory, State, Rectangle, RecCollide
from utils.simBase import vehType


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

        self.tpEnd = 0

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

        self.lsr = LocalSceneReplay(self.rb, self.dv)

        self.gui = GUI('real-time-local')
        self.gui.start()
        self.gui.drawMainWindowWhiteBG(
            (self.dv.x-100, self.dv.y-100), 
            (self.dv.x+100, self.dv.y+100)
            )
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
        netBoundary = ''
        currTime = datetime.now()
        insertQuery = '''INSERT INTO simINFO VALUES (?, ?, ?, ?, ?, ?, ?, ?);'''
        conn = sqlite3.connect(self.dataBase2)
        cur = conn.cursor()
        cur.execute(
            insertQuery,
            (currTime, self.dv.x, self.dv.y, self.dv.radius, 
             None, netBoundary, 'ego track interactive replay', simNote))

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

    def getAllvType(self):
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()
        cur.execute("""SELECT DISTINCT vTypeID, length, width, maxAccel, maxDecel, maxSpeed
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
            self.lsr.outOfRange.add(vehid)
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
        veh.LLRSet, veh.LLRDict, veh.LCRDict = veh.getLaneLevelRoute(self.rb)

        return veh

    def isInvolved(self, veh: Vehicle, currVehs: dict[str, Vehicle]) -> bool:
        # if the vehicle's dbTrajectory is too short, limsim will take over it 
        # until the vehicle drive out of the AoI, avoiding the vhicles's suddenly fading.
        if len(veh.dbTrajectory.states) <= 10:
            return True
        else:
            for cv in currVehs.values():
                if self.VTCollisionCheck(veh, cv):
                    return True
            return False

    def setTrajectories(self, trajectories: dict[str, Trajectory]):
        for vid, vins in self.lsr.currVehicles.items():
            dbTrajectory = self.dbTrajectory(vid, self.timeStep)
            if vid in trajectories.keys():
                if not vins.dbTrajectory:
                    vins.plannedTrajectory = trajectories[vid]
                    vins.dbTrajectory = None
                else:
                    if self.isInvolved(vins, self.lsr.vehINAoI):
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

    def updateVeh(self, veh: Vehicle):
        message = veh.replayUpdate()
        if message == 'Failure':
            self.lsr.outOfRange.add(veh.id)

    def drawSce(self):
        node = dpg.add_draw_node(parent="Canvas")
        cx, cy = self.dv.x, self.dv.y

        self.lsr.plotScene(node, cx, cy, self.gui.ctf)

        self.dv.plotArea(node, cx, cy, self.gui.ctf)

        if self.lsr.vehINAoI:
            for vAoI in self.lsr.vehINAoI.values():
                vAoI.plotSelf('AoI', node, cx, cy, self.gui.ctf)
                if vAoI.plannedTrajectory:
                    vAoI.plotTrajectory(node, cx, cy, self.gui.ctf)
                    continue
                if vAoI.dbTrajectory:
                    vAoI.plotDBTrajectory(node, cx, cy, self.gui.ctf)
                #     vAoI.plotDBTrajectory(node, ex, ey)
                # else:
                #     vAoI.plotDBTrajectory(node, ex, ey)
        if self.lsr.outOfAoI:
            for vSce in self.lsr.outOfAoI.values():
                vSce.plotSelf('Sce', node, cx, cy, self.gui.ctf)
                # vSce.plotTrajectory(node, ex, ey)

        dpg.draw_text(
            (10, 20),
            'Interactive replay {}'.format(self.dataBase),
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
            recA = Rectangle([stateA.x, stateB.x],
                             veh1.length, veh1.width, stateA.yaw)
            recB = Rectangle([stateB.x, stateB.y],
                             veh2.length, veh2.width, stateB.yaw)
            rc = RecCollide(recA, recB)
            if rc.isCollide():
                return True
        return False

    # new vehicles
    def newVehCollisionCheck(self, veh: Vehicle, currVehs: dict[str, Vehicle]) -> bool:
        for cv in currVehs.values():
            if self.VTCollisionCheck(veh, cv):
                return True
        return False

    def getSce(self):
        nextFrameVehs = self.getNextFrameVehs()
        if nextFrameVehs:
            dpg.delete_item("Canvas", children_only=True)
            for veh in self.lsr.currVehicles.values():
                self.updateVeh(veh)
            newVehs = nextFrameVehs - self.lsr.currVehicles.keys()
            for nv in newVehs:
                veh = self.initVeh(nv, self.timeStep)
                # if veh.id == '107':
                #     print(veh.dbTrajectory)
                self.updateVeh(veh)
                if not self.newVehCollisionCheck(veh, self.lsr.vehINAoI):
                    self.lsr.currVehicles[nv] = veh
                # self.sr.currVehicles[nv] = veh
            self.lsr.updateSurroundVeh()
        else:
            if not self.tpEnd:
                print('The simulation is over.')
                self.tpEnd = 1
                self.gui.is_running = 0

    def exportSce(self):
        return self.lsr.exportScene()

    def render(self):
        self.gui.update_inertial_zoom()
        dpg.delete_item('Canvas', children_only=True)
        self.lsr.updateScene(self.dataBase, self.timeStep)
        self.drawSce()
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
