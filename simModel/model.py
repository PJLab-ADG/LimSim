import time
import heapq
import random
import math
import numpy as np
from datetime import datetime
from trafficManager.vehicle import Vehicle
from simModel.movingScene import MovingScene
from simModel.networkBuild import NetworkBuild
from simModel.common.dataQueue import RenderQueue, FocusPos, SimPause
from simModel.common.dataBase import Database
from trafficManager.vehicle import Vehicle, DummyVehicle
from trafficManager.planning import planning, routingLane
from trafficManager.evaluation import ScoreCalculator


class Model:

    def __init__(
        self, netFile: str, run_time: int = None, demands: str = None, egoID: int = -1
    ) -> None:
        print(
            "-" * 10,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model initializes ...",
            "-" * 10,
        )
        self.run_time = run_time
        self.demands = demands
        self.timeStep = 0
        self.frequency = 4
        self.plotEngine = None
        self.ego = Vehicle(egoID)
        self.vehRunning: dict[str, Vehicle] = {}
        self.vehDemand: dict[str, Vehicle] = {}
        self.netInfo = NetworkBuild(netFile)
        self.netInfo.getData()
        self.ms = MovingScene(self.netInfo)
        self.renderQueue = RenderQueue(10)
        self.focusPos = FocusPos()
        self.simPause = SimPause()
        self.db = Database()

    def start(self):
        """
        Initializes the simulation by generating the initial vehicle flow,
        routing lanes for the vehicles, creating traffic lights, and initializing the database.
        """
        self.getDemand()  # init flow
        self.route = routingLane(self.netInfo, self.vehDemand)
        self.netInfo.createTrafficLight(self.frequency)
        self.db.initDB()

    def getVehInfo(self, veh: Vehicle):
        """Retrieves and updates the information of a given vehicle."""
        veh.xQ.append(veh.x)
        veh.yQ.append(veh.y)
        veh.hdgQ.append(veh.hdg)
        veh.velQ.append(veh.vel)
        veh.accQ.append(veh.acc)
        veh.yawQ.append(veh.yaw)
        veh.roadIdQ.append(veh.roadId)
        veh.planXQ = veh.planTra.xQ
        veh.planYQ = veh.planTra.yQ
        veh.planHdgQ = veh.planTra.hdgQ
        veh.planVelQ = veh.planTra.velQ
        veh.planAccQ = veh.planTra.accQ
        veh.planYawQ = veh.planTra.yawQ
        veh.planRoadIdQ = veh.planTra.roadIdQ

    def getVehFlow(self, period):
        """Generates vehicle flow for a given period by reading demand data from a file."""
        random.seed(0)
        carFlow = dict()
        V_MIN, V_MAX = 0, 0
        infilie = open(self.demands)
        demand = infilie.readline()  # skip the first line
        demand = infilie.readline()
        velID = 1
        while len(demand) > 1:
            sub = demand.strip("\n").split(",")
            fromNone, toNone, direction, qr = (
                str(sub[0]),
                str(sub[1]),
                int(sub[2]),
                int(sub[3]),
            )
            lampda = qr / 3600
            arrivalTime = 0
            length, width = 5.0, 2.0
            while arrivalTime < period:
                # generate arrival time
                timeHeadway = round(-1 / lampda * math.log(random.random()), 2)
                arrivalTime += timeHeadway
                initVel = round(random.random() * (V_MAX - V_MIN), 2) + V_MIN
                carFlow[velID] = Vehicle(
                    velID,
                    arrivalTime,
                    fromNone,
                    toNone,
                    direction,
                    initVel,
                    length,
                    width,
                )
                velID += 1
            demand = infilie.readline()
        return carFlow

    def getDemand(self):
        """
        Retrieves the vehicle demand for a given period of time.

        The demand is calculated by first determining the period of time to consider (30% of the total run time).
        Then, it generates the vehicle flow for this period using the `getVehFlow` method.
        Finally, it updates the `vehDemand` dictionary with the vehicles that have an arrival time greater than the current time step.
        """
        period = self.run_time * 0.3
        carFlow = self.getVehFlow(period)
        for t in np.arange(period):
            for vehId, veh in carFlow.items():
                if vehId not in self.vehDemand and veh.arrivalTime > t:
                    self.vehDemand[vehId] = veh

    def update_evluation_data(self):
        """Updates the evaluation data for the current simulation."""
        vehs = list(self.ms.vehInAoI.values())
        score_calc = ScoreCalculator(self.ego, vehs, self.netInfo, self.frequency)
        self.ego.drivingScore = score_calc.calculate()

    def moveStep(self):
        """
        Advances the simulation by one time step.

        Increments the internal time step counter, updates the state of traffic lights,
        and plans the movement of vehicles based on the current time step and simulation frequency.
        """
        self.timeStep += 1
        for index in self.netInfo.tls:
            self.netInfo.tls[index].state_calculation(self.timeStep)
        self.vehDemand, self.vehRunning = planning(
            self.netInfo, self.vehDemand, self.vehRunning, self.timeStep, self.frequency
        )

    def updateVeh(self):
        """
        Updates the vehicle information and its surroundings in the simulation.

        This function checks the status of the ego vehicle and updates its information
        accordingly. It also updates the evaluation data, scene, and surrounding vehicles.
        """
        if self.ego.id == -1 and self.vehRunning:
            self.ego = list(self.vehRunning.values())[0]
        # if ego arrive the destination
        if self.vehRunning and self.ego.id != 0 and self.ego.id not in self.vehRunning:
            state = [
                self.ego.x,
                self.ego.y,
                self.ego.scf,
                self.ego.tcf,
                self.ego.roadId,
            ]
            self.ego = DummyVehicle(id=0, state=state)
        if self.ego.id in self.vehRunning or self.ego.id == 0:
            for veh in self.vehRunning.values():
                self.getVehInfo(veh)
            if self.ego.id > 0:
                self.update_evluation_data()
            self.ms.updateScene(self.ego)
            self.ms.updateSurroudVeh(self.ego, self.vehRunning)
            self.simTime = self.timeStep / self.frequency
            self.ms.getPlotInfo(self.ego, self.netInfo.tls)
            self.renderQueue.put((self.ms.vehInfo, self.simTime))
            self.db.updateDB(self.vehRunning, self.netInfo.tls, self.timeStep)
        self.updateFocusPos()

    def updateFocusPos(self):
        """
        Updates the focus position in the simulation.

        If a new focus position is available, it retrieves the position, removes it from the queue,
        and checks if there is a nearby vehicle. If a nearby vehicle is found, it updates the ego vehicle.
        Otherwise, it converts the mouse position to Frenet coordinates and creates a dummy vehicle.
        If the conversion fails, it prints an error message.
        """
        if self.focusPos.getPos():
            newFocusPos = self.focusPos.getPos()
            self.focusPos.queue.pop()
            nearVehId = self.getNearVeh(newFocusPos)
            if nearVehId:
                self.ego = self.vehRunning[nearVehId]
            else:
                get_mouse_info = self.netInfo.cartesian2Frenet(
                    newFocusPos[0], newFocusPos[1]
                )
                if get_mouse_info:
                    state = newFocusPos + get_mouse_info
                    self.ego = DummyVehicle(id=0, state=state)
                else:
                    print("please choose a road or a vehicle")

    def getNearVeh(self, pos: list):
        """Retrieves the ID of the nearest vehicle to a given position."""
        disList = []
        for vehId, veh in self.vehRunning.items():
            dis = np.hypot(veh.x - pos[0], veh.y - pos[1])
            disList.append([dis, vehId])
        heapq.heapify(disList)
        if len(disList) > 0:
            minDis, nearVehId = heapq.heappop(disList)
            if minDis <= 10:
                return nearVehId

    def replayMoveStep(self):
        self.timeStep += 1
        if not self.run_time:
            self.run_time = self.db.getRunTime()

    def replayUpdateVeh(self):
        self.simTime = self.timeStep / self.frequency
        try:
            vehInfo, tlsInfo = self.db.getDB(self.timeStep)
        except:
            return
        self.vehRunning = {}
        for vehId in vehInfo:
            self.vehRunning[vehId] = vehInfo[vehId]
        if self.vehRunning:
            if self.ego.id == -1:
                self.ego = list(self.vehRunning.values())[0]
            elif self.ego.id > 0 and self.ego.id in self.vehRunning:
                self.ego = self.vehRunning[self.ego.id]
            elif self.ego.id != 0:
                # the ego arrives the destination
                state = [
                    self.ego.x,
                    self.ego.y,
                    self.ego.scf,
                    self.ego.tcf,
                    self.ego.roadId,
                ]
                self.ego = DummyVehicle(id=0, state=state)
        if self.ego.id in self.vehRunning or self.ego.id == 0:
            if self.ego.id > 0:
                self.update_evluation_data()
            self.ms.updateScene(self.ego)
            self.ms.updateSurroudVeh(self.ego, self.vehRunning)
            self.ms.getPlotInfo(self.ego, tlsInfo)
            self.renderQueue.put((self.ms.vehInfo, self.simTime))
        self.updateFocusPos()

    def destroy(self):
        time.sleep(2)
        if self.plotEngine:
            self.plotEngine.gui.destroy()

    def end(self):
        return self.timeStep >= self.run_time
