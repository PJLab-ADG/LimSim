import traci
from traci import TraCIException
from math import sqrt
from queue import Queue
import dearpygui.dearpygui as dpg
import sqlite3


from simModel.common.networkBuild import NetworkBuild, Rebuild
from simModel.common.carFactory import Vehicle, egoCar, DummyVehicle
from utils.roadgraph import RoadGraph
from utils.simBase import CoordTF


class MovingScene:
    def __init__(self, netInfo: NetworkBuild, ego: egoCar) -> None:
        self.netInfo = netInfo
        self.ego = ego
        self.edges: set = None
        self.junctions: set = None
        self.currVehicles: dict[str, Vehicle] = {}
        self.vehINAoI: dict[str, Vehicle] = {}
        self.outOfAoI: dict[str, Vehicle] = {}

    # if lane-lenght <= the self.ego's deArea, return current edge, current
    # edge's upstream intersection and current edge's downstream intersection.
    # else, judge if the upstream intersection or downstream intersection
    # is in the range of the vehicle's deArea.
    def updateScene(self, dataQue: Queue, timeStep: int):
        ex, ey = traci.vehicle.getPosition(self.ego.id)
        currGeox = int(ex // 100)
        currGeoy = int(ey // 100)

        sceGeohashIDs = (
            (currGeox-1, currGeoy-1),
            (currGeox, currGeoy-1),
            (currGeox+1, currGeoy-1),
            (currGeox-1, currGeoy),
            (currGeox, currGeoy),
            (currGeox+1, currGeoy),
            (currGeox-1, currGeoy+1),
            (currGeox, currGeoy+1),
            (currGeox+1, currGeoy+1),
        )

        NowEdges: set = set()
        NowJuncs: set = set()

        for sgh in sceGeohashIDs:
            try:
                geohash = self.netInfo.geoHashes[sgh]
            except KeyError:
                continue
            NowEdges = NowEdges | geohash.edges
            NowJuncs = NowJuncs | geohash.junctions

        self.edges = NowEdges
        self.junctions = NowJuncs

        NowTLs = {}
        for jid in NowJuncs:
            junc = self.netInfo.getJunction(jid)
            for jlid in junc.JunctionLanes:
                jl = self.netInfo.getJunctionLane(jlid)
                tlid = jl.tlLogic
                if tlid:
                    if tlid not in NowTLs.keys():
                        currPhaseIndex = traci.trafficlight.getPhase(tlid)
                        tlLogic = self.netInfo.getTlLogic(tlid)
                        currPhase = tlLogic.currPhase(currPhaseIndex)
                        nextPhase = tlLogic.nextPhase(currPhaseIndex)
                        switchTime = round(traci.trafficlight.getNextSwitch(
                            tlid) - traci.simulation.getTime(), 1)
                        NowTLs[tlid] = (currPhase, nextPhase, switchTime)
                        dataQue.put((
                            'trafficLightStates',
                            (timeStep, tlid, currPhase, nextPhase, switchTime)
                        ))
                    else:
                        currPhase, nextPhase, switchTime = NowTLs[tlid]
                    jl.currTlState = currPhase[jl.tlsIndex]
                    jl.nexttTlState = nextPhase[jl.tlsIndex]
                    jl.switchTime = switchTime

    def addVeh(self, vdict: dict, vid: str) -> None:
        if vdict and vid in vdict.keys():
            return
        else:
            vehIns = Vehicle(vid)
            vdict[vid] = vehIns

    # getSurroundVeh will update all vehicle's attributes
    # so don't update again in other steps
    def updateSurroudVeh(self):
        nextStepVehicles = set()
        for ed in self.edges:
            nextStepVehicles = nextStepVehicles | set(
                traci.edge.getLastStepVehicleIDs(ed)
            )

        for jc in self.junctions:
            jinfo = self.netInfo.getJunction(jc)
            if jinfo.JunctionLanes:
                for il in jinfo.JunctionLanes:
                    nextStepVehicles = nextStepVehicles | set(
                        traci.lane.getLastStepVehicleIDs(il)
                    )

        newVehicles = nextStepVehicles - self.currVehicles.keys()
        for nv in newVehicles:
            self.addVeh(self.currVehicles, nv)

        ex, ey = traci.vehicle.getPosition(self.ego.id)
        vehInAoI = {}
        outOfAoI = {}
        outOfRange = set()
        for vk, vv in self.currVehicles.items():
            if vk == self.ego.id:
                continue
            try:
                x, y = traci.vehicle.getPosition(vk)
            except TraCIException:
                # vehicle is leaving the network.
                outOfRange.add((vk, 0))
                continue
            if sqrt(pow((ex - x), 2) + pow((ey - y), 2)) <= self.ego.deArea:
                try:
                    vehArrive = vv.arriveDestination(self.netInfo)
                except:
                    vehArrive = False

                if vehArrive:
                    outOfRange.add((vk, 1))
                else:
                    vehInAoI[vk] = vv
            elif sqrt(pow((ex - x), 2) + pow((ey - y), 2)) <= 2 * self.ego.deArea:
                try:
                    vehArrive = vv.arriveDestination(self.netInfo)
                except:
                    vehArrive = False

                if vehArrive:
                    outOfRange.add((vk, 1))
                else:
                    outOfAoI[vk] = vv
            else:
                vv.exitControlMode()
                outOfRange.add((vk, 0))

        for vid, atag in outOfRange:
            del(self.currVehicles[vid])
            # if the vehicle arrived the destination, remove it from the simulation
            if atag:
                try:
                    traci.vehicle.remove(vid)
                except TraCIException:
                    pass

        self.vehINAoI = vehInAoI
        self.outOfAoI = outOfAoI

    def plotScene(self, node: dpg.node, ex: float, ey: float, ctf: CoordTF):
        if self.edges:
            for ed in self.edges:
                self.netInfo.plotEdge(ed, node, ex, ey, ctf)

        if self.junctions:
            for jc in self.junctions:
                self.netInfo.plotJunction(jc, node, ex, ey, ctf)

    def exportScene(self):
        roadgraph = RoadGraph()

        for eid in self.edges:
            Edge = self.netInfo.getEdge(eid)
            roadgraph.edges[eid] = Edge
            for lane in Edge.lanes:
                roadgraph.lanes[lane] = self.netInfo.getLane(lane)

        for junc in self.junctions:
            Junction = self.netInfo.getJunction(junc)
            for jl in Junction.JunctionLanes:
                juncLane = self.netInfo.getJunctionLane(jl)
                roadgraph.junction_lanes[juncLane.id] = juncLane

        # export vehicles' information using dict.
        vehicles = {
            'egoCar': self.ego.export2Dict(self.netInfo),
            'carInAoI': [av.export2Dict(self.netInfo) for av in self.vehINAoI.values()],
            'outOfAoI': [sv.export2Dict(self.netInfo) for sv in self.outOfAoI.values()]
        }

        return roadgraph, vehicles


class SceneReplay:
    def __init__(self, netInfo: Rebuild, ego: egoCar) -> None:
        self.netInfo = netInfo
        self.ego = ego
        self.currVehicles: dict[str, Vehicle] = {}
        self.vehINAoI: dict[str, Vehicle] = {}
        self.outOfAoI: dict[str, Vehicle] = {}
        self.outOfRange = set()
        self.edges: set = None
        self.junctions: set = None

    def updateScene(self, dataBase: str, timeStep: int):
        ex, ey = self.ego.x, self.ego.y
        currGeox = int(ex // 100)
        currGeoy = int(ey // 100)

        sceGeohashIDs = (
            (currGeox-1, currGeoy-1),
            (currGeox, currGeoy-1),
            (currGeox+1, currGeoy-1),
            (currGeox-1, currGeoy),
            (currGeox, currGeoy),
            (currGeox+1, currGeoy),
            (currGeox-1, currGeoy+1),
            (currGeox, currGeoy+1),
            (currGeox+1, currGeoy+1),
        )

        NowEdges: set = set()
        NowJuncs: set = set()

        for sgh in sceGeohashIDs:
            try:
                geohash = self.netInfo.geoHashes[sgh]
            except KeyError:
                continue
            NowEdges = NowEdges | geohash.edges
            NowJuncs = NowJuncs | geohash.junctions

        self.edges = NowEdges
        self.junctions = NowJuncs

        NowTLs = {}
        conn = sqlite3.connect(dataBase)
        cur = conn.cursor()
        cur.execute(
            '''SELECT * FROM trafficLightStates WHERE frame=%i;''' % timeStep)
        tlsINFO = cur.fetchall()
        if tlsINFO:
            for tls in tlsINFO:
                frame, tlid, currPhase, nextPhase, switchTime = tls
                NowTLs[tlid] = (currPhase, nextPhase, switchTime)

        cur.close()
        conn.close()

        if NowTLs:
            for jid in NowJuncs:
                junc = self.netInfo.getJunction(jid)
                if junc:
                    for jlid in junc.JunctionLanes:
                        jl = self.netInfo.getJunctionLane(jlid)
                        tlid = jl.tlLogic
                        if tlid:
                            try:
                                currPhase, nextPhase, switchTime = NowTLs[tlid]
                            except KeyError:
                                continue
                            jl.currTlState = currPhase[jl.tlsIndex]
                            jl.nexttTlState = nextPhase[jl.tlsIndex]
                            jl.switchTime = switchTime

    def updateSurroudVeh(self):
        outOfRange = set()
        for vid in self.outOfRange:
            try:
                del (self.currVehicles[vid])
            except KeyError:
                pass
            self.outOfRange = set()

        ex, ey = self.ego.x, self.ego.y
        vehInAoI = {}
        outOfAoI = {}
        for vk, vv in self.currVehicles.items():
            if vk == self.ego.id:
                continue
            x, y = vv.x, vv.y
            if sqrt(pow((ex - x), 2) + pow((ey - y), 2)) <= self.ego.deArea:
                try: 
                    vehArrive = vv.arriveDestination(self.netInfo)
                except:
                    vehArrive = False

                if vehArrive:
                    outOfRange.add((vk, 1))
                else:
                    vehInAoI[vk] = vv
            else:
                outOfAoI[vk] = vv

        for vid, atag in outOfRange:
            del(self.currVehicles[vid])
            # if the vehicle arrived the destination, remove it from the simulation
            if atag:
                try:
                    del(self.currVehicles[vid])
                except KeyError:
                    pass

        self.vehINAoI = vehInAoI
        self.outOfAoI = outOfAoI

    def exportScene(self):
        roadgraph = RoadGraph()

        for eid in self.edges:
            Edge = self.netInfo.getEdge(eid)
            roadgraph.edges[eid] = Edge
            for lane in Edge.lanes:
                roadgraph.lanes[lane] = self.netInfo.getLane(lane)

        for junc in self.junctions:
            Junction = self.netInfo.getJunction(junc)
            for jl in Junction.JunctionLanes:
                juncLane = self.netInfo.getJunctionLane(jl)
                roadgraph.junction_lanes[juncLane.id] = juncLane

        # export vehicles' information using dict.
        vehicles = {
            'egoCar': self.ego.export2Dict(self.netInfo),
            'carInAoI': [av.export2Dict(self.netInfo) for av in self.vehINAoI.values()],
            'outOfAoI': [sv.export2Dict(self.netInfo) for sv in self.outOfAoI.values()]
        }

        return roadgraph, vehicles

    def plotScene(self, node: dpg.node, ex: float, ey: float, ctf: CoordTF):
        if self.edges:
            for ed in self.edges:
                self.netInfo.plotEdge(ed, node, ex, ey, ctf)

        if self.junctions:
            for jc in self.junctions:
                self.netInfo.plotJunction(jc, node, ex, ey, ctf)
