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


class LocalScene:
    def __init__(self, netInfo: NetworkBuild, localPos: DummyVehicle) -> None:
        self.netInfo = netInfo
        self.localPos = localPos
        self.edges, self.junctions = self.getRoadGraph()
        self.currVehicles: dict[str, Vehicle] = {}
        self.vehINAoI: dict[str, Vehicle] = {}
        self.outOfAoI: dict[str, Vehicle] = {}

    def getRoadGraph(self) -> tuple[set[str]]:
        ex, ey = self.localPos.x, self.localPos.y
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

        edges: set = set()
        juncs: set = set()

        for sgh in sceGeohashIDs:
            try:
                geohash = self.netInfo.geoHashes[sgh]
            except KeyError:
                continue
            edges = edges | geohash.edges
            juncs = juncs | geohash.junctions

        return edges, juncs

    def updateScene(self, dataQue: Queue, timeStep: int):
        NowTLs = {}
        for jid in self.junctions:
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

    def updateSurroundVeh(self):
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

        cx, cy = self.localPos.x, self.localPos.y
        vehInAoI = {}
        outOfAoI = {}
        outOfRange = set()
        for vk, vv in self.currVehicles.items():
            try:
                x, y = traci.vehicle.getPosition(vk)
            except TraCIException:
                outOfRange.add(vk)
                continue
            if sqrt(pow((cx - x), 2) + pow((cy - y), 2)) <= self.localPos.radius:
                try:
                    vehArrive = vv.arriveDestination(self.netInfo)
                except:
                    vehArrive = False
                if not vehArrive:
                    vehInAoI[vk] = vv
            elif sqrt(pow((cx - x), 2) + pow((cy - y), 2)) <= 2 * self.localPos.radius:
                try:
                    vehArrive = vv.arriveDestination(self.netInfo)
                except:
                    vehArrive = False
                if not vehArrive:
                    outOfAoI[vk] = vv
            else:
                vv.exitControlMode()
                outOfRange.add(vk)

        for vid in outOfRange:
            del (self.currVehicles[vid])

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
            'carInAoI': [av.export2Dict(self.netInfo) for av in self.vehINAoI.values()],
            'outOfAoI': [sv.export2Dict(self.netInfo) for sv in self.outOfAoI.values()]
        }

        return roadgraph, vehicles


class LocalSceneReplay:
    def __init__(self, netInfo: Rebuild, localPos: DummyVehicle) -> None:
        self.netInfo = netInfo
        self.localPos = localPos
        self.edges, self.junctions = self.getRoadGraph()
        self.currVehicles: dict[str, Vehicle] = {}
        self.vehINAoI: dict[str, Vehicle] = {}
        self.outOfAoI: dict[str, Vehicle] = {}
        self.outOfRange: set[str] = set()

    def getRoadGraph(self) -> tuple[set[str]]:
        ex, ey = self.localPos.x, self.localPos.y
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

        edges: set = set()
        juncs: set = set()

        for sgh in sceGeohashIDs:
            try:
                geohash = self.netInfo.geoHashes[sgh]
            except KeyError:
                continue
            edges = edges | geohash.edges
            juncs = juncs | geohash.junctions

        return edges, juncs

    def updateScene(self, dataBase: str, timeStep: int):
        NowTLs = {}
        conn = sqlite3.connect(dataBase)
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM trafficLightStates WHERE frame=%i;""" % timeStep
        )
        tlsINFO = cur.fetchall()
        if tlsINFO:
            for tls in tlsINFO:
                frame, tlid, currPhase, nextPhase, switchTime = tls
                NowTLs[tlid] = (currPhase, nextPhase, switchTime)

        cur.close()
        conn.close()

        if NowTLs:
            for jid in self.junctions:
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

    def updateSurroundVeh(self):
        for vid in self.outOfRange:
            try:
                del (self.currVehicles[vid])
            except KeyError:
                pass
        self.outOfRange = set()

        cx, cy = self.localPos.x, self.localPos.y
        vehInAoI = {}
        outOfAoI = {}
        for vk, vv in self.currVehicles.items():
            x, y = vv.x, vv.y
            if sqrt(pow((cx - x), 2) + pow((cy - y), 2)) <= self.localPos.radius:
                vehInAoI[vk] = vv
            else:
                outOfAoI[vk] = vv

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
