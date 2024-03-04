# Preprocess for scenario generation
# Get each element's shape from `.net.xml`
# Get network topology from `.net.xml`

# for: NetworkBuild with Frenet
from __future__ import annotations
from utils.simBase import CoordTF, deduceEdge, MapCoordTF
from utils.cubic_spline import Spline2D
from utils.roadgraph import Junction, Edge, NormalLane, OVERLAP_DISTANCE, JunctionLane
from queue import Queue
import sqlite3
from threading import Thread
import numpy as np
import xml.etree.ElementTree as ET
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
from rich import print
from datetime import datetime
from matplotlib.patches import Polygon
from typing import Dict, List, Set

class geoHash:
    def __init__(self, id: tuple[int]) -> None:
        self.id = id
        self.edges: set[str] = set()
        self.junctions: set[str] = set()


class NetworkBuild:
    def __init__(
        self,  dataBase: str, networkFile: str,
    ) -> None:
        self.dataBase = dataBase
        self.networkFile = networkFile
        self.edges: Dict[str, Edge] = {}
        self.lanes: Dict[str, NormalLane] = {}
        self.junctions: Dict[str, Junction] = {}
        self.junctionLanes: Dict[str, JunctionLane] = {}
        self.tlJunctions: Set[str] = set()   # id of junctions which has traffic light
        self.dataQue = Queue()
        self.geoHashes: dict[tuple[int], geoHash] = {}

    def getEdge(self, eid: str) -> Edge:
        try:
            return self.edges[eid]
        except KeyError:
            return

    def getLane(self, lid: str) -> NormalLane:
        try:
            return self.lanes[lid]
        except KeyError:
            return

    def getJunction(self, jid: str) -> Junction:
        try:
            return self.junctions[jid]
        except KeyError:
            return

    def getJunctionLane(self, jlid: str) -> JunctionLane:
        try:
            return self.junctionLanes[jlid]
        except KeyError:
            return

    def affGridIDs(self, centerLine: list[tuple[float]]) -> set[tuple[int]]:
        affGridIDs = set()
        for poi in centerLine:
            poixhash = int(poi[0] // 100)
            poiyhash = int(poi[1] // 100)
            affGridIDs.add((poixhash, poiyhash))

        return affGridIDs

    def insertCommit(self):
        conn = sqlite3.connect(self.dataBase, check_same_thread=False)
        cur = conn.cursor()
        commitCnt = 0
        while not self.dataQue.empty():
            tableName, data, process = self.dataQue.get()
            sql = '{} INTO {} VALUES '.format(process, tableName) + \
                '(' + '?,'*(len(data)-1) + '?' + ')'
            try:
                cur.execute(sql, data)
            except sqlite3.OperationalError as e:
                print(sql, data)
            commitCnt += 1
            if commitCnt == 10000:
                conn.commit()
                commitCnt = 0
        conn.commit()
        cur.close()
        conn.close()

        print('[green bold]Network information commited at {}.[/green bold]'.format(
            datetime.now().strftime('%H:%M:%S.%f')[:-3]))

    def processRawShape(self, rawShape: str) -> list:
        rawList = rawShape.split(' ')
        floatShape = [list(map(float, p.split(','))) for p in rawList]
        return floatShape
    
    def getLaneAttrib(self, laneElem: ET.Element):
        try:
            laneSpeed = float(laneElem.attrib['speed'])
        except:
            laneSpeed = 13.89
        try:
            laneWidth = float(laneElem.attrib['width'])
        except KeyError:
            laneWidth = 3.2
        try:
            laneType = laneElem.attrib['type']
        except KeyError:
            laneType = ''
        try:
            laneAllow = laneElem.attrib['allow']
        except KeyError:
            laneAllow = ''
        try:
            laneDisallow = laneElem.attrib['disallow']
        except:
            laneDisallow = ''
        laneLength = float(laneElem.attrib['length'])
        return laneSpeed, laneWidth, laneType, laneAllow, laneDisallow, laneLength

    def processEdge(self, eid: str, child: ET.Element):
        if eid[0] == ':':
            for gchild in child:
                ilid = gchild.attrib['id']
                (
                    ilspeed, ilwidth, ilaneType, 
                    ilaneAllow, ilaneDisallow, ilLength
                ) = self.getLaneAttrib(gchild)
                self.junctionLanes[ilid] = JunctionLane(
                    id=ilid, width=ilwidth, speed_limit=ilspeed,
                    sumo_length=ilLength, laneType=ilaneType,
                    laneAllow=ilaneAllow, laneDisallow=ilaneDisallow
                )
                self.dataQue.put((
                    'junctionLaneINFO', (
                        ilid, ilwidth, ilspeed, ilLength, 0, 
                        ilaneType, ilaneAllow, ilaneDisallow
                    ), 'INSERT'
                ))
        else:
            fromNode = child.attrib['from']
            toNode = child.attrib['to']
            edge = Edge(id=eid, from_junction=fromNode, to_junction=toNode)
            laneNumber = 0
            for gchild in child:
                if gchild.tag == 'lane':
                    lid = gchild.attrib['id']
                    (
                        lspeed, lwidth, ltype, lallow, ldisallow, lLength
                    ) = self.getLaneAttrib(gchild)
                    rawShape = gchild.attrib['shape']
                    lshape = self.processRawShape(rawShape)
                    lane = NormalLane(
                        id=lid, width=lwidth, speed_limit=lspeed,
                        sumo_length=lLength, affiliated_edge=edge,
                        laneType=ltype, laneAllow=lallow, 
                        laneDisallow=ldisallow
                    )
                    self.dataQue.put((
                        'laneINFO', (
                            lid, rawShape, lwidth, lspeed, eid, 
                            lLength, ltype, lallow, ldisallow
                        ), 'INSERT'
                    ))
                    shapeUnzip = list(zip(*lshape))

                    # interpolate shape points for better represent shape
                    shapeUnzip = [
                        np.interp(
                            np.linspace(0, len(shapeUnzip[0])-1, 50),
                            np.arange(0, len(shapeUnzip[0])),
                            shapeUnzip[i]
                        ) for i in range(2)
                    ]
                    lane.course_spline = Spline2D(shapeUnzip[0], shapeUnzip[1])
                    lane.getPlotElem()
                    self.lanes[lid] = lane
                    edge.lanes.add(lane.id)
                    laneAffGridIDs = self.affGridIDs(lane.center_line)
                    edge.affGridIDs = edge.affGridIDs | laneAffGridIDs
                    laneNumber += 1
            edge.lane_num = laneNumber
            for gridID in edge.affGridIDs:
                try:
                    geohash = self.geoHashes[gridID]
                except KeyError:
                    geohash = geoHash(gridID)
                    self.geoHashes[gridID] = geohash
                geohash.edges.add(eid)
            self.edges[eid] = edge
            self.dataQue.put((
                'edgeINFO', (eid, laneNumber, fromNode, toNode), 'INSERT'
            ))

    def processConnection(self, child: ET.Element):
        fromEdgeID = child.attrib['from']
        fromEdge = self.getEdge(fromEdgeID)
        fromLaneIdx = child.attrib['fromLane']
        fromLaneID = fromEdgeID + '_' + fromLaneIdx
        fromLane = self.getLane(fromLaneID)
        toEdgeID = child.attrib['to']
        toLaneIdx = child.attrib['toLane']
        toLaneID = toEdgeID + '_' + toLaneIdx
        toLane = self.getLane(toLaneID)
        if fromLane and toLane:
            direction = child.attrib['dir']
            junctionLaneID = child.attrib['via']
            junctionLane = self.getJunctionLane(junctionLaneID)
            self.dataQue.put((
                'connectionINFO', (
                    fromLaneID, toLaneID, direction, junctionLaneID
                ), 'INSERT'
            ))
            if junctionLane.sumo_length < 1:
                fromLane.next_lanes[toLaneID] = (toLaneID, 's')
                fromEdge.next_edge_info[toEdgeID].add(fromLaneID)
            else:
                # junctionLane = self.getJunctionLane(junctionLaneID)
                if 'tl' in child.attrib.keys():
                    linkIndex = int(child.attrib['linkIndex'])
                    junctionLane.tlsIndex = linkIndex
                self.dataQue.put((
                    'junctionLaneINFO', (
                        junctionLane.id, junctionLane.width,
                        junctionLane.speed_limit,
                        junctionLane.sumo_length,
                        junctionLane.tlsIndex,
                        junctionLane.laneType,
                        junctionLane.laneAllow,
                        junctionLane.laneDisallow
                    ), 'REPLACE'
                ))
                center_line = []
                for si in np.linspace(
                    fromLane.course_spline.s[-1] - OVERLAP_DISTANCE,
                    fromLane.course_spline.s[-1], num=20
                ):
                    center_line.append(
                        fromLane.course_spline.calc_position(si))
                for si in np.linspace(0, OVERLAP_DISTANCE, num=20):
                    center_line.append(
                        toLane.course_spline.calc_position(si)
                    )
                junctionLane.course_spline = Spline2D(
                    list(zip(*center_line))[0], list(zip(*center_line))[1]
                )
                junctionLane.getPlotElem()
                junctionLane.last_lane_id = fromLaneID
                junctionLane.next_lane_id = toLaneID
                fromLane.next_lanes[toLaneID] = (junctionLaneID, direction)
                fromEdge.next_edge_info[toEdgeID].add(fromLaneID)
                # add this junctionLane to it's parent Junction's JunctionLanes
                fromEdge = self.getEdge(fromEdgeID)
                juncID = fromEdge.to_junction
                junction = self.getJunction(juncID)
                junctionLane.affJunc = juncID
                try:
                    jlAffGridIDs = self.affGridIDs(junctionLane.center_line)
                except ValueError:
                    print(junctionLane.id)
                    print(junctionLane.next_lane_id)
                    print(junctionLane.center_line)
                junction.affGridIDs = junction.affGridIDs | jlAffGridIDs
                junction.JunctionLanes.add(junctionLaneID)

    def getData(self):
        elementTree = ET.parse(self.networkFile)
        root = elementTree.getroot()
        for child in root:
            if child.tag == 'edge':
                eid = child.attrib['id']
                # Some useless internal lanes will be generated by the follow codes.
                self.processEdge(eid, child)
            elif child.tag == 'junction':
                jid = child.attrib['id']
                junc = Junction(jid)
                if jid[0] != ':':
                    intLanes = child.attrib['intLanes']
                    if intLanes:
                        intLanes = intLanes.split(' ')
                        for il in intLanes:
                            ilins = self.getJunctionLane(il)
                            ilins.affJunc = jid
                            junc.JunctionLanes.add(il)
                    jrawShape = child.attrib['shape']
                    juncShape = self.processRawShape(jrawShape)
                    # Add the first point to form a closed shape
                    juncShape.append(juncShape[0])
                    junc.shape = juncShape
                    self.junctions[jid] = junc
                    self.dataQue.put((
                        'junctionINFO', (jid, jrawShape), 'INSERT'
                    ))
            elif child.tag == 'tlLogic':
                self.tlJunctions.add(child.attrib['id'])
            elif child.tag == 'connection':
                # in .net.xml, the elements 'edge' come first than elements
                # 'connection', so the follow codes can work well.
                self.processConnection(child)
        for junction in self.junctions.values():
            for gridID in junction.affGridIDs:
                try:
                    geohash = self.geoHashes[gridID]
                except KeyError:
                    geohash = geoHash(gridID)
                    self.geoHashes[gridID] = geohash
                geohash.junctions.add(junction.id)

        for ghid, ghins in self.geoHashes.items():
            ghx, ghy = ghid
            ghEdges = ','.join(ghins.edges)
            ghJunctions = ','.join(ghins.junctions)
            self.dataQue.put((
                'geohashINFO',
                (ghx, ghy, ghEdges, ghJunctions), 'INSERT'
            ))

    def buildTopology(self):
        for eid, einfo in self.edges.items():
            fj = self.getJunction(einfo.from_junction)
            tj = self.getJunction(einfo.to_junction)
            fj.outgoing_edges.add(eid)
            tj.incoming_edges.add(eid)

        print('[green bold]Network building finished at {}.[/green bold]'.format(
            datetime.now().strftime('%H:%M:%S.%f')[:-3]))

        Th = Thread(target=self.insertCommit)
        Th.start()

    def plotLane(self, lane: NormalLane, ax: plt.Axes):
        lines = list(zip(*lane.left_bound))
        ax.plot(lines[0], lines[1], color='#000000', alpha=0.5, linewidth=0.5)

    def plotEdge(self, eid: str, ax: plt.Axes):
        edge = self.getEdge(eid)
        vehicle_lane = 0
        for lane_index in range(edge.lane_num):
            lane_id = edge.id + '_' + str(lane_index)
            lane = self.getLane(lane_id)
            if lane.width > 1:
                self.plotLane(lane, ax)
                if vehicle_lane == 0:
                    vehicle_lane = 1
            if vehicle_lane == 1:
                last_lane = self.getLane(edge.id + '_' + str(lane_index - 1))
                if last_lane != None:
                    self.plotLane(last_lane, ax)
                vehicle_lane = 2
        # 根据左右边界获取 edge 的封闭图形
        lane_id = edge.id + '_' + str(0)
        lane = self.getLane(lane_id)
        bbx = lane.right_bound[::]
        lane_id = edge.id + '_' + str(edge.lane_num - 1)
        lane = self.getLane(lane_id)
        bbx.extend(lane.left_bound[::-1])
        bbx.append(bbx[0])

        ax.add_patch(Polygon(bbx, closed=True, fill=True, color='#000000', alpha=0.2, linewidth=0.8))

    def plotJunctionLane(self, jlid: str, ax: plt.Axes,):
        juncLane = self.getJunctionLane(jlid)
        if juncLane and juncLane.width > 0.8:
            try:
                center_line = juncLane.center_line
            except AttributeError:
                return
            if juncLane.currTlState:
                if juncLane.currTlState == 'r':
                    jlColor = "#FF6B81"
                    alpha = 0.3
                elif juncLane.currTlState == 'y':
                    jlColor = "#FBC531"
                    alpha = 0.3
                elif juncLane.currTlState == 'g' or juncLane.currTlState == 'G':
                    jlColor = "#27AE60"
                    alpha = 0.5
            else:
                jlColor = "#000000"
                alpha = 0.2
            center_point = list(zip(*center_line))
            ax.plot(center_point[0], center_point[1], color=jlColor, alpha=alpha, linewidth=8)

    def plotJunction(self, jid: str, ax: plt.Axes):
        junction = self.getJunction(jid)
        polyShape = list(zip(*junction.shape))
        ax.plot(polyShape[0], polyShape[1], color='#000000', alpha=0.5, linewidth=0.5)
        for jl in junction.JunctionLanes:
            self.plotJunctionLane(jl, ax)


class Rebuild(NetworkBuild):
    def __init__(self,
                 dataBase: str,
                 ) -> None:
        networkFile: str = ''
        obsFile: str = ''
        super().__init__(dataBase, networkFile)

    def getData(self):
        conn = sqlite3.connect(self.dataBase)
        cur = conn.cursor()

        cur.execute('SELECT * FROM junctionINFO;')
        junctionINFO = cur.fetchall()
        if junctionINFO:
            for ji in junctionINFO:
                junctionID = ji[0]
                jrawShape = ji[1]
                juncShape = self.processRawShape(jrawShape)
                # Add the first point to form a closed shape
                juncShape.append(juncShape[0])
                junc = Junction(junctionID)
                junc.shape = juncShape
                self.junctions[junctionID] = junc

        cur.execute('SELECT * FROM edgeINFO;')
        edgeInfo = cur.fetchall()
        if edgeInfo:
            for ed in edgeInfo:
                eid, laneNumber, fromJunction, toJunction = ed
                self.edges[eid] = Edge(
                    id=eid, lane_num=laneNumber,
                    from_junction=fromJunction,
                    to_junction=toJunction
                )

        cur.execute('SELECT * FROM laneINFO;')
        laneINFO = cur.fetchall()
        if laneINFO:
            for la in laneINFO:
                (
                    lid, rawShape, lwidth, lspeed, 
                    eid, llength, ltype, lallow, ldisallow
                ) = la
                lshape = self.processRawShape(rawShape)
                lane = NormalLane(
                    id=lid, width=lwidth, speed_limit=lspeed,
                    affiliated_edge=self.getEdge(eid), sumo_length=llength,
                    laneType=ltype, laneAllow=lallow,
                    laneDisallow=ldisallow
                )
                shapeUnzip = list(zip(*lshape))
                # interpolate shape points for better represent shape
                shapeUnzip = [
                    np.interp(
                        np.linspace(0, len(shapeUnzip[0])-1, 50),
                        np.arange(0, len(shapeUnzip[0])),
                        shapeUnzip[i]
                    ) for i in range(2)
                ]
                lane.course_spline = Spline2D(shapeUnzip[0], shapeUnzip[1])
                lane.getPlotElem()
                self.lanes[lid] = lane
                self.getEdge(eid).lanes.add(lid)

        cur.execute('SELECT * FROM junctionLaneINFO;')
        JunctionLaneINFO = cur.fetchall()
        if JunctionLaneINFO:
            for jl in JunctionLaneINFO:
                (
                    jlid, jlwidth, jlspeed, jlLength, 
                    tlsIndex, jltype, jlallow, jldisallow
                ) = jl
                self.junctionLanes[jlid] = JunctionLane(
                    id=jlid, width=jlwidth,
                    speed_limit=jlspeed, 
                    sumo_length=jlLength,
                    tlsIndex=tlsIndex,
                    laneType=jltype,
                    laneAllow=jlallow,
                    laneDisallow=jldisallow
                )

        cur.execute('SELECT * FROM connectionINFO;')
        connectionINFO = cur.fetchall()
        if connectionINFO:
            for ci in connectionINFO:
                fromLaneID, toLaneID, direction, junctionLaneID = ci
                fromLane = self.getLane(fromLaneID)
                fromEdgeID = deduceEdge(fromLaneID)
                fromEdge = self.getEdge(fromEdgeID)
                junctionLane = self.getJunctionLane(junctionLaneID)
                if not junctionLane:
                    print(
                        'The JunctionLane is not found in database: ',
                        junctionLaneID
                    )
                toEdgeID = deduceEdge(toLaneID)
                if junctionLane.sumo_length < 1:
                    fromLane.next_lanes[toLaneID] = (toLaneID, 's')
                    fromEdge.next_edge_info[toEdgeID].add(fromLaneID)
                else:
                    junctionLane = self.getJunctionLane(junctionLaneID)
                    fromEdgeID = deduceEdge(fromLaneID)
                    center_line = []
                    for si in np.linspace(
                        fromLane.course_spline.s[-1] - OVERLAP_DISTANCE,
                        fromLane.course_spline.s[-1], num=20
                    ):
                        center_line.append(
                            fromLane.course_spline.calc_position(si))
                    for si in np.linspace(0, OVERLAP_DISTANCE, num=20):
                        center_line.append(
                            self.getLane(
                                toLaneID).course_spline.calc_position(si)
                        )
                    junctionLane.course_spline = Spline2D(
                        list(zip(*center_line))[0], list(zip(*center_line))[1]
                    )
                    junctionLane.getPlotElem()
                    junctionLane.last_lane_id = fromLaneID
                    junctionLane.next_lane_id = toLaneID
                    fromLane.next_lanes[toLaneID] = (
                        junctionLaneID, direction)
                    fromEdge.next_edge_info[toEdgeID].add(fromLaneID)
                    # add this junctionLane to it's parent Junction's JunctionLanes
                    fromEdge = self.getEdge(fromEdgeID)
                    junction = self.getJunction(fromEdge.to_junction)
                    junction.JunctionLanes.add(junctionLaneID)

        cur.execute('SELECT * FROM geohashINFO;')
        geohashINFO = cur.fetchall()
        if geohashINFO:
            for gi in geohashINFO:
                ghx, ghy, ghEdges, ghJunctions = gi
                ghID = (ghx, ghy)
                geohash = geoHash(ghID)
                if ghEdges:
                    geohash.edges = set(ghEdges.split(','))
                if ghJunctions:
                    geohash.junctions = set(ghJunctions.split(','))
                self.geoHashes[ghID] = geohash

        cur.close()
        conn.close()

    def buildTopology(self):
        for k, v in self.edges.items():
            fj = self.getJunction(v.from_junction)
            tj = self.getJunction(v.to_junction)
            fj.outgoing_edges.add(k)
            tj.incoming_edges.add(k)

        print('[green bold]Network building finished at {}.[/green bold]'.format(
            datetime.now().strftime('%H:%M:%S.%f')[:-3]))
