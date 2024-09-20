import itertools
import numpy as np
import heapq
import numpy as np
from scipy.spatial import distance
from lxml import etree
from simModel.common.opendriveparser import parse_opendrive
from simModel.common.opendriveparser.elements.road import Road
from simModel.common.opendriveparser.elements.roadLanes import Lane, LaneSection
from simModel.common.opendriveparser.elements.roadPlanView import Bezier
from simModel.common.opendriveparser.elements.junction import Junction
from simModel.common.dataQueue import NetInfo, JunctionInfo, LaneInfo
from simModel.common.geom.Vector3D import Vector3D
from trafficManager.trafficLight import TrafficLight


class geoHash:
    def __init__(self, id: tuple[int]) -> None:
        """
        Initializes a geoHash object with a unique identifier and empty sets for lane sections and junctions.
        Determines the range of the GUI screen rendering area.
        """
        self.id = id
        self.laneSections: set[str] = set()
        self.junctions: set[str] = set()


class NetworkBuild:
    def __init__(self, networkFile: str) -> None:
        self.networkFile = networkFile
        self.roads: dict[str, Road] = {}  # roads in the road network
        self.lanes: dict[str, Lane] = {}  # lanes in the road network
        self.junctions: dict[int, Junction] = {}  # junctions in the road network
        self.roadLanes: dict[str, str] = {}  # lanes in each road
        self.junctionLanes: dict[str, list[str]] = {}  # lanes in each junction
        self.geoHashes: dict[tuple[int], geoHash] = {}  # area hashes
        self.tls: dict[(str, str), TrafficLight] = {}  # traffic light for junctionlanes
        self.drivingRoad: dict[str, list[str]] = {}  # adjacent roads of each road
        self.boundries: list[float] = []  # boudries of x, y to plot map
        self.precision = 2  # scatter spacing for road line (meter)
        self.plotInfo = None

    def getData(self):
        parser = etree.XMLParser()
        rootNode = etree.parse(self.networkFile, parser).getroot()
        roadNetwork = parse_opendrive(rootNode)
        print("The opendrive file has been pasred.")
        self.roads = roadNetwork.roads
        self.junctions = roadNetwork.junctions
        self.getWaypoint()
        print("The road waypoint has been got.")
        self.getLaneTopology()
        self.getRoadTopology()
        self.getDrivingRoad()
        print("The road network topology has been built.")
        self.getBoundry()
        self.getJunctionBoundry()
        self.getRoadBoundry()
        self.getPlotInfo()
        print("The road boundary has been got.")
        print("The network information have been loaded.")

    def affGridIDs(self, centerLine: list[tuple[float]]) -> set[tuple[int]]:
        """
        Returns a set of grid IDs associated with the given center line.

        The function takes a list of 2D points representing the center line and
        calculates the corresponding grid IDs by hashing the x and y coordinates.
        """
        affGridIDs = set()
        for poi in centerLine:
            poixhash = int(poi[0] // 100)
            poiyhash = int(poi[1] // 100)
            affGridIDs.add((poixhash, poiyhash))
        return affGridIDs

    def getWaypoint(self):
        """
        Retrieves the waypoints for all lanes in all roads.
        This function iterates over each road, lane section, and lane to calculate the waypoints.
        It takes into account the lane offsets, widths, and the road's plan view to generate the waypoints.
        The function returns no value, but instead updates the lane sections and lanes with the calculated waypoints.
        """
        for road in self.roads.values():
            scf_whole_road = 0
            for lane_section_idx, lane_section in enumerate(road.lanes.laneSections):
                left_type_list = []
                for lane in lane_section.leftLanes:
                    left_type_list.append(lane.type)

                right_type_list = []
                for lane in lane_section.rightLanes:
                    right_type_list.append(lane.type)
                if len(road.lanes.laneOffsets) > 0:
                    offsets = road.lanes.laneOffsets[lane_section_idx]
                    a_of, b_of, c_of, d_of = offsets.a, offsets.b, offsets.c, offsets.d
                else:
                    a_of, b_of, c_of, d_of = [0.0, 0.0, 0.0, 0.0]

                scf = 0
                center_waypoints = []
                left_waypoints = []
                for i in range(len(lane_section.leftLanes) + 1):
                    left_waypoints.append([])
                right_waypoints = []
                for i in range(len(lane_section.rightLanes) + 1):
                    right_waypoints.append([])

                while (
                    scf <= lane_section.length
                    and scf_whole_road <= road.planView.getLength()
                ):
                    # yaw is the forward vecot of the current point
                    pos, yaw = road.planView.calc(scf_whole_road)
                    leftVector = Vector3D()
                    leftVector.x, leftVector.y = np.cos(yaw), np.sin(yaw)
                    leftVector.rotation2D(np.pi / 2)
                    pos[0] += leftVector.x * (
                        a_of + scf * b_of + pow(scf, 2) * c_of + pow(scf, 3) * d_of
                    )
                    pos[1] += leftVector.y * (
                        a_of + scf * b_of + pow(scf, 2) * c_of + pow(scf, 3) * d_of
                    )
                    center_waypoints.append(pos.tolist())
                    left_waypoints[0].append(pos.tolist())
                    right_waypoints[0].append(pos.tolist())
                    # left driving lanes
                    current_idx = 1
                    if len(lane_section.leftLanes) > 0:
                        for idxLane in range(len(lane_section.leftLanes)):
                            x_prev_idx, y_prev_idx = left_waypoints[current_idx - 1][-1]
                            curLane = lane_section.leftLanes[idxLane]
                            currentLane_width = curLane.widths[0]
                            a_width, b_width = (
                                currentLane_width.a,
                                currentLane_width.b,
                            )
                            c_width, d_width = (
                                currentLane_width.c,
                                currentLane_width.d,
                            )
                            x = x_prev_idx + leftVector.x * (
                                a_width
                                + scf * b_width
                                + pow(scf, 2) * c_width
                                + pow(scf, 3) * d_width
                            )
                            y = y_prev_idx + leftVector.y * (
                                a_width
                                + scf * b_width
                                + pow(scf, 2) * c_width
                                + pow(scf, 3) * d_width
                            )

                            left_waypoints[current_idx].append([x, y])
                            current_idx += 1

                    current_idx = 1
                    if len(lane_section.rightLanes) > 0:
                        for idxLane in range(len(lane_section.rightLanes)):
                            x_prev_idx, y_prev_idx = right_waypoints[current_idx - 1][
                                -1
                            ]
                            currentLane_width = lane_section.rightLanes[idxLane].widths[
                                0
                            ]
                            a_width, b_width = (
                                currentLane_width.a,
                                currentLane_width.b,
                            )
                            c_width, d_width = (
                                currentLane_width.c,
                                currentLane_width.d,
                            )
                            x = x_prev_idx - leftVector.x * (
                                a_width
                                + scf * b_width
                                + pow(scf, 2) * c_width
                                + pow(scf, 3) * d_width
                            )
                            y = y_prev_idx - leftVector.y * (
                                a_width
                                + scf * b_width
                                + pow(scf, 2) * c_width
                                + pow(scf, 3) * d_width
                            )

                            right_waypoints[current_idx].append([x, y])
                            current_idx += 1

                    scf += self.precision
                    scf_whole_road += self.precision

                    # ensure that the maximum length can be obtained
                    if (
                        scf >= lane_section.length
                        and scf - self.precision < lane_section.length - 1e-6
                    ):
                        scf = lane_section.length
                    if (
                        scf_whole_road >= road.planView.getLength()
                        and scf_whole_road - self.precision
                        < road.planView.getLength() - 1e-6
                    ):
                        scf_whole_road = road.planView.getLength()

                for i in range(len(lane_section.leftLanes)):
                    lane = lane_section.leftLanes[i]
                    lane.setBorders(left_waypoints[i + 1], "left")
                    lane.setBorders(left_waypoints[i], "right")
                    lane.centerLine = []
                    for j in range(len(left_waypoints[i])):
                        x1, y1 = left_waypoints[i + 1][j]
                        x2, y2 = left_waypoints[i][j]
                        lane.centerLine.append(((x1 + x2) / 2, (y1 + y2) / 2))
                    # print(lane.leftBorders, lane.rightBorders)
                for i in range(len(lane_section.rightLanes)):
                    lane = lane_section.rightLanes[i]
                    lane.setBorders(right_waypoints[i + 1], "right")
                    lane.setBorders(right_waypoints[i], "left")
                    lane.centerLine = []
                    for j in range(len(right_waypoints[i])):
                        x1, y1 = right_waypoints[i + 1][j]
                        x2, y2 = right_waypoints[i][j]
                        lane.centerLine.append(((x1 + x2) / 2, (y1 + y2) / 2))
                lane_section.centerLine = center_waypoints
                laneAffGridIDs = self.affGridIDs(lane_section.centerLine)
                lane_section.affGridIDs = lane_section.affGridIDs | laneAffGridIDs
                for gridID in lane_section.affGridIDs:
                    try:
                        geohash = self.geoHashes[gridID]
                    except KeyError:
                        geohash = geoHash(gridID)
                        self.geoHashes[gridID] = geohash
                    geohash.laneSections.add(lane_section.idx)

    def getBoundry(self):
        """
        Retrieves and calculates the boundary points of the network based on the start positions of all road geometries.

        The function iterates through all road geometries, collects their start positions, and then calculates the minimum and maximum x and y coordinates.
        It then uses these values to determine the network's width, offset, and boundary points.

        """
        AllStartPoints = []
        for road in self.roads.values():
            geometries = road.planView.geometries
            for geometry in geometries:
                AllStartPoints.append(geometry.getStartPosition())
        self.margin = 30
        min_x = min(AllStartPoints, key=lambda point: point[0])[0] - self.margin
        min_y = min(AllStartPoints, key=lambda point: point[1])[1] - self.margin
        max_x = max(AllStartPoints, key=lambda point: point[0])[0] + self.margin
        max_y = max(AllStartPoints, key=lambda point: point[1])[1] + self.margin
        self._world_width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)
        self.boundries = [min_x, min_y, max_x, max_y]

    def getRoadBoundry(self):
        """
        Retrieves and calculates the boundary points of each road in the network.

        This function iterates through all roads without junctions, collects their lane sections,
        and then determines the left and right boundary points for each road.
        """
        for road in self.roads.values():
            if road.junction is None:
                for lansection in road.lanes.laneSections:
                    leftLanes = [
                        lane for lane in lansection.leftLanes if lane.type == "driving"
                    ]
                    rightLanes = [
                        lane for lane in lansection.rightLanes if lane.type == "driving"
                    ]
                    if leftLanes:
                        leftestLane = max(leftLanes, key=lambda lane: lane.id_int)
                        left_bound_ls = leftestLane.rightBorders[:]
                    else:
                        left_bound_ls = lansection.centerLine[:][::-1]
                    if rightLanes:
                        rightestLane = min(rightLanes, key=lambda lane: lane.id_int)
                        right_bound_ls = rightestLane.rightBorders[:]
                    else:
                        right_bound_ls = lansection.centerLine[:]

            if len(left_bound_ls) > 0 and len(right_bound_ls) > 0:
                right_bound_ls.extend(left_bound_ls)
                right_bound_ls.append(right_bound_ls[0])
            road.boundary = right_bound_ls

    def getJunctionBoundry(self):
        """
        This function calculates the boundary points of a junction by finding the convex hull of the points
        formed by the left and right borders of the lanes in the junction. It then filters out the points
        that are not on the boundary and orders them sequentially.
        """
        for junction in self.junctions.values():
            if junction.id not in self.junctionLanes:
                continue
            lines = []
            for laneId in self.junctionLanes[junction.id]:
                lines.append(self.lanes[laneId].leftBorders)
                lines.append(self.lanes[laneId].rightBorders)

            points = list(itertools.chain(*lines))
            points = np.array([list(p) for p in points])
            import matplotlib.pyplot as plt
            from scipy.spatial import ConvexHull

            hull = ConvexHull(points)
            boundary_indices = hull.vertices
            boundary_points = points[boundary_indices]

            def getBoundaryIndex(points, boundary_points):
                boundary_index = [[1, 1, 1, 1] for _ in points]
                for i in range(len(points)):
                    point = points[i]
                    for boundary_point in boundary_points:
                        if (
                            boundary_point[0] < point[0]
                            and boundary_point[1] < point[1]
                        ):
                            boundary_index[i][0] = 0
                        if (
                            boundary_point[0] > point[0]
                            and boundary_point[1] > point[1]
                        ):
                            boundary_index[i][1] = 0
                reversed_points = [[p[0], -p[1]] for p in points]
                reversed_boundary_point = [[p[0], -p[1]] for p in boundary_points]
                for i in range(len(reversed_points)):
                    point = reversed_points[i]
                    for boundary_point in reversed_boundary_point:
                        if (
                            boundary_point[0] < point[0]
                            and boundary_point[1] < point[1]
                        ):
                            boundary_index[i][2] = 0
                        if (
                            boundary_point[0] > point[0]
                            and boundary_point[1] > point[1]
                        ):
                            boundary_index[i][3] = 0
                boundary_index = [sum(i) for i in boundary_index]
                boundary_indices = np.where(np.array(boundary_index) >= 1)[0]
                new_boundary_points = points[boundary_indices]
                return new_boundary_points

            # run the following function twice
            boundary_points = getBoundaryIndex(points, boundary_points)
            boundary_points = getBoundaryIndex(points, boundary_points)

            def find_nearest_points_sequential(points, current_point_array=None):
                if current_point_array is None:
                    current_point_array = np.array([points[0]])
                points_array = np.array(points)
                result = []
                while len(points_array) > 0:
                    current_point_array = np.array([current_point_array]).reshape(-1, 2)
                    distances = distance.cdist(current_point_array, points_array)
                    nearest_index = np.argmin(distances)
                    nearest_point = points_array[nearest_index]
                    result.append(nearest_point.tolist())
                    current_point_array = nearest_point
                    points_array = np.delete(points_array, nearest_index, axis=0)
                return result

            junction.boundary = find_nearest_points_sequential(boundary_points)
            junction.boundary.append(junction.boundary[0])

    def getRoadTopology(self):
        """
        Retrieves and constructs the road topology by analyzing the lanes and their connections.

        Iterates through each road, lane section, and lane to identify driving lanes with next lane connections.
        The topology is stored in a dictionary with road ID and direction as keys, and a list of connected lane IDs and directions as values.
        """
        self.roadTopology = {}
        for road in self.roads.values():
            for laneSection in road.lanes.laneSections:
                for lane in laneSection.leftLanes + laneSection.rightLanes:
                    if lane.type == "driving" and lane.next != []:
                        for lane_next in lane.next:
                            direction = 1 if lane.direction > 0 else -1
                            if (road.id, direction) not in self.roadTopology:
                                self.roadTopology[(road.id, direction)] = []
                            if (
                                self.lanes[lane_next.id].roadId,
                                self.lanes[lane_next.id].direction,
                            ) not in self.roadTopology[(road.id, direction)]:
                                self.roadTopology[(road.id, direction)].append(
                                    (
                                        self.lanes[lane_next.id].roadId,
                                        self.lanes[lane_next.id].direction,
                                    )
                                )

    def getDrivingRoad(self):
        """
        Retrieves the driving roads by analyzing the road topology and identifying
        the roads that are connected to each other.

        Iterates through each road and its connections to determine the inroads and
        outroads. The driving roads are then constructed by combining the inroads,
        outroads, and the current road.
        """
        for roadId in self.roads.keys():
            inroads, outroads = [], []
            inroads = [
                key for key, value in self.roadTopology.items() if (roadId, -1) in value
            ]
            inroads.extend(
                [
                    key
                    for key, value in self.roadTopology.items()
                    if (roadId, 1) in value
                ]
            )
            if (roadId, -1) in self.roadTopology.keys():
                outroads = self.roadTopology[(roadId, -1)].copy()
            if (roadId, 1) in self.roadTopology.keys():
                outroads.extend(self.roadTopology[(roadId, 1)])
            self.drivingRoad[roadId] = [item[0] for item in inroads + outroads] + [
                roadId
            ]

    def getLaneTopology(self):
        """
        This function generates the lane topology for a given road network. It processes each road and its lanes,
        setting their IDs, directions, and borders. It also establishes connections between lanes at junctions,
        creating new roads and lanes as necessary. Finally, it updates the geometry of lanes at junctions to ensure
        smooth connections.
        """
        for road in self.roads.values():
            if road.junction == None:
                for laneSection in road.lanes.laneSections:
                    for lane in laneSection.leftLanes + laneSection.rightLanes:
                        if lane.type == "driving":
                            laneId = (
                                str(road.id)
                                + "_"
                                + str(laneSection.idx)
                                + "_"
                                + str(lane.id_int)
                            )
                            if lane.id_int > 0:
                                # opendrive follows the left-hand traffic rule
                                lane.direction = 1
                                temp_left = lane.leftBorders.copy()
                                lane.setBorders(lane.rightBorders[::-1], "left")
                                lane.setBorders(temp_left[::-1], "right")
                                lane.centerLine = lane.centerLine[::-1]
                            else:
                                lane.direction = -1
                            lane.id = laneId
                            lane.roadId = road.id
                            lane.length = laneSection.length
                            lane.laneSection_idx = laneSection.idx
                            self.lanes[laneId] = lane
                            if road.id not in self.roadLanes:
                                self.roadLanes[road.id] = []
                            self.roadLanes[road.id].append(laneId)

        for lane1 in self.lanes.values():
            for lane2 in self.lanes.values():
                if lane1.roadId != lane2.roadId:
                    if (
                        lane1.centerLine
                        and lane2.centerLine
                        and (
                            np.hypot(
                                lane1.centerLine[-1][0] - lane2.centerLine[0][0],
                                lane1.centerLine[-1][1] - lane2.centerLine[0][1],
                            )
                            < 1
                        )
                    ):
                        lane1.next.append(lane2)
                        lane2.last.append(lane1)
                elif (
                    lane1.roadId == lane2.roadId
                    and lane1.laneSection_idx == lane2.laneSection_idx
                ):
                    if lane1.id_int < 0 and lane1.id_int == lane2.id_int + 1:
                        lane1.right = lane2
                        lane2.left = lane1
                    elif lane1.id_int > 0 and lane1.id_int == lane2.id_int + 1:
                        lane1.left = lane2
                        lane2.right = lane1

        for junction in self.junctions.values():
            if junction.id not in self.junctionLanes:
                self.junctionLanes[junction.id] = []
            inlanes, outlanes = [], []
            for connection in junction.connections:
                predecessor = self.roads[
                    str(connection.connectingRoad)
                ].link.predecessor
                predecessorRoad = self.roads[str(predecessor.elementId)]
                successor = self.roads[str(connection.connectingRoad)].link.successor
                successorRoad = self.roads[str(successor.elementId)]

                if predecessor.contactPoint == "start":
                    laneSection = predecessorRoad.lanes.laneSections[0]
                    for lane in laneSection.leftLanes:
                        if lane.type == "driving" and lane not in inlanes:
                            inlanes.append(lane)
                    for lane in laneSection.rightLanes:
                        if lane.type == "driving" and lane not in outlanes:
                            outlanes.append(lane)
                else:
                    laneSection = predecessorRoad.lanes.laneSections[-1]
                    for lane in laneSection.rightLanes:
                        if lane.type == "driving" and lane not in inlanes:
                            inlanes.append(lane)
                    for lane in laneSection.leftLanes:
                        if lane.type == "driving" and lane not in outlanes:
                            outlanes.append(lane)

                if successor.contactPoint == "start":
                    laneSection = successorRoad.lanes.laneSections[0]
                    for lane in laneSection.leftLanes:
                        if lane.type == "driving" and lane not in inlanes:
                            inlanes.append(lane)
                    for lane in laneSection.rightLanes:
                        if lane.type == "driving" and lane not in outlanes:
                            outlanes.append(lane)
                else:
                    laneSection = successorRoad.lanes.laneSections[-1]
                    for lane in laneSection.rightLanes:
                        if lane.type == "driving" and lane not in inlanes:
                            inlanes.append(lane)
                    for lane in laneSection.leftLanes:
                        if lane.type == "driving" and lane not in outlanes:
                            outlanes.append(lane)

            cnt = 0
            for inlane in inlanes:
                for outlane in outlanes:
                    if inlane.roadId != outlane.roadId:
                        # create a new road
                        connectingRoad = Road()
                        connectingRoad.id = "j" + junction.id + "-" + str(cnt)
                        newLaneSection = LaneSection()
                        newLaneSection.idx = 0
                        connectingRoad.lanes.laneSections.append(newLaneSection)
                        self.roads[connectingRoad.id] = connectingRoad
                        connectingRoad.junction = junction.id
                        # create a new connection lane
                        connectionLane = Lane()
                        connectionLane.type = "driving"
                        connectionLane.id_int = -1
                        connectionLane.direction = -1
                        connectionLane.id = (
                            connectingRoad.id
                            + "_"
                            + str(laneSection.idx)
                            + "_"
                            + str(connectionLane.id_int)
                        )
                        connectionLane.roadId = connectingRoad.id
                        newLaneSection.rightLanes.append(connectionLane)
                        self.lanes[connectionLane.id] = connectionLane
                        connectionLane.last = [inlane]
                        connectionLane.next = [outlane]
                        inlane.next.append(connectionLane)
                        outlane.last.append(connectionLane)
                        self.junctionLanes[junction.id].append(connectionLane.id)
                        self.lanes[connectionLane.id] = connectionLane
                        cnt += 1
            if self.junctionLanes[junction.id] == []:
                del self.junctionLanes[junction.id]

        for lane in self.lanes.values():
            if lane.id[0] == "j":
                # the numbers of last lanes and next lanes for junction lane are only 1
                inline_left = lane.last[0].leftBorders[-2::]
                inline_right = lane.last[0].rightBorders[-2::]
                inline_center = lane.last[0].centerLine[-2::]
                outline_left = lane.next[0].leftBorders[0:2]
                outline_right = lane.next[0].rightBorders[0:2]
                outline_center = lane.next[0].centerLine[0:2]

                road = self.roads[lane.roadId]
                road.planView.addBezier(
                    self.getControlPoints(inline_left, outline_left)
                )
                road.length = road.planView.getLength()
                num = max(int(road.length / self.precision), 10)
                leftBorders = []
                for t in np.arange(0, 1.0 + 1 / num, 1 / num):
                    scf = min(road.length, t * road.length)
                    pos, hdg = road.planView.calc(scf)
                    leftBorders.append(pos)

                right_curve = Bezier(self.getControlPoints(inline_right, outline_right))
                num = max(int(right_curve._length / self.precision), 10)
                rightBorders = []
                for t in np.arange(0, 1.0 + 1 / num, 1 / num):
                    scf = min(right_curve._length, t * right_curve._length)
                    pos, hdg = right_curve.calcPosition(scf)
                    rightBorders.append(pos)

                center_curve = Bezier(
                    self.getControlPoints(inline_center, outline_center)
                )
                num = max(int(center_curve._length / self.precision), 10)
                centerLine = []
                for t in np.arange(0, 1.0 + 1 / num, 1 / num):
                    scf = min(center_curve._length, t * center_curve._length)
                    pos, hdg = center_curve.calcPosition(scf)
                    centerLine.append(pos)

                lane.setBorders(leftBorders, "left")
                lane.setBorders(rightBorders, "right")
                lane.centerLine = centerLine
                lane.length = center_curve._length
                road.lanes.laneSections[0].centerLine = centerLine

    def getControlPoints(self, inline, outline):
        """Calculates the control points for a Bezier curve that connects two line segments."""
        dis = np.hypot(inline[1][0] - outline[0][0], inline[1][1] - outline[0][1])
        point_vector = Vector3D()
        point_vector.x, point_vector.y = (
            inline[1][0] - inline[0][0],
            inline[1][1] - inline[0][1],
        )
        control_point1 = (
            inline[1][0] + point_vector.x / point_vector.length * dis / 3,
            inline[1][1] + point_vector.y / point_vector.length * dis / 3,
        )
        point_vector = Vector3D()
        point_vector.x, point_vector.y = (
            outline[1][0] - outline[0][0],
            outline[1][1] - outline[0][1],
        )
        control_point2 = (
            outline[0][0] - point_vector.x / point_vector.length * dis / 3,
            outline[0][1] - point_vector.y / point_vector.length * dis / 3,
        )
        controlPoints = [inline[1], control_point1, control_point2, outline[0]]
        return controlPoints

    def frenet2Cartesian(self, veh, roadId, laneId, direction, scf, tcf, yaw):
        """Converts Frenet coordinates to Cartesian coordinates."""
        while scf > self.roads[roadId].length:
            road = self.roads[roadId]
            scf -= road.length
            for i in range(len(veh.route)):
                if (
                    roadId == self.lanes[veh.route[i]].roadId
                    and direction == self.lanes[veh.route[i]].direction
                ):
                    if len(veh.route) <= i + 1:
                        return
                    lane_next_id = veh.route[i + 1]
                    road_next_id = self.lanes[lane_next_id].roadId
                    next_direction = self.lanes[lane_next_id].direction
                    break
            # tcf may change in next road
            tcf = self.getTcfNextRoad(
                roadId, direction, road_next_id, next_direction, tcf, yaw
            )
            roadId, laneId, direction = road_next_id, lane_next_id, next_direction
        [x, y, hdg] = self.getPosition(roadId, direction, scf, tcf, yaw)
        if np.hypot(veh.x, veh.y) > 1 and np.hypot(x - veh.x, y - veh.y) > 100:
            pdb.set_trace()
        return roadId, laneId, direction, x, y, hdg, scf, tcf

    def getTcfNextRoad(self, roadId, direction, road_next_id, next_direction, tcf, yaw):
        """Calculates the tcf value for the next road segment."""
        # get the Cartesian coordinates of the end of current road
        road = self.roads[roadId]
        scf = road.length
        x, y, _ = self.getPosition(roadId, direction, scf, tcf, yaw)
        # get the Cartesian coordinates of the start of current road
        road_next = self.roads[road_next_id]
        scf_next = 0 if next_direction == -1 else road_next.length
        pos, hdg = road_next.planView.calc(scf_next)
        road_forward = Vector3D()
        point_vector = Vector3D()
        road_forward.x, road_forward.y = np.cos(hdg), np.sin(hdg)
        point_vector.x, point_vector.y = x - pos[0], y - pos[1]
        tcf_next = point_vector.length
        road_forward.rotation2D(np.pi / 2)
        return tcf_next

    def getPosition(self, roadId, direction, scf, tcf, yaw):
        """Calculates the position and heading of a vehicle on a road."""
        pos = None
        if direction == 1:
            scf = self.roads[roadId].length - scf
        pos, road_hdg = self.roads[roadId].planView.calc(scf)
        x, y = pos[0], pos[1]
        hdg = road_hdg - yaw

        # Returns the lateral vector based on forward vector
        leftVector = Vector3D()
        leftVector.x, leftVector.y = np.cos(road_hdg), np.sin(road_hdg)
        leftVector.rotation2D(-np.pi / 2)
        if direction == 1:
            x, y = x - leftVector.x * tcf, y - leftVector.y * tcf
            hdg += np.pi
        else:
            x, y = x + leftVector.x * tcf, y + leftVector.y * tcf
        return [x, y, hdg]

    def nearestLanes(self, x, y):
        """choose the ten nearest road"""
        nearestLanes = []
        cnt = 0
        for _, lane in enumerate(self.lanes.values()):
            for pos in lane.centerLine:
                temp = (pos[0] - x) ** 2 + (pos[1] - y) ** 2
                nearestLanes.append([temp, lane.id])
                cnt += 1
        heapq.heapify(nearestLanes)
        res = []
        for _ in range(min(10, cnt)):
            res.append(heapq.heappop(nearestLanes))
        return res

    def cartesian2Frenet(self, x, y, default_roadId=None, defaultLaneId=None):
        """Converts Cartesian coordinates to Frenet coordinates."""
        if default_roadId is not None:
            nearestLaneId = self.roadLanes[default_roadId]
        elif default_roadId is not None:
            nearestLaneId = [default_roadId]
        else:
            nearestLaneId = self.nearestLanes(x, y)
        road_forward = Vector3D()
        point_vector = Vector3D()
        for _, laneId in nearestLaneId:
            roadId = self.lanes[laneId].roadId
            road = self.roads[roadId]
            s_current = 0
            t_current = 0
            # assuming tha each step is 0.3m
            while s_current <= road.length:
                pos, yaw = road.planView.calc(s_current)
                road_forward.x, road_forward.y = np.cos(yaw), np.sin(yaw)
                point_vector.x, point_vector.y = x - pos[0], y - pos[1]
                if road_forward.dot(point_vector) < 0:
                    break
                s_current += 0.3
            t_current = point_vector.length
            # check whether point is on the right side of the road
            return (s_current, t_current, roadId)
        return None

    def createTrafficLight(self, frequency):
        """
        Create traffic lights for each junction in the network.

        This function creates traffic lights for each junction in the network. It iterates over each junction and its corresponding lanes. For each lane, it checks if the last road ID is already in the cycle set for the junction. If not, it adds the road ID to the cycle set and appends the lane ID to the corresponding road ID. Finally, it calculates the start time, green time, yellow time, and red time for each traffic light and assigns it to the corresponding junction and connecting lane.
        """
        cycleSet = {}
        for junctionId in self.junctionLanes.keys():
            cycleSet[junctionId] = {}
            for laneId in self.junctionLanes[junctionId]:
                connectingLane = self.lanes[laneId]
                if connectingLane.last[0].roadId not in cycleSet[junctionId]:
                    cycleSet[junctionId][connectingLane.last[0].roadId] = []
                cycleSet[junctionId][connectingLane.last[0].roadId].append(
                    connectingLane.id
                )
        for junctionId in self.junctionLanes.keys():
            st, gt, yt = 0, 20 * frequency, 3 * frequency
            rt = (gt + yt) * (len(cycleSet[junctionId]) - 1)
            for incomingRoad in cycleSet[junctionId]:
                for connectinglaneId in cycleSet[junctionId][incomingRoad]:
                    self.tls[(junctionId, connectinglaneId)] = TrafficLight(
                        st, gt, yt, rt
                    )
                st += gt + yt

    def getPlotInfo(self):
        """Retrieves and organizes the necessary information for plotting the network."""
        lanes = dict()
        for lane in self.lanes.values():
            lanes[lane.id] = LaneInfo(
                lane.id, lane.type, lane.leftBorders, lane.rightBorders, lane.centerLine
            )

        junctions = dict()
        for junction in self.junctions.values():
            junctions[junction.id] = JunctionInfo(
                junction.boundary, self.junctionLanes[junction.id]
            )
        self.plotInfo = NetInfo(self.boundries, lanes, junctions)
