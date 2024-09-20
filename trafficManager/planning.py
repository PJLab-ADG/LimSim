from trafficManager.planner import Planner
from simModel.networkBuild import NetworkBuild
from trafficManager.vehicle import Vehicle, Trajectory
import numpy as np
import heapq


def dijkstra(graph, start):
    """
    This function implements Dijkstra's algorithm to find the shortest path in a graph.

    Parameters:
    graph (dict): A dictionary representing the graph, where each key is a node and its corresponding value is another dictionary.
                  The inner dictionary's keys are the node's neighbors, and its values are the weights of the edges.
    start (node): The node from which to start the search.

    Returns:
    tuple: A tuple containing two dictionaries. The first dictionary's keys are the nodes, and its values are the shortest distances from the start node.
           The second dictionary's keys are the nodes, and its values are the previous nodes in the shortest path.
    """
    distances = {node: float("infinity") for node in graph}
    distances[start] = 0
    previous = {node: None for node in graph}
    queue = [(0, start)]
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))
    return distances, previous


def get_shortest_path(previous, start, end):
    """
    This function reconstructs the shortest path from a start node to an end node using the previous nodes dictionary.
    """
    path = []
    current_node = end
    while current_node is not None:
        path.append(current_node)
        current_node = previous[current_node]
    path.reverse()
    return path if path[0] == start else []


def routingRoad(netInfo: NetworkBuild, vehDemand: dict):
    """
    This function generates the shortest route for each vehicle in the given vehicle demand based on the network information.
    Not available in the default example.
    """
    graph = dict()
    for fromNode, toNodes in netInfo.roadTopology.items():
        graph[fromNode] = dict()
        road = netInfo.roads[fromNode[0]]
        for toNode in toNodes:
            graph[fromNode][toNode] = road.length
    inValidRoute = []
    for vehId, veh in vehDemand.items():
        fromNode = (veh.roadId, veh.direction)
        distances, previous_nodes = dijkstra(graph, fromNode)
        end_node_1 = (veh.toNode, 1)
        end_node_2 = (veh.toNode, -1)
        end_node = (
            end_node_1 if distances[end_node_1] < distances[end_node_2] else end_node_2
        )
        veh.route = get_shortest_path(previous_nodes, fromNode, end_node)
        if len(veh.route) == 0:
            inValidRoute.append(vehId)
    for vehId in inValidRoute:
        del vehDemand[vehId]


def routingLane(netInfo: NetworkBuild, vehDemand: dict):
    """
    This function generates the shortest lane route for each vehicle in the given vehicle demand based on the network information.
    """
    netInfo.graph = dict()
    for lane in netInfo.lanes.values():
        if lane.roadId not in netInfo.graph:
            netInfo.graph[lane.id] = dict()
        for lane_next in lane.next:
            netInfo.graph[lane.id][lane_next.id] = lane.length

    inValidRoute = []
    for vehId, veh in vehDemand.items():
        if veh.roadId not in netInfo.roadLanes or veh.toNode not in netInfo.roadLanes:
            inValidRoute.append(vehId)
            continue
        if not veh.tcf:
            fromLane, toLane1, toLane2 = None, None, None
            for laneId in netInfo.roadLanes[veh.roadId]:
                lane = netInfo.lanes[laneId]
                if lane.direction == veh.direction:
                    if fromLane == None or abs(lane.id_int) > abs(fromLane.id_int):
                        fromLane = lane
            if fromLane == None:
                inValidRoute.append(vehId)
                continue
            veh.laneId = fromLane.id
            veh.direction = fromLane.direction
            _, veh.tcf, _ = netInfo.cartesian2Frenet(
                fromLane.centerLine[0][0],
                fromLane.centerLine[0][1],
                defaultLaneId=veh.laneId,
            )
        for laneId in netInfo.roadLanes[veh.toNode]:
            lane = netInfo.lanes[laneId]
            if lane.direction == 1:
                if toLane1 == None or abs(lane.id_int) > abs(toLane1.id_int):
                    toLane1 = lane
            else:
                if toLane2 == None or abs(lane.id_int) > abs(toLane2.id_int):
                    toLane2 = lane
        distances, previous_nodes = dijkstra(netInfo.graph, fromLane.id)
        if toLane1 != None and toLane2 == None:
            toLane = toLane1
        elif toLane1 == None and toLane2 != None:
            toLane = toLane2
        elif toLane1 != None and toLane2 != None:
            toLane = (
                toLane1 if distances[toLane1.id] < distances[toLane2.id] else toLane2
            )
        else:
            inValidRoute.append(vehId)
            continue
        veh.route = get_shortest_path(previous_nodes, fromLane.id, toLane.id)
        if len(veh.route) == 0:
            inValidRoute.append(vehId)
    for vehId in inValidRoute:
        del vehDemand[vehId]


def planning(
    netInfo: NetworkBuild, vehDemand: dict, vehRunning: dict, t: int, frequency: int
):
    """This function generates a planning trajectory for vehicles in a network."""
    # planning order: sort by scf descend for each road
    planning_list = []
    veh_in_road = dict()
    for vehId, veh in vehRunning.items():
        if veh.roadId not in veh_in_road:
            veh_in_road[veh.roadId] = dict()
        veh_in_road[veh.roadId][veh] = veh.scf
    for roadId, vehs in veh_in_road.items():
        vehs = sorted(vehs.items(), key=lambda item: item[1], reverse=True)
        planning_list.extend([item[0] for item in vehs])
    # new arriving vehicle
    for vehId in vehDemand:
        if vehDemand[vehId] != -1:
            if t > vehDemand[vehId].arrivalTime:
                planning_list.append(vehDemand[vehId])
    for veh in planning_list:
        if not veh.planner:
            veh.planner = Planner(veh, netInfo)
        # PLAN_LEN: int, the period of planning
        # PLAN_GAP: int, the gap of planning, the gap should be not larger than PLAN_LEN
        if len(veh.planTra.states) <= (veh.PLAN_LEN - veh.PLAN_GAP) * frequency:
            trajectory = veh.planner.planning(vehRunning, frequency)
            if veh.laneId not in veh.route:
                distances, previous_nodes = dijkstra(netInfo.graph, veh.laneId)
                veh.route = get_shortest_path(previous_nodes, veh.laneId, veh.route[-1])
            # record planning trajectory
            veh.planTra = Trajectory()
            for i in range(0, len(trajectory)):
                frenet_state = trajectory[i]
                cartesian_state = veh.getState(frenet_state, netInfo)
                if cartesian_state:
                    veh.planTra.states.append(cartesian_state)
        else:
            # del last state
            veh.planTra.states.pop(0)
        if not veh.planTra:
            print("Error: planning trajectory for vehicle {} is empty".format(vehId))
            vehRunning.pop(veh.id)
            continue
        # get next status from the planning trajectory
        state = veh.planTra.states[0]
        # if arriving at a different road, replan at next timstep
        if state.laneId != veh.laneId:
            veh.planTra = Trajectory()
            # arrving at a junction and in red light
            junction = netInfo.roads[state.roadId].junction
            if (
                junction != None
                and netInfo.tls[(junction, state.laneId)].currentState != "g"
            ):
                veh.stop()
                continue
        # cosillion check
        newVehIdx = True if veh.id not in vehRunning else False
        if tra_check(veh, state, vehRunning, newVehIdx):
            veh.stop()
            veh.planTra = Trajectory()
            continue
        # arriving at to_road
        if (
            state.roadId == veh.toNode
            and state.scf >= netInfo.roads[veh.toNode].length / 2
        ):
            vehRunning.pop(veh.id)
            veh.planTra = Trajectory([])
            continue
        # update vehicle status
        veh.move(state)
        # new arriving vehicle
        # permit the vehicle to enter the network, and del from demand
        if veh.id not in vehRunning:
            vehRunning[veh.id] = veh
            vehDemand.pop(veh.id)
    return vehDemand, vehRunning


def tra_check(
    veh: Vehicle,
    status: list,
    vehicle_running: dict,
    newVehIdx: bool = False,
) -> bool:
    """
    Checks for potential collisions between a vehicle and other vehicles in the network.

    """
    for other in vehicle_running.values():
        if veh != other:
            dis = np.hypot(status.x - other.x, status.y - other.y)
            if newVehIdx and dis < 10:
                return True
            if status.roadId == other.roadId:
                if (
                    status.direction == other.direction
                    and abs(status.tcf - other.tcf) < 1
                ):
                    if status.scf <= other.scf and dis <= 10:
                        return True
            else:
                hdg_relative = abs(status.hdg - other.hdg)
                while hdg_relative > 2 * np.pi:
                    hdg_relative -= 2 * np.pi
                if hdg_relative > np.pi:
                    hdg_relative = 2 * np.pi - hdg_relative
                # other and veh are in the adjacent roads, so the hdgs are close
                if dis < 10 and hdg_relative < 1 / 3 * np.pi:
                    if (
                        np.dot(
                            (status.x - other.x, status.y - other.y),
                            (np.cos(status.hdg), np.sin(status.hdg)),
                        )
                        < 0
                    ):
                        return True
                # other and veh are in cross roads
                if dis < 10 and hdg_relative >= 1 / 3 * np.pi:
                    # they tends to be close, and the vehicle with low speed should yield
                    for i in range(len(veh.planTra)):
                        for j in range(len(other.planTra)):
                            vx = veh.planTra.xQ[i]
                            vy = veh.planTra.yQ[i]
                            ox = other.planTra.xQ[j]
                            oy = other.planTra.yQ[j]
                            if i >= j and np.hypot(vx - ox, vy - oy) < 3:
                                return True
    return False
