import numpy as np
import time
import signal
import sys
from math import sin, cos
from datetime import datetime
from multiprocessing import Process
import dearpygui.dearpygui as dpg
from simModel.model import Model
from visualization.coord import CoordTF, MapCoordTF
from simModel.common.dataQueue import VehicleInfo
from simModel.common.dataQueue import LaneInfo
from visualization.gui import GUI


class PlotEngine(Process):
    def __init__(self, model: Model) -> None:
        """
        Initializes a PlotEngine instance with the given model.
        """
        super().__init__()
        self.renderQueue = model.renderQueue
        self.focusPos = model.focusPos
        self.netInfo = model.netInfo.plotInfo
        self.simPause = model.simPause
        self.mapCoordTF: MapCoordTF = None
        self.coordTF: CoordTF = None
        self.gui = GUI()
        self.ego = None
        self.simTime: float = None
        self.time_start = time.time()

    def run(self):
        """
        Runs the main loop of the PlotEngine process.

        This method sets up a signal handler for the SIGINT signal, starts the GUI, and enters a loop that continues until the DearPyGui window is closed.
        Inside the loop, it calls the `getInfoLoop` method to retrieve information from the render queue and update the GUI accordingly.
        If the `mapCoordTF` attribute is not set, it calls the `drawMapBG` method to draw the map background.
        If the `ego` attribute is set, it calls the `getSce` method to get the scene information for the ego vehicle at the current simulation time, updates the inertial zoom level of the GUI, and calls the `getNewFocusPos` method to get the new focus position.
        Finally, it renders a frame of the DearPyGui window.
        """
        signal.signal(signal.SIGINT, self.signal_handler)
        self.gui.start()
        while dpg.is_dearpygui_running():
            self.getInfoLoop()
            if not self.mapCoordTF:
                self.drawMapBG()
            if self.ego:
                self.getSce(self.ego, self.simTime)
                self.gui.update_inertial_zoom()
                self.getNewFocusPos()
                dpg.render_dearpygui_frame()

    def signal_handler(self):
        """
        Handles the SIGINT signal by printing a message indicating that the program has been interrupted and the GUI is ready to exit, then exits the program with a status code of 0.
        """
        print(
            "-" * 20,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "-" * 20,
            "\n The program is interrupted and the GUI is ready to exit... \n",
        )
        sys.exit(0)

    def getInfoLoop(self):
        """
        Retrieves information from the render queue and updates the simulation state.
        Checks if the GUI is running and resumes or pauses the simulation accordingly.
        If the render queue is not empty, it retrieves the vehicle information and simulation time,
        and updates the ego, vehicles in area of interest, vehicles out of area of interest, and traffic lights.
        Note: When the GUI rendering task is too large, the mouse click interaction may fail.
        """
        if self.gui.is_running:
            self.simPause.resumeSim()
        else:
            self.simPause.pauseSim()
            return
        if len(self.renderQueue.queue) > 0:
            self.vehInfo, self.simTime = self.renderQueue.get()
            # find the key from self.vehInfo
            self.ego = self.vehInfo.egoInfo
            self.vehInAoI = self.vehInfo.vehInAoI
            self.outOfAoI = self.vehInfo.outOfAoI
            self.tls = self.vehInfo.tls

    def getNewFocusPos(self):
        """
        Updates the ego vehicle and AoI based on the current mouse position.
        If the mouse position is available, it calculates the offset of the mouse position from the previous ego vehicle position and updates the GUI's coordinate transformation factor (ctf) offset accordingly.
        Note: When the GUI rendering task is too large, the mouse click interaction may fail.
        """
        if self.gui.mouse_pos:
            dpb_x, dpb_y = self.gui.mouse_pos
            ex, ey = self.ego.x, self.ego.y
            mouse_pos = self.gui.ctf.dpgCoordInverse(dpb_x, dpb_y, ex, ey)
            self.focusPos.setPos(mouse_pos)
            # Type 1: the network's perspective keep unchanged
            self.gui.ctf.offset = (
                self.gui.ctf.offset[0] + (mouse_pos[0] - ex),
                self.gui.ctf.offset[1] - (mouse_pos[1] - ey),
            )
            # Type 2: intilize the network's perspective with the new ego position
            # self.gui.ctf.offset = (0, 0)
            # self.gui.ctf.dpgDrawSizeInit()
            self.gui.mouse_pos = None

    def drawScene(self, ego: VehicleInfo, simTime: float):
        """
        Draws the simulation scene, including the ego vehicle, area of interest,
        trajectories, and simulation information.
        """
        ex, ey = ego.x, ego.y
        node = dpg.add_draw_node(parent="Canvas")
        self.plotScene(node, ex, ey, self.gui.ctf)
        if ego.id > 0:
            self.plotSelf(ego, "ego", node, ex, ey, self.gui.ctf)
        self.plotdeArea(ego, node, ex, ey, self.gui.ctf)
        self.plotTrajectory(ego, node, ex, ey, self.gui.ctf)
        if self.vehInAoI:
            for v1 in self.vehInAoI.values():
                self.plotSelf(v1, "AoI", node, ex, ey, self.gui.ctf)
                self.plotTrajectory(v1, node, ex, ey, self.gui.ctf)
        if self.outOfAoI:
            for v2 in self.outOfAoI.values():
                self.plotSelf(v2, "outOfAoI", node, ex, ey, self.gui.ctf)
                self.plotTrajectory(v2, node, ex, ey, self.gui.ctf)
        mvNode = dpg.add_draw_node(parent="movingScene")
        mvCenterx, mvCentery = self.mapCoordTF.dpgCoord(ex, ey)
        dpg.draw_circle(
            (mvCenterx, mvCentery),
            ego.deArea * self.mapCoordTF.zoomScale,
            thickness=0,
            fill=(243, 156, 18, 60),
            parent=mvNode,
        )
        infoNode = dpg.add_draw_node(parent="simInfo")
        dpg.draw_text(
            (5, 5),
            "Time step: %.2f s" % (simTime),
            color=(85, 230, 193),
            size=20,
            parent=infoNode,
        )
        dpg.draw_text(
            (5, 25),
            "Ego vehicle: %s" % ego.id,
            color=(75, 207, 250),
            size=20,
            parent=infoNode,
        )
        dpg.draw_text(
            (5, 45),
            "Current road: %s" % ego.roadId,
            color=(249, 202, 36),
            size=20,
            parent=infoNode,
        )

    def getSce(self, ego: VehicleInfo, simTime: float):
        """
        Retrieves and updates the simulation scene based on the provided ego vehicle information and simulation time.
        """
        dpg.delete_item("Canvas", children_only=True)
        dpg.delete_item("movingScene", children_only=True)
        dpg.delete_item("simInfo", children_only=True)
        self.drawScene(ego, simTime)
        self.plotVState(ego)
        self.plotScore(ego)

    def drawMapBG(self):
        """
        Draws the background map based on the network boundaries and junctions.
        """
        x1, y1, x2, y2 = self.netInfo.boundries
        self.mapCoordTF = MapCoordTF((x1, y1), (x2, y2), "macroMap")
        mNode = dpg.add_draw_node(parent="mapBackground")
        for jid in self.netInfo.junctions.keys():
            self.plotMapJunction(jid, mNode, self.mapCoordTF)
        self.gui.drawMainWindowWhiteBG((x1 - 100, y1 - 100), (x2 + 100, y2 + 100))

    def plotMapJunction(self, jid: str, node: dpg.node, ctf: MapCoordTF):
        """
        Plots a map junction on the simulation canvas.
        """
        junction = self.netInfo.junctions[jid]
        polyShape = [ctf.dpgCoord(p[0], p[1]) for p in junction.boundary]
        dpg.draw_polyline(polyShape, color=(255, 255, 255, 200), parent=node)
        for lane in self.netInfo.lanes.values():
            if lane.type == "driving" and lane.id[0] != "j":
                self.plotMapLane(lane, node, ctf)

    def plotMapLane(self, lane: LaneInfo, node: dpg.node, ctf: MapCoordTF):
        """
        Plots a map lane on the simulation canvas based on the provided lane information, flag, node, and coordinate transformation.
        """
        left_bound_tf = [ctf.dpgCoord(wp[0], wp[1]) for wp in lane.leftBorders]
        dpg.draw_polyline(left_bound_tf, color=(255, 255, 255), parent=node)
        right_bound_tf = [ctf.dpgCoord(wp[0], wp[1]) for wp in lane.rightBorders]
        dpg.draw_polyline(right_bound_tf, color=(255, 255, 255, 100), parent=node)

    def plotScore(self, ego: VehicleInfo):
        """
        Plots the driving score of the ego vehicle, including past and planned scores.
        """
        score_past, score_plan = ego.drivingScore
        if len(score_past) >= 20:
            x = list(range(-19, 1))
            y = list(score_past)[-20:]
        else:
            y = list(score_past)
            x = list(range(-len(y) + 1, 1))
        dpg.set_value("score_past_tag", [x, y])

        y = list(score_plan)
        x = list(range(0, len(score_plan)))
        dpg.set_value("score_plan_tag", [x, y])

    def plotVState(self, ego: VehicleInfo):
        """
        Plots the velocity and acceleration state of the ego vehicle, including past and planned states.
        """
        laneMaxSpeed = 15
        dpg.set_axis_limits("v_y_axis", 0, laneMaxSpeed)
        if len(ego.velQ) >= 20:
            vx = list(range(-19, 1))
            vy = list(ego.velQ)[-20:]
        else:
            vy = list(ego.velQ)
            vx = list(range(-len(vy) + 1, 1))
        dpg.set_value("v_series_tag", [vx, vy])

        if len(ego.accQ) >= 20:
            ax = list(range(-19, 1))
            ay = list(ego.accQ)[-20:]
        else:
            ay = list(ego.accQ)
            ax = list(range(-len(ay) + 1, 1))
        dpg.set_value("a_series_tag", [ax, ay])

        vfy = list(ego.planVelQ)
        vfx = list(range(0, len(vfy)))
        dpg.set_value("v_series_tag_future", [vfx, vfy])

        afy = list(ego.planAccQ)
        afx = list(range(0, len(afy)))
        dpg.set_value("a_series_tag_future", [afx, afy])

    def plotScene(self, node: dpg.node, ex: float, ey: float, ctf: CoordTF):
        """
        Plots a scene of the network, including lanes and junctions, based on the provided node and coordinate transformation.
        """
        for lane in self.netInfo.lanes.values():
            self.plotLane(lane, ex, ey, node, ctf)
        if self.netInfo.junctions:
            for jc in self.netInfo.junctions:
                self.plotJunction(jc, ex, ey, node, ctf, self.tls)

    def plotJunction(
        self, jid: str, ex: float, ey: float, node: dpg.node, ctf: CoordTF, tls
    ):
        """
        Plots a junction on a given node with the specified coordinate transformation.
        """
        junction = self.netInfo.junctions[jid]
        if junction.boundary:
            polyShape = [ctf.dpgCoord(p[0], p[1], ex, ey) for p in junction.boundary]
            dpg.draw_polygon(
                polyShape, color=(0, 0, 0), thickness=1, fill=(0, 0, 0, 30), parent=node
            )
        if jid in self.netInfo.junctions:
            for laneId in self.netInfo.junctions[jid].junctionLanes:
                lane = self.netInfo.lanes[laneId]
                self.plotJunctionLane(
                    lane, node, ex, ey, ctf, tls[(jid, laneId)].currentState
                )

    def plotJunctionLane(
        self,
        lane: LaneInfo,
        node: dpg.node,
        ex: float,
        ey: float,
        ctf: CoordTF,
        tlsState: float,
    ):
        """
        Plots a junction lane on a given node with the specified coordinate transformation.
        """
        arrowColor = None
        centerLine = lane.centerLine

        center_line_tf = [ctf.dpgCoord(wp[0], wp[1], ex, ey) for wp in centerLine]
        if tlsState:
            if tlsState == "r":
                jlColor = (255, 23, 13, 30)
            elif tlsState == "y":
                jlColor = (230, 200, 0, 30)
            elif tlsState == "g" or tlsState == "G":
                jlColor = (50, 205, 50, 100)
                arrowColor = (0, 120, 60, 100)
        else:
            jlColor = (0, 0, 0, 30)
        if arrowColor:
            arrow_curve = center_line_tf[0:7]
            dpg.draw_polyline(arrow_curve, color=arrowColor, thickness=3, parent=node)
            dpg.draw_circle(
                center_line_tf[0],
                ctf.zoomScale * 2,
                thickness=0,
                fill=arrowColor,
                parent=node,
            )
        dpg.draw_polyline(center_line_tf, color=jlColor, thickness=8, parent=node)

    def plotLane(
        self, lane: LaneInfo, ex: float, ey: float, node: dpg.node, ctf: CoordTF
    ):
        """
        Plots a lane on a given node with the specified coordinate transformation.
        """
        if lane.type == "driving" and lane.id[0] != "j":
            left_bound_tf = [
                ctf.dpgCoord(wp[0], wp[1], ex, ey) for wp in lane.leftBorders
            ]
            dpg.draw_polyline(
                left_bound_tf, color=(0, 0, 0, 100), thickness=2, parent=node
            )
            right_bound_tf = [
                ctf.dpgCoord(wp[0], wp[1], ex, ey) for wp in lane.rightBorders
            ]
            dpg.draw_polyline(
                right_bound_tf, color=(0, 0, 0, 100), thickness=2, parent=node
            )
            left_bound_tf.reverse()
            if len(right_bound_tf) > 0:
                right_bound_tf.extend(left_bound_tf)
                right_bound_tf.append(right_bound_tf[0])
                dpg.draw_polygon(
                    right_bound_tf,
                    color=(0, 0, 0),
                    thickness=2,
                    fill=(0, 0, 0, 30),
                    parent=node,
                )

    def plotSelf(
        self,
        veh: VehicleInfo,
        vtag: str,
        node: dpg.node,
        ex: float,
        ey: float,
        ctf: CoordTF,
    ):
        """
        Plots a vehicle on a given node with the specified coordinate transformation.
        """
        rotateMat = np.array(
            [[cos(veh.hdg), -sin(veh.hdg)], [sin(veh.hdg), cos(veh.hdg)]]
        )
        vertexes = [
            np.array([[veh.length / 2], [veh.width / 2 - 0.5]]),
            np.array([[veh.length / 2], [-veh.width / 2 + 0.5]]),
            np.array([[-veh.length / 2], [-veh.width / 2]]),
            np.array([[-veh.length / 2], [veh.width / 2]]),
        ]
        rotVertexes = [np.dot(rotateMat, vex) for vex in vertexes]
        relativeVex = [[veh.x + rv[0] - ex, veh.y + rv[1] - ey] for rv in rotVertexes]
        drawVex = [
            [
                ctf.zoomScale * (ctf.drawCenter + rev[0] + ctf.offset[0]),
                ctf.zoomScale * (ctf.drawCenter - rev[1] + ctf.offset[1]),
            ]
            for rev in relativeVex
        ]
        if vtag == "ego":
            vcolor = (211, 84, 0)
        elif vtag == "AoI":
            vcolor = (41, 128, 185)
        else:
            vcolor = (99, 110, 114)
        dpg.draw_polygon(drawVex, color=vcolor, fill=vcolor, parent=node)
        dpg.draw_text(
            ctf.dpgCoord(veh.x, veh.y, ex, ey),
            # veh.id,
            str(int(veh.id)),
            color=(0, 0, 0),
            size=20,
            parent=node,
        )

    def plotTrajectory(
        self, veh: VehicleInfo, node: dpg.node, ex: float, ey: float, ctf: CoordTF
    ):
        """
        Plots a vehicle's trajectory on a given node with the specified coordinate transformation.
        """
        if veh.planXQ:
            tps = [
                ctf.dpgCoord(veh.planXQ[i], veh.planYQ[i], ex, ey)
                for i in range(len(veh.planXQ))
            ]
            dpg.draw_polyline(tps, color=(205, 132, 241), parent=node, thickness=2)

    def plotdeArea(
        self, veh: VehicleInfo, node: dpg.node, ex: float, ey: float, ctf: CoordTF
    ):
        """
        Plots a vehicle's detection area on a given node with the specified coordinate transformation.
        """
        cx, cy = ctf.dpgCoord(veh.x, veh.y, ex, ey)
        dpg.draw_circle(
            (cx, cy),
            ctf.zoomScale * veh.deArea,
            thickness=0,
            fill=(243, 156, 18, 20),
            parent=node,
        )
