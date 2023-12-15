from utils.simBase import CoordTF, MapCoordTF
from simModel.common.RenderDataQueue import RenderDataQueue, VRD, RGRD, ERD, LRD, JLRD
from simModel.common.networkBuild import NetworkBuild
from utils.roadgraph import RoadGraph

from rich import print
import numpy as np 
from math import cos, pi, sin
from typing import Tuple, List, Dict
import dearpygui.dearpygui as dpg
from multiprocessing import Process


class GUI(Process):
    def __init__(
        self, queue: RenderDataQueue, 
        netBoundary: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> None:
        super().__init__()
        self.queue = queue
        self.netBoundary = netBoundary

        self.zoom_speed: float = 1.0
        self.is_dragging: bool = False
        self.old_offset = (0, 0)

    def setup(self):
        dpg.create_context()
        dpg.create_viewport(
            title="TrafficSimulator",
            width=1670, height=870)
        dpg.setup_dearpygui()

    def setup_themes(self):
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_FrameRounding, 3,
                    category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FrameBorderSize, 0.5,
                    category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowBorderSize, 0,
                    category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_color(
                    dpg.mvNodeCol_NodeBackground, (255, 255, 255)
                )

        dpg.bind_theme(global_theme)

        with dpg.theme(tag="ResumeButtonTheme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (5, 150, 18))
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered, (12, 207, 23))
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonActive, (2, 120, 10))

        with dpg.theme(tag="PauseButtonTheme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (150, 5, 18))
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered, (207, 12, 23))
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonActive, (120, 2, 10))

        with dpg.theme(tag="plot_theme_v"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, (255, 165, 2),
                    category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3,
                    category=dpg.mvThemeCat_Plots
                )

        with dpg.theme(tag="plot_theme_v_future"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line,
                    (255, 165, 2, 70),
                    category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3,
                    category=dpg.mvThemeCat_Plots
                )

        with dpg.theme(tag="plot_theme_a"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, (0, 148, 50),
                    category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3,
                    category=dpg.mvThemeCat_Plots
                )

        with dpg.theme(tag="plot_theme_a_future"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line,
                    (0, 148, 50, 70),
                    category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3,
                    category=dpg.mvThemeCat_Plots
                )

    def create_windows(self):
        with dpg.font_registry():
            default_font = dpg.add_font("simModel/common/fonts/Meslo.ttf", 18)

        dpg.bind_font(default_font)

        dpg.add_window(
            tag="MainWindow",
            label="Microscopic simulation",
            no_close=True,
            # no_collapse=True,
            # no_resize=True,
            # no_move=True
        )

        self.BGnode = dpg.add_draw_node(tag="CanvasBG", parent="MainWindow")
        dpg.add_draw_node(tag="Canvas", parent="MainWindow")

        with dpg.window(
            tag='macroMap',
            label='City-level map',
            no_close=True,
        ):
            dpg.add_draw_node(tag="mapBackground", parent="macroMap")
            dpg.add_draw_node(tag="movingScene", parent="macroMap")

    def create_handlers(self):
        with dpg.handler_registry():
            dpg.add_mouse_down_handler(callback=self.mouse_down)
            dpg.add_mouse_drag_handler(callback=self.mouse_drag)
            dpg.add_mouse_release_handler(callback=self.mouse_release)
            dpg.add_mouse_wheel_handler(callback=self.mouse_wheel)

    def resize_windows(self):
        dpg.set_item_width("MainWindow", 700)
        dpg.set_item_height("MainWindow", 700)
        dpg.set_item_pos("MainWindow", (520, 120))

        dpg.set_item_width('macroMap', 500)
        dpg.set_item_height('macroMap', 500)
        dpg.set_item_pos('macroMap', (10, 120))

    def drawMainWindowWhiteBG(self):
        pmin, pmax =  self.netBoundary
        centerx = (pmin[0] + pmax[0]) / 2
        centery = (pmin[1] + pmax[1]) / 2
        dpg.draw_rectangle(
            self.ctf.dpgCoord(pmin[0], pmin[1], centerx, centery),
            self.ctf.dpgCoord(pmax[0], pmax[1], centerx, centery),
            thickness=0,
            fill=(255, 255, 255),
            parent=self.BGnode
        )

    def mouse_down(self):
        if not self.is_dragging:
            if dpg.is_item_hovered("MainWindow"):
                self.is_dragging = True
                self.old_offset = self.ctf.offset

    def mouse_drag(self, sender, app_data):
        if self.is_dragging:
            self.ctf.offset = (
                self.old_offset[0] + app_data[1]/self.ctf.zoomScale,
                self.old_offset[1] + app_data[2]/self.ctf.zoomScale
            )

    def mouse_release(self):
        self.is_dragging = False

    def mouse_wheel(self, sender, app_data):
        if dpg.is_item_hovered("MainWindow"):
            self.zoom_speed = 1 + 0.01*app_data

    def update_inertial_zoom(self, clip=0.005):
        if self.zoom_speed != 1:
            self.ctf.dpgDrawSize *= self.zoom_speed
            self.zoom_speed = 1+(self.zoom_speed - 1) / 1.05
        if abs(self.zoom_speed - 1) < clip:
            self.zoom_speed = 1

    def plotVehicle(self, node, ex: float, ey: float, vtag: str, vrd: VRD):
        rotateMat = np.array(
            [
                [cos(vrd.yaw), -sin(vrd.yaw)],
                [sin(vrd.yaw), cos(vrd.yaw)]
            ]
        )
        vertexes = [
            np.array([[vrd.length/2], [vrd.width/2]]),
            np.array([[vrd.length/2], [-vrd.width/2]]),
            np.array([[-vrd.length/2], [-vrd.width/2]]),
            np.array([[-vrd.length/2], [vrd.width/2]])
        ]
        rotVertexes = [np.dot(rotateMat, vex) for vex in vertexes]
        relativeVex = [
            [vrd.x+rv[0]-ex, vrd.y+rv[1]-ey] for rv in rotVertexes
        ]
        drawVex = [
            [
                self.ctf.zoomScale*(self.ctf.drawCenter+rev[0]+self.ctf.offset[0]),
                self.ctf.zoomScale*(self.ctf.drawCenter-rev[1]+self.ctf.offset[1])
            ] for rev in relativeVex
        ]
        if vtag == 'ego':
            vcolor = (211, 84, 0)
        elif vtag == 'AoI':
            vcolor = (41, 128, 185)
        else:
            vcolor = (99, 110, 114)

        dpg.draw_polygon(drawVex, color=vcolor, fill=vcolor, parent=node)
        dpg.draw_text(
            self.ctf.dpgCoord(vrd.x, vrd.y, ex, ey),
            vrd.id,
            color=(0, 0, 0),
            size=20,
            parent=node
        )

    def plotdeArea(self, node, egoVRD: VRD, ex: float, ey: float):
        cx, cy = self.ctf.dpgCoord(egoVRD.x, egoVRD.y, ex, ey)
        try:
            dpg.draw_circle(
                (cx, cy),
                self.ctf.zoomScale * egoVRD.deArea,
                thickness=2,
                fill=(243, 156, 18),
                parent=node
            )
        except Exception as e:
            raise e

    def plotTrajectory(self, node, ex: float, ey: float, vrd: VRD):
        tps = [
            self.ctf.dpgCoord(
                vrd.trajectoryXQ[i],
                vrd.trajectoryYQ[i],
                ex, ey
            ) for i in range(len(vrd.trajectoryXQ))
        ]
        dpg.draw_polyline(
            tps, color=(205, 132, 241),
            parent=node, thickness=2
        )

    def drawVehicles(
        self, node, VRDDict:Dict[str, List[VRD]], ex: float, ey: float
    ):
        egoVRD = VRDDict['egoCar'][0]
        self.plotVehicle(node, ex, ey, 'ego', egoVRD)
        self.plotdeArea(node, egoVRD, ex, ey)
        if egoVRD.trajectoryXQ:
            self.plotTrajectory(node, ex, ey, egoVRD)
        for avrd in VRDDict['carInAoI']:
            self.plotVehicle(node, ex, ey, 'AoI', avrd)
            if avrd.trajectoryXQ:
                self.plotTrajectory(node, ex, ey, avrd)
        for svrd in VRDDict['outOfAoI']:
            self.plotVehicle(node, ex, ey, 'other', svrd)

    
    def get_line_tf(self, line: List[float], ex, ey) -> List[float]:
        return [
            self.ctf.dpgCoord(wp[0], wp[1], ex, ey) for wp in line
        ]
    

    def drawLane(self, node, lrd: LRD, ex, ey, flag: int):
        if flag & 0b10:
            return
        else:
            left_bound_tf = self.get_line_tf(lrd.left_bound, ex, ey)
            dpg.draw_polyline(
                left_bound_tf, color=(0, 0, 0, 100), 
                thickness=2, parent=node
            )

    def drawEdge(self, node, erd: ERD, rgrd: RGRD, ex, ey):
        for lane_index in range(erd.num_lanes):
            lane_id = erd.id + '_' + str(lane_index)
            lrd = rgrd.get_lane_by_id(lane_id)
            flag = 0b00
            if  lane_index == 0:
                flag += 1
                right_bound_tf = self.get_line_tf(lrd.right_bound, ex, ey)
            if lane_index == erd.num_lanes - 1:
                flag += 2
                left_bound_tf = self.get_line_tf(lrd.left_bound, ex, ey)
            self.drawLane(node, lrd, ex, ey, flag)

        left_bound_tf.reverse()
        right_bound_tf.extend(left_bound_tf)
        right_bound_tf.append(right_bound_tf[0])
        dpg.draw_polygon(
            right_bound_tf, color=(0, 0, 0),
            thickness=2, fill=(0, 0, 0, 30), parent=node
        )

    def drawJunctionLane(self, node, jlrd: JLRD, ex, ey):
        if jlrd.center_line:
            center_line_tf = self.get_line_tf(jlrd.center_line, ex, ey)
            if jlrd.currTlState:
                if jlrd.currTlState == 'r':
                    # jlColor = (232, 65, 24)
                    jlColor = (255, 107, 129, 100)
                elif jlrd.currTlState == 'y':
                    jlColor = (251, 197, 49, 100)
                elif jlrd.currTlState == 'g' or jlrd.currTlState == 'G':
                    jlColor = (39, 174, 96, 50)
            else:
                jlColor = (0, 0, 0, 30)
            dpg.draw_polyline(
                center_line_tf, color=jlColor,
                thickness=17,  parent=node
            )
        

    def drawRoadgraph(self, node, rgrd: RGRD, ex, ey):
        for eid, erd in rgrd.edges.items():
            self.drawEdge(node, erd, rgrd, ex, ey)

        for jlid, jlrd in rgrd.junction_lanes.items():
            self.drawJunctionLane(node, jlrd, ex, ey)


    def render_loop(self):
        self.update_inertial_zoom()
        self.drawMainWindowWhiteBG()
        dpg.delete_item("Canvas", children_only=True)
        dpg.delete_item("movingScene", children_only=True)
        canvasNode = dpg.add_draw_node(parent="Canvas")
        movingSceNode = dpg.add_draw_node(parent='movingScene')
        try:
            roadgraphRenderData, VRDDict = self.queue.get()
            egoVRD = VRDDict['egoCar'][0]
            ex = egoVRD.x
            ey = egoVRD.y
            self.drawRoadgraph(canvasNode, roadgraphRenderData, ex, ey)
            self.drawVehicles(canvasNode, VRDDict, ex, ey)
            # self.drawMovingSce(movingSceNode, egoVRD)
        except TypeError:
            return

    def run(self):
        self.setup()
        self.create_windows()
        self.create_handlers()
        self.resize_windows()
        self.ctf = CoordTF(120, 'MainWindow')
        dpg.show_viewport()
        # self.drawMapBG()
        while dpg.is_dearpygui_running():
            self.render_loop()
            dpg.render_dearpygui_frame()