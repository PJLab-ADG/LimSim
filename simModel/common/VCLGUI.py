from utils.simBase import CoordTF
from simModel.common.RenderDataQueue import (
    RenderDataQueue, VRD, RGRD, ERD, LRD, JLRD, ImageQueue, DecisionQueue)


from rich import print
import numpy as np 
from math import cos, pi, sin
from typing import Tuple, List, Dict
import dearpygui.dearpygui as dpg
from multiprocessing import Process


class GUI(Process):
    def __init__(
        self, renderQueue: RenderDataQueue, 
        iamgeQueue: ImageQueue,
        decisionQueue: DecisionQueue,
        netBoundary: Tuple[Tuple[float, float], Tuple[float, float]]
    ) -> None:
        super().__init__()
        self.renderQueue = renderQueue
        self.imageQueue = iamgeQueue
        self.decisionQueue = decisionQueue
        self.netBoundary = netBoundary

        self.zoom_speed: float = 1.0
        self.is_dragging: bool = False
        self.old_offset = (0, 0)

    def setup(self):
        dpg.create_context()
        dpg.create_viewport(
            title="TrafficSimulator",
            width=1490, height=1010)
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

    def create_windows(self):
        with dpg.font_registry():
            default_font = dpg.add_font("simModel/common/fonts/Meslo.ttf", 18)

        dpg.bind_font(default_font)

        # BEV 视图窗口
        dpg.add_window(tag="MainWindow", label="Microscopic simulation")

        dpg.add_draw_node(tag="CanvasBG", parent="MainWindow")
        dpg.add_draw_node(tag="Canvas", parent="MainWindow")

        # 前视相机窗口
        texture_data = []
        for i in range(0, 700 * 500):
            texture_data.append(255 / 255)
            texture_data.append(255 / 255)
            texture_data.append(255 / 255)
            texture_data.append(255 / 255)

        self.texture_registry = dpg.add_texture_registry(show = True)
        dpg.add_dynamic_texture(
            width=700, height=500, 
            default_value=texture_data, tag='texture_tag',
            parent=self.texture_registry
        )

        with dpg.window(tag='FrontViewCamera', label='Front view camera'):
            dpg.add_image('texture_tag')

        # 辅助信息窗口
        dpg.add_window(tag="InformationWindow", label='Navigation')

        # response 窗口
        dpg.add_window(tag="ResponseWindow", label='Reasoning and decision')

    def create_handlers(self):
        with dpg.handler_registry():
            dpg.add_mouse_down_handler(callback=self.mouse_down)
            dpg.add_mouse_drag_handler(callback=self.mouse_drag)
            dpg.add_mouse_release_handler(callback=self.mouse_release)
            dpg.add_mouse_wheel_handler(callback=self.mouse_wheel)

    def resize_windows(self):
        dpg.set_item_width("MainWindow", 725)
        dpg.set_item_height("MainWindow", 725)
        dpg.set_item_pos("MainWindow", (10, 10))

        dpg.set_item_width('FrontViewCamera', 725)
        dpg.set_item_height('FrontViewCamera', 550)
        dpg.set_item_pos('FrontViewCamera', (750, 10))

        dpg.set_item_width('InformationWindow', 725)
        dpg.set_item_height('InformationWindow', 250)
        dpg.set_item_pos('InformationWindow', (10, 750))

        dpg.set_item_width('ResponseWindow', 725)
        dpg.set_item_height('ResponseWindow', 430)
        dpg.set_item_pos('ResponseWindow', (750, 570))

    def drawMainWindowWhiteBG(self):
        pmin, pmax =  self.netBoundary
        centerx = (pmin[0] + pmax[0]) / 2
        centery = (pmin[1] + pmax[1]) / 2
        dpg.draw_rectangle(
            self.ctf.dpgCoord(pmin[0], pmin[1], centerx, centery),
            self.ctf.dpgCoord(pmax[0], pmax[1], centerx, centery),
            thickness=0,
            fill=(255, 255, 255),
            parent="CanvasBG"
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
        # self.plotdeArea(node, egoVRD, ex, ey)
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
        for erd in rgrd.edges.values():
            self.drawEdge(node, erd, rgrd, ex, ey)

        for jlrd in rgrd.junction_lanes.values():
            self.drawJunctionLane(node, jlrd, ex, ey)

    def showImage(self, image_data: np.ndarray):
        image_data = image_data / 255        
        dpg.set_value('texture_tag', image_data.flatten().tolist())

    def showInformation(self, information: str):
        dpg.delete_item('informationText')
        dpg.add_text(
            information, parent='InformationWindow',
            tag='informationText', wrap=720
            )


    def showResponse(self, response: str):
        dpg.delete_item('responseText')
        dpg.add_text(
            response, parent='ResponseWindow', 
            tag='responseText', wrap=720
            )

    def render_loop(self):
        self.update_inertial_zoom()
        dpg.delete_item("Canvas", children_only=True)
        dpg.delete_item("informationCanvas", children_only=True)
        dpg.delete_item('responseCanvas', children_only=True)
        canvasNode = dpg.add_draw_node(parent="Canvas")
        try:
            roadgraphRenderData, VRDDict = self.renderQueue.get()
            egoVRD = VRDDict['egoCar'][0]
            ex = egoVRD.x
            ey = egoVRD.y
            self.drawRoadgraph(canvasNode, roadgraphRenderData, ex, ey)
            self.drawVehicles(canvasNode, VRDDict, ex, ey)
            # self.drawMovingSce(movingSceNode, egoVRD)
        except TypeError:
            return
        
        try:
            image_data = self.imageQueue.get()
            if isinstance(image_data, np.ndarray):
                self.showImage(image_data)
            else:
                return
        except TypeError:
            return
        
        try:
            information, response = self.decisionQueue.get()
            if information:
                self.showInformation(information)
                self.showResponse(response)
        except TypeError:
            return

    def run(self):
        self.setup()
        self.create_windows()
        self.create_handlers()
        self.resize_windows()
        self.ctf = CoordTF(120, 'MainWindow')
        dpg.show_viewport()
        self.drawMainWindowWhiteBG()
        # self.drawMapBG()
        while dpg.is_dearpygui_running():
            self.render_loop()
            dpg.render_dearpygui_frame()