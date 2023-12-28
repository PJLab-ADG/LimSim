# GUI for vision close-loop replay
# This is no need to be a multi-process GUI

import dearpygui.dearpygui as dpg
from utils.simBase import CoordTF
from typing import Tuple
import numpy as np


class GUI:
    '''
        mode: type of simulation, available mode: `real-time-ego`, `real-time-local`,
            `replay-ego`, `replay-local`. The interactive replay has the same mode
            with real-time simulation, for example, the mode of interactive replay 
            for ego tracking should be set as `real-time-ego`. the mode of 
            interactive replay for local are should be set as `real-time-local`.
    '''

    def __init__(self, mode) -> None:
        self.mode = mode
        self.is_running = True
        self.replayDelay = 0
        self.frameIncre = 0

        self.zoom_speed: float = 1.0
        self.is_dragging: bool = False
        self.old_offset = (0, 0)

        self.setup()
        self.setup_themes()
        self.create_windows()
        self.create_handlers()
        self.resize_windows()

        self.ctf = CoordTF(120, 'MainWindow')

    def setup(self):
        dpg.create_context()
        dpg.create_viewport(
            title="TrafficSimulator",
            width=1490, height=1120)
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

    def create_windows(self):
        with dpg.font_registry():
            dpg.bind_font(dpg.add_font("simModel/common/fonts/Meslo.ttf", 20))

            dpg.add_font(
                "simModel/common/fonts/Meslo.ttf", 
                18, tag='smallFont'
            )

            dpg.add_font(
                "simModel/common/fonts/Meslo.ttf", 
                24, tag='largeFont'
            )

        # 控制面板
        with dpg.window(tag='ControlWindow', label='Menu', no_close=True):
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Pause", tag="PauseResumeButton",
                    callback=self.toggle
                )

                dpg.add_button(
                    label="Next frame", tag="NextFrameButton", 
                    callback=self.nextFrame
                )

                dpg.add_text('   |   Time delay: ', tag='DelayText')
                dpg.add_slider_float(
                    tag="DelayInput",
                    min_value=0, max_value=1,
                    default_value=0, callback=self.setDelay
                )

        dpg.bind_item_theme('PauseResumeButton', 'PauseButtonTheme')
        dpg.bind_item_font('PauseResumeButton', 'largeFont')
        dpg.bind_item_font('NextFrameButton', 'largeFont')
        dpg.bind_item_font('DelayText', 'largeFont')
        dpg.bind_item_font('DelayInput', 'largeFont')


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

        dpg.set_item_width('ControlWindow', 1465)
        dpg.set_item_height('ControlWindow', 0)
        dpg.set_item_pos('ControlWindow', (10, 1010))


    def drawMainWindowWhiteBG(self, pmin: Tuple[float], pmax: Tuple[float]):
        centerx = (pmin[0] + pmax[0]) / 2
        centery = (pmin[1] + pmax[1]) / 2
        dpg.draw_rectangle(
            self.ctf.dpgCoord(pmin[0], pmin[1], centerx, centery),
            self.ctf.dpgCoord(pmax[0], pmax[1], centerx, centery),
            thickness=0,
            fill=(255, 255, 255),
            parent='CanvasBG'
        )

    def showImage(self, image_data: np.ndarray):
        image_data = image_data / 255        
        dpg.set_value('texture_tag', image_data.flatten().tolist())

    def showInformation(self, information: str):
        dpg.delete_item('informationText')
        dpg.add_text(
            information, parent='InformationWindow',
            tag='informationText', wrap=720
            )
        dpg.bind_item_font('informationText', 'smallFont')


    def showResponse(self, response: str):
        dpg.delete_item('responseText')
        dpg.add_text(
            response, parent='ResponseWindow', 
            tag='responseText', wrap=720
            )
        dpg.bind_item_font('responseText', 'smallFont')

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

    def setDelay(self):
        self.replayDelay = dpg.get_value('DelayInput')

    def start(self):
        self.is_running = True
        dpg.show_viewport()

    def nextFrame(self):
        # when the replay model is suspended, click "next frame" button will move
        # one single step
        if not self.is_running:
            self.frameIncre += 1

    def destroy(self):
        self.is_running = False
        dpg.destroy_context()

    def resume(self):
        self.is_running = True
        dpg.set_item_label('PauseResumeButton', 'Pause')
        dpg.bind_item_theme('PauseResumeButton', 'PauseButtonTheme')

    def pause(self):
        self.is_running = False
        dpg.set_item_label('PauseResumeButton', 'Resume')
        dpg.bind_item_theme('PauseResumeButton', 'ResumeButtonTheme')

    def toggle(self):
        if self.is_running:
            self.pause()
        else:
            self.resume()
