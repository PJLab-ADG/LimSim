import dearpygui.dearpygui as dpg
from utils.simBase import CoordTF
from typing import Tuple


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
        if self.mode == 'real-time-ego':
            dpg.create_viewport(title="TrafficSimulator",
                                width=1670, height=760)
        elif self.mode == 'real-time-local':
            dpg.create_viewport(title='TrafficSimulator',
                                width=740, height=760)
        elif self.mode == 'replay-ego':
            dpg.create_viewport(title="TrafficSimulator",
                                width=1670, height=870)
        elif self.mode == 'replay-local':
            dpg.create_viewport(title="TrafficSimulator",
                                width=740, height=870)
        else:
            raise TypeError('Nonexistent gui mode!')
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

        if self.mode == 'replay-ego' or self.mode == 'replay-local':
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

        if self.mode == 'real-time-ego' or self.mode == 'replay-ego':
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

        if self.mode == 'replay-ego' or self.mode == 'replay-local':
            with dpg.window(
                tag='ControlWindow',
                label='Menu',
                no_close=True,
                # no_collapse=True,
                # no_resize=True
            ):
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Pause", tag="PauseResumeButton",
                        callback=self.toggle
                    )

                    dpg.add_button(label="Next frame",
                                   callback=self.nextFrame)

                dpg.add_spacer(height=5)

                with dpg.group(horizontal=True):
                    dpg.add_text('Time delay: ')
                    dpg.add_slider_float(
                        tag="DelayInput",
                        min_value=0, max_value=1,
                        default_value=0, callback=self.setDelay
                    )

            dpg.bind_item_theme('PauseResumeButton', 'PauseButtonTheme')

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

        if self.mode == 'real-time-ego' or self.mode == 'replay-ego':
            with dpg.window(
                tag='vState',
                label='Vehicle states',
                no_close=True,
                # no_collapse=True,
                # no_resize=True,
                # no_move=True
            ):
                with dpg.plot(tag='vehicleStates', height=305, width=400):
                    dpg.add_plot_legend()

                    dpg.add_plot_axis(
                        dpg.mvXAxis, label="time steps (s)", tag='v_x_axis'
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis, label="Velocity (m/s)", tag="v_y_axis"
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis, label="Acceleration (m/s^2)", tag="a_y_axis"
                    )
                    dpg.set_axis_limits('v_y_axis', 0, 15)
                    dpg.set_axis_limits('v_x_axis', -49, 50)
                    dpg.set_axis_limits('a_y_axis', -5, 5)

                    # series belong to a y axis
                    dpg.add_line_series(
                        [], [], parent="v_y_axis", tag="v_series_tag",
                        label='Velocity'
                    )
                    dpg.add_line_series(
                        [], [], parent="v_y_axis", tag="v_series_tag_future"
                    )

                    dpg.add_line_series(
                        [], [], parent="a_y_axis", tag="a_series_tag",
                        label='Acceleration'
                    )
                    dpg.add_line_series(
                        [], [], parent="a_y_axis", tag="a_series_tag_future"
                    )

                    dpg.bind_item_theme("v_series_tag", "plot_theme_v")
                    dpg.bind_item_theme(
                        'v_series_tag_future', 'plot_theme_v_future'
                    )

                    dpg.bind_item_theme("a_series_tag", "plot_theme_a")
                    dpg.bind_item_theme(
                        'a_series_tag_future', 'plot_theme_a_future'
                    )
            with dpg.window(
                tag='sEvaluation',
                label='Evaluation',
                no_close=True,
            ):
                dpg.add_draw_node(tag="radarBackground", parent="sEvaluation")
                dpg.add_draw_node(tag="radarPlot", parent="sEvaluation")

            with dpg.window(
                tag='macroMap',
                label='City-level map',
                no_close=True,
            ):
                dpg.add_draw_node(tag="mapBackground", parent="macroMap")
                dpg.add_draw_node(tag="movingScene", parent="macroMap")

            with dpg.window(
                tag='simInfo',
                label='Simulation information',
                no_close=True,
            ):
                dpg.add_draw_node(tag="infoText", parent="simInfo")

    def create_handlers(self):
        with dpg.handler_registry():
            dpg.add_mouse_down_handler(callback=self.mouse_down)
            dpg.add_mouse_drag_handler(callback=self.mouse_drag)
            dpg.add_mouse_release_handler(callback=self.mouse_release)
            dpg.add_mouse_wheel_handler(callback=self.mouse_wheel)

    def resize_windows(self):
        if self.mode == 'real-time-ego':
            dpg.set_item_width("MainWindow", 700)
            dpg.set_item_height("MainWindow", 700)
            dpg.set_item_pos("MainWindow", (520, 10))

            dpg.set_item_width('vState', 415)
            dpg.set_item_height('vState', 345)
            dpg.set_item_pos('vState', (1230, 10))

            dpg.set_item_width('sEvaluation', 415)
            dpg.set_item_height('sEvaluation', 345)
            dpg.set_item_pos('sEvaluation', (1230, 365))

            dpg.set_item_width('macroMap', 500)
            dpg.set_item_height('macroMap', 500)
            dpg.set_item_pos('macroMap', (10, 10))

            dpg.set_item_width('simInfo', 500)
            dpg.set_item_height('simInfo', 190)
            dpg.set_item_pos('simInfo', (10, 520))
        elif self.mode == 'real-time-local':
            dpg.set_item_width("MainWindow", 700)
            dpg.set_item_height("MainWindow", 700)
            dpg.set_item_pos("MainWindow", (10, 10))
        elif self.mode == 'replay-ego':
            dpg.set_item_width('ControlWindow', 1635)
            dpg.set_item_height('ControlWindow', 0)
            dpg.set_item_pos('ControlWindow', (10, 10))

            dpg.set_item_width("MainWindow", 700)
            dpg.set_item_height("MainWindow", 700)
            dpg.set_item_pos("MainWindow", (520, 120))

            dpg.set_item_width('vState', 415)
            dpg.set_item_height('vState', 345)
            dpg.set_item_pos('vState', (1230, 120))

            dpg.set_item_width('sEvaluation', 415)
            dpg.set_item_height('sEvaluation', 345)
            dpg.set_item_pos('sEvaluation', (1230, 475))

            dpg.set_item_width('macroMap', 500)
            dpg.set_item_height('macroMap', 500)
            dpg.set_item_pos('macroMap', (10, 120))

            dpg.set_item_width('simInfo', 500)
            dpg.set_item_height('simInfo', 190)
            dpg.set_item_pos('simInfo', (10, 630))
        elif self.mode == 'replay-local':
            dpg.set_item_width('ControlWindow', 700)
            dpg.set_item_height('ControlWindow', 0)
            dpg.set_item_pos('ControlWindow', (10, 10))

            dpg.set_item_width("MainWindow", 700)
            dpg.set_item_height("MainWindow", 700)
            dpg.set_item_pos("MainWindow", (10, 120))
        else:
            raise TypeError('Nonexistent mode!')

    def drawMainWindowWhiteBG(self, pmin: Tuple[float], pmax: Tuple[float]):
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
