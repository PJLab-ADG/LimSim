import dearpygui.dearpygui as dpg
from visualization.coord import CoordTF
from typing import Tuple


class GUI:
    def __init__(self) -> None:
        self.is_running = False
        self.frameIncre = 0
        self.mouse_pos = None

        self.zoom_speed: float = 1.0
        self.is_dragging: bool = False
        self.old_offset = (0, 0)
        self.view_scale = (1, 1)

    def start(self):
        self.is_running = True
        self.setup()
        self.setup_themes()
        self.create_windows()
        self.create_handlers()
        self.resize_windows()
        self.ctf = CoordTF(150, "MainWindow")
        dpg.show_viewport()

    def setup(self):
        dpg.create_context()
        dpg.create_viewport(title="TrafficSimulator", width=1670, height=760)
        dpg.setup_dearpygui()

    def setup_themes(self):
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_FrameRounding, 3, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FrameBorderSize, 0.5, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowBorderSize, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_color(dpg.mvNodeCol_NodeBackground, (255, 255, 255))

        dpg.bind_theme(global_theme)

        with dpg.theme(tag="plot_theme_v"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, (255, 165, 2), category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3, category=dpg.mvThemeCat_Plots
                )

        with dpg.theme(tag="plot_theme_v_future"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line,
                    (255, 165, 2, 70),
                    category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3, category=dpg.mvThemeCat_Plots
                )

        with dpg.theme(tag="plot_theme_a"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, (0, 148, 50), category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3, category=dpg.mvThemeCat_Plots
                )

        with dpg.theme(tag="plot_theme_a_future"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line,
                    (0, 148, 50, 70),
                    category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3, category=dpg.mvThemeCat_Plots
                )
        with dpg.theme(tag="plot_theme_score"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, (230, 0, 0), category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3, category=dpg.mvThemeCat_Plots
                )

        with dpg.theme(tag="plot_theme_score_future"):
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line,
                    (230, 0, 0, 70),
                    category=dpg.mvThemeCat_Plots,
                )
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3, category=dpg.mvThemeCat_Plots
                )

    def create_windows(self):
        with dpg.font_registry():
            default_font = dpg.add_font("simModel/common/fonts/Meslo.ttf", 18)

        dpg.bind_font(default_font)

        dpg.add_window(tag="MainWindow", label="Microscopic simulation", no_close=True)

        self.BGnode = dpg.add_draw_node(tag="CanvasBG", parent="MainWindow")
        dpg.add_draw_node(tag="Canvas", parent="MainWindow")

        with dpg.window(tag="vState", label="Vehicle states", no_close=True):
            with dpg.plot(tag="vehicleStates", height=305, width=400):
                dpg.add_plot_legend()

                dpg.add_plot_axis(dpg.mvXAxis, label="Time Steps (s)", tag="v_x_axis")
                dpg.add_plot_axis(dpg.mvYAxis, label="Velocity (m/s)", tag="v_y_axis")
                dpg.add_plot_axis(
                    dpg.mvYAxis, label="Acceleration (m/s^2)", tag="a_y_axis"
                )
                dpg.set_axis_limits("v_y_axis", 0, 15)
                dpg.set_axis_limits("v_x_axis", -20, 20)
                dpg.set_axis_limits("a_y_axis", -5, 5)

                # series belong to a y axis
                dpg.add_line_series(
                    [], [], parent="v_y_axis", tag="v_series_tag", label="Velocity"
                )
                dpg.add_line_series(
                    [], [], parent="v_y_axis", tag="v_series_tag_future"
                )

                dpg.add_line_series(
                    [],
                    [],
                    parent="a_y_axis",
                    tag="a_series_tag",
                    label="Acceleration",
                )
                dpg.add_line_series(
                    [], [], parent="a_y_axis", tag="a_series_tag_future"
                )

                dpg.bind_item_theme("v_series_tag", "plot_theme_v")
                dpg.bind_item_theme("v_series_tag_future", "plot_theme_v_future")

                dpg.bind_item_theme("a_series_tag", "plot_theme_a")
                dpg.bind_item_theme("a_series_tag_future", "plot_theme_a_future")
        with dpg.window(
            tag="sEvaluation",
            label="Evaluation",
            no_close=True,
        ):
            with dpg.plot(tag="drivingScore", height=305, width=400):
                dpg.add_plot_legend()

                dpg.add_plot_axis(
                    dpg.mvXAxis, label="Time Steps (s)", tag="score_x_axis"
                )
                dpg.add_plot_axis(
                    dpg.mvYAxis, label="Driving Score", tag="score_y_axis"
                )

                dpg.set_axis_limits("score_y_axis", -0.2, 1.2)
                dpg.set_axis_limits("score_x_axis", -20, 20)

                # series belong to a y axis
                dpg.add_line_series(
                    [],
                    [],
                    parent="score_y_axis",
                    tag="score_past_tag",
                    label="Driving Score",
                )
                dpg.add_line_series([], [], parent="score_y_axis", tag="score_plan_tag")

                dpg.bind_item_theme("score_past_tag", "plot_theme_score")
                dpg.bind_item_theme("score_plan_tag", "plot_theme_score_future")

        with dpg.window(
            tag="macroMap",
            label="City-level map",
            no_close=True,
        ):
            dpg.add_draw_node(tag="mapBackground", parent="macroMap")
            dpg.add_draw_node(tag="movingScene", parent="macroMap")

        with dpg.window(
            tag="simInfo",
            label="Simulation information",
            no_close=True,
        ):
            dpg.add_draw_node(tag="infoText", parent="simInfo")

    def create_handlers(self):
        with dpg.handler_registry():
            dpg.add_mouse_down_handler(callback=self.mouse_down)
            dpg.add_mouse_drag_handler(callback=self.mouse_drag)
            dpg.add_mouse_release_handler(callback=self.mouse_release)
            dpg.add_mouse_wheel_handler(callback=self.mouse_wheel)
            dpg.add_mouse_double_click_handler(callback=self.mouse_double_click)
            dpg.set_viewport_resize_callback(callback=self.on_resize)

    def on_resize(self):
        width = dpg.get_viewport_width()
        height = dpg.get_viewport_height()
        print(f"Viewport resized to: {width}x{height}")
        self.view_scale = (width / 1670, height / 760)
        self.resize_windows()

    def resize_windows(self):
        dpg.set_item_width("MainWindow", 1670 * self.view_scale[0] - 440 - 520)
        dpg.set_item_height("MainWindow", 700 * self.view_scale[1])
        dpg.set_item_pos("MainWindow", (520, 10))

        dpg.set_item_width("vState", 415)
        dpg.set_item_height("vState", 345 * self.view_scale[1])
        dpg.set_item_pos(
            "vState", (1670 * self.view_scale[0] - 440, 10 * self.view_scale[1])
        )

        dpg.set_item_width("sEvaluation", 415)
        dpg.set_item_height("sEvaluation", 345 * self.view_scale[1])
        dpg.set_item_pos(
            "sEvaluation",
            (1670 * self.view_scale[0] - 440, 365 * self.view_scale[1]),
        )

        dpg.set_item_width("macroMap", 500)
        dpg.set_item_height("macroMap", 500 * self.view_scale[1])
        dpg.set_item_pos("macroMap", (10 * self.view_scale[0], 10 * self.view_scale[1]))

        dpg.set_item_width("simInfo", 500)
        dpg.set_item_height("simInfo", 190 * self.view_scale[1])
        dpg.set_item_pos("simInfo", (10 * self.view_scale[0], 520 * self.view_scale[1]))

    def drawMainWindowWhiteBG(self, pmin: Tuple[float], pmax: Tuple[float]):
        centerx = (pmin[0] + pmax[0]) / 2
        centery = (pmin[1] + pmax[1]) / 2
        dpg.draw_rectangle(
            self.ctf.dpgCoord(pmin[0], pmin[1], centerx, centery),
            self.ctf.dpgCoord(pmax[0], pmax[1], centerx, centery),
            thickness=0,
            fill=(255, 255, 255),
            parent=self.BGnode,
        )

    def mouse_double_click(self):
        if dpg.is_item_hovered("MainWindow"):
            self.mouse_pos = dpg.get_mouse_pos()
            self.zoom_speed = 1
        if dpg.is_item_hovered("macroMap"):
            if self.is_running:
                self.is_running = False
            else:
                self.is_running = True

    def mouse_down(self):
        if not self.is_dragging:
            if dpg.is_item_hovered("MainWindow"):
                self.is_dragging = True
                self.old_offset = self.ctf.offset

    def mouse_drag(self, sender, app_data):
        if self.is_dragging:
            self.ctf.offset = (
                self.old_offset[0] + app_data[1] / self.ctf.zoomScale,
                self.old_offset[1] + app_data[2] / self.ctf.zoomScale,
            )

    def mouse_release(self):
        self.is_dragging = False

    def mouse_wheel(self, sender, app_data):
        if dpg.is_item_hovered("MainWindow"):
            self.zoom_speed = 1 + 0.01 * app_data

    def update_inertial_zoom(self, clip=0.005):
        if self.zoom_speed != 1:
            self.ctf.dpgDrawSize *= self.zoom_speed
            self.zoom_speed = 1 + (self.zoom_speed - 1) / 1.05
        if abs(self.zoom_speed - 1) < clip:
            self.zoom_speed = 1
