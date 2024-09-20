from datetime import datetime
from simModel.model import Model
from visualization.plotEngine import PlotEngine
import sys
import logger
import time

log = logger.setup_app_level_logger(file_name="app_debug.log")


file_paths = {
    "CarlaTown01": "networkFiles/CarlaTown/Town01.xodr",
    "CarlaTown02": "networkFiles/CarlaTown/Town02.xodr",
    "CarlaTown03": "networkFiles/CarlaTown/Town03.xodr",
    "CarlaTown04": "networkFiles/CarlaTown/Town04.xodr",
    "CarlaTown05": "networkFiles/CarlaTown/Town05.xodr",
    "CarlaTown06": "networkFiles/CarlaTown/Town06.xodr",
    "CarlaTown07": "networkFiles/CarlaTown/Town07.xodr",
}


def run_model(net_file, run_time, demands):
    model = Model(net_file, run_time, demands)
    model.start()
    model.updateVeh()
    print(
        "-" * 10,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "real-time simulation is running ...",
        "-" * 10,
    )
    plotEngine = PlotEngine(model)
    plotEngine.start()

    while not model.end():
        if model.simPause.pause.value == 0:
            model.moveStep()
        model.updateVeh()
        time.sleep(0.00)
        if model.timeStep % 10 == 0:
            print(
                "running time: {:>4d} / {} | number of vehicles on the road: {:>3d}".format(
                    model.timeStep, model.run_time, len(model.vehRunning)
                )
            )
    print(
        "-" * 10,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "the simulation is end.",
        "-" * 10,
    )
    model.destroy()
    plotEngine.terminate()
    plotEngine.join()


def replay_model(net_file):
    model = Model(net_file)
    model.replayMoveStep()
    model.replayUpdateVeh()
    print(
        "-" * 10,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "replayed simulation is running ...",
        "-" * 10,
    )
    plotEngine = PlotEngine(model)
    plotEngine.start()
    while not model.end():
        if model.simPause.pause.value == 0:
            model.replayMoveStep()
        model.replayUpdateVeh()
        time.sleep(0.03)
        if model.timeStep % 10 == 0:
            print(
                "running time: {:>4d} / {} | number of vehicles on the road: {:>3d}".format(
                    model.timeStep, model.run_time, len(model.vehRunning)
                )
            )
    print(
        "-" * 10,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "the simulation is end.",
        "-" * 10,
    )
    model.destroy()
    plotEngine.terminate()
    plotEngine.join()


if __name__ == "__main__":
    net_file = file_paths["CarlaTown01"]
    # Two modes are avialable for simulation
    # The replay mode requires reading database information
    if len(sys.argv) > 1 and sys.argv[1] == "replay":
        replay_model(net_file)
    else:
        run_model(net_file, run_time=1000, demands="demands.txt")
