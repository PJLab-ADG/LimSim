from simModel.fixedScene import interReplay
from trafficManager.traffic_manager import TrafficManager

import logger
# config a logger, set use_stdout=True to output log to terminal
log = logger.setup_app_level_logger(file_name="app_debug.log",
                                    level="DEBUG",
                                    use_stdout=False)


firmodel = interReplay.InterReplayModel(
    dataBase='fixedSceneTest.db'
)
planner = TrafficManager(firmodel)

while not firmodel.tpEnd:
    firmodel.moveStep()
    if firmodel.timeStep % 5 == 0:
        roadgraph, vehicles = firmodel.exportSce()
        if roadgraph:
            trajectories = planner.plan(
                firmodel.timeStep * 0.1, roadgraph, vehicles
            )
        else:
            trajectories = {}
        firmodel.setTrajectories(trajectories)
    else:
        firmodel.setTrajectories({})
firmodel.gui.destroy()
