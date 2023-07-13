import logger
# config a logger, set use_stdout=True to output log to terminal
log = logger.setup_app_level_logger(file_name="app_debug.log",
                                    level="DEBUG",
                                    use_stdout=False)

from trafficManager.traffic_manager import TrafficManager
from simModel.egoTracking import interReplay

irmodel = interReplay.InterReplayModel(
    dataBase='egoTrackingTest.db', startFrame=5000)
planner = TrafficManager(irmodel)

while not irmodel.tpEnd:
    irmodel.moveStep()
    if irmodel.timeStep % 5 == 0:
        roadgraph, vehicles = irmodel.exportSce()
        if roadgraph:
            trajectories = planner.plan(
                irmodel.timeStep * 0.1, roadgraph, vehicles)
        else:
            trajectories = {}
        irmodel.setTrajectories(trajectories)
    else:
        irmodel.setTrajectories({})
irmodel.gui.destroy()