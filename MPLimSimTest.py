from simModel.egoTracking.MPModel import Model
from simModel.common.MPGUI import GUI
from simModel.common.RenderDataQueue import RenderDataQueue
from trafficManager.traffic_manager import TrafficManager

import time

import traci

ego_id = 10
sumo_gui = False
sumo_net_file = "./networkFiles/CarlaTown06/Town06.net.xml"
sumo_rou_file = "./networkFiles/CarlaTown06/carlavtypes.rou.xml,networkFiles/CarlaTown06/Town06.rou.xml"
database = 'MPModelTest.db'



if __name__ == '__main__':
    Queue = RenderDataQueue(5)
    model = Model(
        egoID=str(ego_id), netFile=sumo_net_file, rouFile=sumo_rou_file,
        RDQ=Queue, dataBase=database, SUMOGUI=sumo_gui, simNote=''
    )

    model.start()
    planner = TrafficManager(model)
    netBoundary = traci.simulation.getNetBoundary()

    gui = GUI(Queue, netBoundary)
    gui.start()

    while not model.tpEnd:
        model.moveStep()
        if model.timeStep % 5 == 0:
            roadgraph, vehicles = model.exportSce()
            if model.tpStart and roadgraph:
                trajectories = planner.plan(
                    model.timeStep * 0.1, roadgraph, vehicles
                )
                model.setTrajectories(trajectories)
            else:
                model.ego.exitControlMode()
        model.updateVeh()


    model.destroy()
    gui.join()
    gui.terminate()