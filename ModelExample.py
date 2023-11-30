from simModel.egoTracking.model import Model
from trafficManager.traffic_manager import TrafficManager

import logger

log = logger.setup_app_level_logger(file_name="app_debug.log")


file_paths = {
    "corridor": (
        "networkFiles/corridor/corridor.net.xml",
        "networkFiles/corridor/corridor.rou.xml",
    ),
    "CarlaTown05": (
        "networkFiles/CarlaTown05/Town05.net.xml",
        "networkFiles/CarlaTown05/carlavtypes.rou.xml,networkFiles/CarlaTown05/Town05.rou.xml",
    ),
    "CarlaTown06": (
        "./networkFiles/CarlaTown06/Town06.net.xml",
        "./networkFiles/CarlaTown06/carlavtypes.rou.xml,networkFiles/CarlaTown06/Town06.rou.xml"
    ),
    "bigInter": (
        "networkFiles/bigInter/bigInter.net.xml",
        "networkFiles/bigInter/bigInter.rou.xml",
    ),
    "roundabout": (
        "networkFiles/roundabout/roundabout.net.xml",
        "networkFiles/roundabout/roundabout.rou.xml",
    ),
    "bilbao":   (
        "networkFiles/bilbao/osm.net.xml",
        "networkFiles/bilbao/osm.rou.xml",
    ),
    #######
    # Please make sure you have request the access from https://github.com/ozheng1993/UCF-SST-CitySim-Dataset and put the road network files (.net.xml) in the relevent networkFiles/CitySim folder
    "freewayB": (
        "networkFiles/CitySim/freewayB/freewayB.net.xml",
        "networkFiles/CitySim/freewayB/freewayB.rou.xml",
    ),
    "Expressway_A": (
        "networkFiles/CitySim/Expressway_A/Expressway_A.net.xml",
        "networkFiles/CitySim/Expressway_A/Expressway_A.rou.xml",
    ),
    ########
}


def run_model(
    net_file,
    rou_file,
    ego_veh_id="61",
    data_base="egoTrackingTest.db",
    SUMOGUI=0,
    sim_note="example simulation, LimSim-v-0.2.0.",
    carla_cosim=False,
):
    model = Model(
        ego_veh_id,
        net_file,
        rou_file,
        dataBase=data_base,
        SUMOGUI=SUMOGUI,
        simNote=sim_note,
        carla_cosim=carla_cosim,
    )
    model.start()
    planner = TrafficManager(model)

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


if __name__ == "__main__":
    net_file, rou_file = file_paths['CarlaTown06']
    run_model(net_file, rou_file, ego_veh_id="11", carla_cosim=False)
