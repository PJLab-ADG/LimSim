import cv2
import time
import base64
import logging
import traci
from matplotlib import pyplot as plt

from sumo_integration.carla_simulation import CarlaSimulation
from sumo_integration.sumo_simulation import SumoSimulation
from run_synchronization import SimulationSynchronization

from simModel.egoTracking.MPModel import Model
from simModel.common.MPGUI import GUI
from simModel.common.RenderDataQueue import RenderDataQueue, DecisionDataQueue
from trafficManager.traffic_manager import TrafficManager

# logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

# BASIC_SETTINGS
ego_id = 10
sumo_gui = False
sumo_cfg_file = './networkFiles/CarlaTown06/Town06.sumocfg'
sumo_net_file = "./networkFiles/CarlaTown06/Town06.net.xml"
sumo_rou_file = "./networkFiles/CarlaTown06/carlavtypes.rou.xml,networkFiles/CarlaTown06/Town06.rou.xml"
carla_host = '127.0.0.1'
carla_port = 2000
step_length = 0.1
tls_manager = 'sumo'
sync_vehicle_color = True
sync_vehicle_lights = True
database = 'visionCloseLoop.db'


if __name__ == '__main__':
    renderQueue = RenderDataQueue(5)
    decisionQueue = DecisionDataQueue(5)
    model = Model(
        egoID=str(ego_id), netFile=sumo_net_file, rouFile=sumo_rou_file,
        RDQ=renderQueue, dataBase=database, SUMOGUI=sumo_gui, simNote=''
    )
    model.start()
    planner = TrafficManager(model)
    netBoundary = traci.simulation.getNetBoundary()

    gui = GUI(renderQueue, decisionQueue, netBoundary)
    gui.start()

    # CARLA Co-simulation initialization
    sumo_simulation = SumoSimulation(sumo_cfg_file)
    carla_simulation = CarlaSimulation(
        carla_host, carla_port, step_length
    )
    carla_simulation.client.reload_world()
    synchronization = SimulationSynchronization(
        sumo_simulation, carla_simulation, 
        str(ego_id), tls_manager,
        sync_vehicle_color, 
        sync_vehicle_lights
    )

    while not model.tpEnd:
        start = time.time()
        model.moveStep()
        synchronization.tick()

        if model.timeStep % 1 == 0:
            roadgraph, vehicles = model.exportSce()
            if model.tpStart and roadgraph:
                carla_ego = synchronization.getEgo()
                if carla_ego:
                    synchronization.moveSpectator(carla_ego)
                    synchronization.setFrontViewCamera(carla_ego)
                    try:
                        image_buffer = synchronization.getFrontViewImage()
                        decisionQueue.put(image_buffer)
                        _, buffer = cv2.imencode('.png', image_buffer)
                        image_base64 = base64.b64encode(buffer).decode('utf-8')
                        model.dbBridge.commitData(
                            'visualPromptsINFO', 
                            (model.timeStep, image_base64, '', '')
                            )
                    except NameError:
                        continue
                model.setTrajectories({})
            else:
                model.ego.exitControlMode()
        model.updateVeh()
    
    synchronization.close()
    model.destroy()
    gui.join()
    gui.terminate()