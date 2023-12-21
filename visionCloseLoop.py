import os
import cv2
import time
import base64
import logging
import traci
import json

from sumo_integration.carla_simulation import CarlaSimulation
from sumo_integration.sumo_simulation import SumoSimulation
from run_synchronization import SimulationSynchronization

from simModel.egoTracking.MPModel import Model
from simModel.common.MPGUI import GUI
from simModel.common.RenderDataQueue import RenderDataQueue, DecisionDataQueue
from trafficManager.traffic_manager import TrafficManager
from DriverAgent.Informer import Informer
from DriverAgent.VLMAgent import VLMAgent

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
    informer = Informer()
    api_key = os.environ.get('OPENAI_API_KEY')
    vlmagent = VLMAgent(api_key)

    while not model.tpEnd:
        start = time.time()
        model.moveStep()
        synchronization.tick()

        if model.timeStep % 20 == 0:
            roadgraph, vehicles = model.exportSce()
            if model.tpStart and roadgraph:
                carla_ego = synchronization.getEgo()
                if carla_ego:
                    synchronization.moveSpectator(carla_ego)
                    synchronization.setFrontViewCamera(carla_ego)
                    try:
                        image_buffer = synchronization.getFrontViewImage()
                        _, buffer = cv2.imencode('.png', image_buffer)
                        image_base64 = base64.b64encode(buffer).decode('utf-8')
                        actionInfo = informer.getActionInfo(vehicles, roadgraph)
                        naviInfo = informer.getNaviInfo(vehicles)
                        if naviInfo:
                            information = actionInfo + naviInfo
                        else:
                            information = actionInfo
                        print('information: ', information)
                        decisionQueue.put((image_buffer, information))
                        response, ego_behavior = vlmagent.makeDecision(information, image_base64)
                        print('ego behavior: ', ego_behavior)
                        model.dbBridge.commitData(
                            'visualPromptsINFO', 
                            (model.timeStep, image_base64, '', information)
                            )
                        model.dbBridge.commitData(
                            'promptsINFO',
                            (model.timeStep, json.dumps(response))
                        )
                        trajectories = planner.plan(
                            model.timeStep * 0.1, roadgraph, vehicles, ego_behavior
                        )

                    except NameError:
                        continue
                model.setTrajectories(trajectories)
            else:
                model.ego.exitControlMode()
        model.updateVeh()
    
    synchronization.close()
    model.destroy()
    gui.join()
    gui.terminate()