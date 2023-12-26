import time
import base64
import logging
import cv2
from matplotlib import pyplot as plt
import yaml
from easydict import EasyDict as edict

# Import custom modules
from sumo_integration.carla_simulation import CarlaSimulation
from sumo_integration.sumo_simulation import SumoSimulation
from run_synchronization import SimulationSynchronization

from simModel.egoTracking.model import Model
from trafficManager.traffic_manager import TrafficManager
from DriverAgent.Informer import Informer

# Import agents
from DriverAgent.Informer import Informer as Informer
from DriverAgent.VLMAgent import VLMAgent as VLMAgent
from DriverAgent.LLMAgent import LLMAgent as LLMAgent
from DriverAgent.VLMDescriptor import VLMDescriptor as VLMDescriptor
from DriverAgent.EnvDescriptor import EnvDescriptor as EnvDescriptor

# Load configurations
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize logging
logging.basicConfig(format='%(levelname)s: %(message)s',
                    level=logging.getLevelName(config['logging_level']))


def initialize_model(config):
    agents = edict()
    agents.informer = Informer()
    if config["agent_type"] == "VLM":
        agents.agent = VLMAgent()
    elif config["agent_type"] == "LLM":
        agents.agent = LLMAgent()
        if config["descriptor_type"] == "VLM":
            agents.descriptor = VLMDescriptor()
        elif config["descriptor_type"] == "GT":
            agents.descriptor = EnvDescriptor()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return agents, Model(
        egoID=str(config['ego_id']),
        netFile=config['sumo_net_file'],
        rouFile=config['sumo_rou_file'],
        dataBase=config['database'],
        SUMOGUI=config['sumo_gui'],
        simNote=''
    )


def initialize_simulation(config):
    sumo_simulation = SumoSimulation(config['sumo_cfg_file'])
    carla_simulation = CarlaSimulation(
        config['carla_host'], config['carla_port'], config['step_length']
    )
    carla_simulation.client.reload_world()
    synchronization = SimulationSynchronization(
        sumo_simulation, carla_simulation,
        str(config['ego_id']), config['tls_manager'],
        config['sync_vehicle_color'],
        config['sync_vehicle_lights']
    )
    return synchronization


def main_loop(model: Model, agents: edict, traffic_manager: TrafficManager, synchronization: SimulationSynchronization, config):
    while not model.tpEnd:
        model.moveStep()
        synchronization.tick()

        decision_frames = 10 / config['decision_frequency']
        if model.timeStep % decision_frames == 0:
            roadgraph, vehicles = model.exportSce()
            if model.tpStart and roadgraph:
                carla_ego = synchronization.getEgo()
                if carla_ego:
                    synchronization.moveSpectator(carla_ego)
                    synchronization.setFrontViewCamera(carla_ego)

                    trajectories = handle_ego_car(
                        synchronization, model, agents, traffic_manager, roadgraph, vehicles, config)

                model.setTrajectories(trajectories)
            else:
                model.ego.exitControlMode()
        model.updateVeh()

    synchronization.close()
    model.destroy()


"""
There are mainly two pipeline to handle the ego car:

            Informer Message
                  |
                  v
            Camera Image 
             /         \
            /           \
           v             v
        VLM Agent   Vision Descriptor
            |       (Custom VLM or GT)
            |           |
            |           v
             \        LLM Agent
              \        /
               v      v
               Decision 
                   |
                   v
                Planner
"""


def handle_ego_car(synchronization, model: Model, agents: edict(), traffic_manager: TrafficManager, roadgraph, vehicles, config):
    try:
        # Step 0: Get Informer messages
        information = agents.informer.getActionInfo(
            vehicles, roadgraph) + '\n\n' + agents.informer.getNaviInfo(vehicles)

        # Step 1: Get the ego car's camera image
        image_buffer = synchronization.getFrontViewImage()
        plt.imshow(image_buffer)
        plt.pause(0.1)
        _, buffer = cv2.imencode('.png', image_buffer)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        model.dbBridge.commitData(
            'visualPromptsINFO',
            (model.timeStep, image_base64, '', '')
        )

        # Step 2: Get the ego car's decision
        if config["agent_type"] == "VLM":
            ego_behaviour = agents.agent.makeDecision(
                information, image_base64)
        elif config["agent_type"] == "LLM":
            # todo: add descriptor
            if config["descriptor_type"] == "VLM":
                descriptions = agents.descriptor.getDescription()
            elif config["descriptor_type"] == "GT":

                descriptions = agents.descriptor.getDescription(
                    roadgraph, vehicles, traffic_manager, model.timeStep * 0.1)

            ego_behaviour, response, human_question, fewshot, result = agents.agent.makeDecision(
                information, image_base64, descriptions)
             
            agents.agent.dbBridge.insertPrompts(model.timeStep * 0.1, vehicles["egoCar"]["id"], result, human_question, fewshot, response)
            

        # Step 3: Plan the ego car's trajectory
        # todo: modify original traffic manager plan function
        #       to accept ego car's behaviour and can choose weather plan for other cars
        paths = traffic_manager.plan(model.timeStep, roadgraph, vehicles, ego_behaviour, other_plan=True)

        # Step 4: evaluate the ego car's decision
        
    except NameError:
        pass

    return paths


if __name__ == '__main__':
    agents, model = initialize_model(config)
    model.start()
    synchronization = initialize_simulation(config)
    traffic_manager = TrafficManager(model)

    main_loop(model, agents, traffic_manager, synchronization, config)
