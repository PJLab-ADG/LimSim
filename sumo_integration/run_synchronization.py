#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Script to integrate CARLA and SUMO simulations
"""

# ==================================================================================================
# -- imports ---------------------------------------------------------------------------------------
# ==================================================================================================

import logging, logger
import time
from typing import Dict, Optional

import numpy as np

# ==================================================================================================
# -- find carla module -----------------------------------------------------------------------------
# ==================================================================================================

import glob
import os
import sys

try:
    sys.path.append(
        glob.glob('../../PythonAPI/carla/dist/carla-*%d.%d-%s.egg' %
                  (sys.version_info.major, sys.version_info.minor,
                   'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla  # pylint: disable=import-error
import numpy as np
# ==================================================================================================
# -- find traci module -----------------------------------------------------------------------------
# ==================================================================================================

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

# ==================================================================================================
# -- sumo integration imports ----------------------------------------------------------------------
# ==================================================================================================

from sumo_integration.bridge_helper import BridgeHelper  # pylint: disable=wrong-import-position
from sumo_integration.carla_simulation import CarlaSimulation  # pylint: disable=wrong-import-position
from sumo_integration.constants import INVALID_ACTOR_ID  # pylint: disable=wrong-import-position
from sumo_integration.sumo_simulation import SumoSimulation  # pylint: disable=wrong-import-position

from simModel.DataQueue import CameraImages

# ==================================================================================================
# -- synchronization_loop --------------------------------------------------------------------------
# ==================================================================================================

logger = logger.setup_app_level_logger(logger_name = "prompt", file_name="prompt_debug.log")
logging = logging.getLogger("prompt").getChild(__name__)

class SimulationSynchronization(object):
    """
    SimulationSynchronization class is responsible for the synchronization of sumo and carla
    simulations.
    """
    def __init__(self,
                sumo_simulation: SumoSimulation,
                carla_simulation: CarlaSimulation,
                ego_id: str,
                tls_manager='none',
                sync_vehicle_color=False,
                sync_vehicle_lights=False,
                ):

        self.sumo = sumo_simulation
        self.carla = carla_simulation

        self.tls_manager = tls_manager
        self.sync_vehicle_color = sync_vehicle_color
        self.sync_vehicle_lights = sync_vehicle_lights
        self.ego_id = ego_id

        if tls_manager == 'carla':
            self.sumo.switch_off_traffic_lights()
        elif tls_manager == 'sumo':
            self.carla.switch_off_traffic_lights()

        # Mapped actor ids.
        self.sumo2carla_ids = {}  # Contains only actors controlled by sumo.
        self.carla2sumo_ids = {}  # Contains only actors controlled by carla.

        BridgeHelper.blueprint_library = self.carla.world.get_blueprint_library()
        BridgeHelper.offset = self.sumo.get_net_offset()

        # Configuring carla simulation in sync mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.carla.step_length
        self.carla.world.apply_settings(settings)

        # configure camera 
        self.CAM_FONRT_RECORD = False
        self.CAM_FRONT_RIGHT_RECORD = False
        self.CAM_FRONT_LEFT_RECORD = False
        self.CAM_BACK_LEFT_RECORD = False
        self.CAM_BACK_RECORD = False
        self.CAM_BACK_RIGHT_RECORD = False

        # nuScenes cameras settings
        self.cameraInfo = {
            'CAM_FRONT': {
                'transform': carla.Transform(
                    carla.Location(x=1.5, y=0, z=2),
                    carla.Rotation(yaw=0, pitch=0, roll=0)
                ),
                'fov': '70',
                'record': self.CAM_FONRT_RECORD,
            },
            'CAM_FRONT_RIGHT': {
                'transform': carla.Transform(
                    carla.Location(x=1.5, y=0.7, z=2),
                    carla.Rotation(yaw=55, pitch=0, roll=0)
                ),
                'fov': '70',
                'record': self.CAM_FRONT_RIGHT_RECORD
            },
            'CAM_FRONT_LEFT': {
                'transform': carla.Transform(
                    carla.Location(x=1.5, y=-0.7, z=2),
                    carla.Rotation(yaw=-55, pitch=0, roll=0)
                ),
                'fov': '70',
                'record': self.CAM_FRONT_LEFT_RECORD
            },
            'CAM_BACK_LEFT': {
                'transform': carla.Transform(
                    carla.Location(x=-0.7, y=0, z=2),
                    carla.Rotation(yaw=-110, pitch=0, roll=0)
                ),
                'fov': '70',
                'record': self.CAM_BACK_LEFT_RECORD
            },
            'CAM_BACK': {
                'transform': carla.Transform(
                    carla.Location(x=-1.5, y=0, z=2),
                    carla.Rotation(yaw=180, pitch=0, roll=0)
                ),
                'fov': '110',
                'record': self.CAM_BACK_RECORD
            },
            'CAM_BACK_RIGHT': {
                'transform': carla.Transform(
                    carla.Location(x=-0.7, y=0, z=2),
                    carla.Rotation(yaw=110, pitch=0, roll=0)
                ),
                'fov': '70',
                'record': self.CAM_BACK_RIGHT_RECORD
            }
        }

        self.cameras: Dict[str, CAMActor] = {}

    def tick(self):
        """
        Tick to simulation synchronization
        """
        # -----------------
        # sumo-->carla sync
        # -----------------
        self.sumo.tick()

        # Spawning new sumo actors in carla (i.e, not controlled by carla).
        sumo_spawned_actors = self.sumo.spawned_actors - set(self.carla2sumo_ids.values())
        for sumo_actor_id in sumo_spawned_actors:
            self.sumo.subscribe(sumo_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            carla_blueprint = BridgeHelper.get_carla_blueprint(sumo_actor, self.sync_vehicle_color)
            if carla_blueprint is not None:
                carla_transform = BridgeHelper.get_carla_transform(
                    sumo_actor.transform,
                    sumo_actor.extent)

                carla_actor_id = self.carla.spawn_actor(carla_blueprint, carla_transform)
                if carla_actor_id != INVALID_ACTOR_ID:
                    self.sumo2carla_ids[sumo_actor_id] = carla_actor_id
            else:
                self.sumo.unsubscribe(sumo_actor_id)

        # Destroying sumo arrived actors in carla.
        for sumo_actor_id in self.sumo.destroyed_actors:
            if sumo_actor_id in self.sumo2carla_ids:
                self.carla.destroy_actor(self.sumo2carla_ids.pop(sumo_actor_id))

        # Updating sumo actors in carla.
        for sumo_actor_id in self.sumo2carla_ids:
            carla_actor_id = self.sumo2carla_ids[sumo_actor_id]

            try:
                sumo_actor = self.sumo.get_actor(sumo_actor_id)
            except KeyError:
                continue
            carla_actor = self.carla.get_actor(carla_actor_id)

            carla_transform = BridgeHelper.get_carla_transform(
                sumo_actor.transform,
                sumo_actor.extent)
            if self.sync_vehicle_lights:
                carla_lights = BridgeHelper.get_carla_lights_state(
                    carla_actor.get_light_state(),
                    sumo_actor.signals)
            else:
                carla_lights = None

            self.carla.synchronize_vehicle(carla_actor_id, carla_transform, carla_lights)

        # Updates traffic lights in carla based on sumo information.
        if self.tls_manager == 'sumo':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                sumo_tl_state = self.sumo.get_traffic_light_state(landmark_id)
                carla_tl_state = BridgeHelper.get_carla_traffic_light_state(sumo_tl_state)

                self.carla.synchronize_traffic_light(landmark_id, carla_tl_state)

        # -----------------
        # carla-->sumo sync
        # -----------------
        self.carla.tick()

        # Spawning new carla actors (not controlled by sumo)
        carla_spawned_actors = self.carla.spawned_actors - set(self.sumo2carla_ids.values())
        for carla_actor_id in carla_spawned_actors:
            carla_actor = self.carla.get_actor(carla_actor_id)

            type_id = BridgeHelper.get_sumo_vtype(carla_actor)
            color = carla_actor.attributes.get('color', None) if self.sync_vehicle_color else None
            if type_id is not None:
                sumo_actor_id = self.sumo.spawn_actor(type_id, color)
                if sumo_actor_id != INVALID_ACTOR_ID:
                    self.carla2sumo_ids[carla_actor_id] = sumo_actor_id
                    self.sumo.subscribe(sumo_actor_id)


        # Destroying required carla actors in sumo.
        for carla_actor_id in self.carla.destroyed_actors:
            if carla_actor_id in self.carla2sumo_ids:
                self.sumo.destroy_actor(self.carla2sumo_ids.pop(carla_actor_id))

        # Updating carla actors in sumo.
        for carla_actor_id in self.carla2sumo_ids:
            sumo_actor_id = self.carla2sumo_ids[carla_actor_id]

            carla_actor = self.carla.get_actor(carla_actor_id)
            sumo_actor = self.sumo.get_actor(sumo_actor_id)

            sumo_transform = BridgeHelper.get_sumo_transform(
                carla_actor.get_transform(),
                carla_actor.bounding_box.extent)
            if self.sync_vehicle_lights:
                carla_lights = self.carla.get_actor_light_state(carla_actor_id)
                if carla_lights is not None:
                    sumo_lights = BridgeHelper.get_sumo_lights_state(
                        sumo_actor.signals,
                        carla_lights)
                else:
                    sumo_lights = None
            else:
                sumo_lights = None

            self.sumo.synchronize_vehicle(sumo_actor_id, sumo_transform, sumo_lights)

        # Updates traffic lights in sumo based on carla information.
        if self.tls_manager == 'carla':
            common_landmarks = self.sumo.traffic_light_ids & self.carla.traffic_light_ids
            for landmark_id in common_landmarks:
                carla_tl_state = self.carla.get_traffic_light_state(landmark_id)
                sumo_tl_state = BridgeHelper.get_sumo_traffic_light_state(carla_tl_state)

                # Updates all the sumo links related to this landmark.
                self.sumo.synchronize_traffic_light(landmark_id, sumo_tl_state)

    def moveSpectator(self, vehicle):
        spectator = self.carla.world.get_spectator()
        transform = vehicle.get_transform()
        offset_x = 0
        offset_y = 0
        offset_z = 50
        location = transform.location + carla.Location(x=offset_x, y=offset_y, z=offset_z)
        spectator.set_transform(carla.Transform(location,carla.Rotation(pitch=-90 ,yaw=0,roll=0)))

    def allCamBeenSetted(self) -> bool:
        return self.CAM_FONRT_RECORD and self.CAM_FRONT_LEFT_RECORD \
            and self.CAM_FRONT_RIGHT_RECORD and self.CAM_BACK_LEFT_RECORD \
            and self.CAM_BACK_RECORD and self.CAM_BACK_RIGHT_RECORD

    def setCameras(self, vehicle) -> None:
        if self.allCamBeenSetted():
            return
        else:
            for k, v in self.cameraInfo.items():
                if not v['record']:
                    self.cameras[k] = CAMActor(
                        k, self, v['transform'], v['fov'], vehicle
                    )
                    v['record'] = True
                else:
                    continue

    def getCameraImages(self) -> Optional[CameraImages]:
        images = {}
        for k, v in self.cameras.items():
            pic = v.getImage()
            if isinstance(pic, np.ndarray):
                images[k] = pic
            else:
                return None
        return CameraImages(
            images['CAM_FRONT'], images['CAM_FRONT_RIGHT'], images['CAM_FRONT_LEFT'],
            images['CAM_BACK_LEFT'], images['CAM_BACK'], images['CAM_BACK_RIGHT']
        )

    def getEgo(self):
        if self.ego_id in self.sumo2carla_ids:
            vehicle = self.carla.get_actor(self.sumo2carla_ids[self.ego_id])
            return vehicle
        else:
            return None


    def close(self):
        """
        Cleans synchronization.
        """
        # Configuring carla simulation in async mode.
        settings = self.carla.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.carla.world.apply_settings(settings)

        # Destroying synchronized actors.
        for carla_actor_id in self.sumo2carla_ids.values():
            self.carla.destroy_actor(carla_actor_id)

        for sumo_actor_id in self.carla2sumo_ids.values():
            self.sumo.destroy_actor(sumo_actor_id)

        # Closing sumo and carla client.
        self.carla.close()
        # self.sumo.close()

# reference: https://github.com/cf206cd/carla_nuscenes/blob/main/carla_nuscenes/sensor.py
class CAMActor:
    def __init__(
        self, name: str, sync: SimulationSynchronization,
        transform: carla.Transform, fov: str,
        vehicle, max_size: int = 5
    ) -> None:
        """
        Initializes the CameraSensor object.

        Args:
            name (str): The name of the CameraSensor object.
            sync (SimulationSynchronization): The SimulationSynchronization object.
            transform (carla.Transform): The transform of the CameraSensor object.
            fov (str): The field of view of the CameraSensor object.
            vehicle: The vehicle to attach the CameraSensor object to.
            max_size (int, optional): The maximum size of the image list. Defaults to 5.

        Returns:
            None
        """
        self.name = name
        cam_bp = sync.carla.world.get_blueprint_library().find('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', '1600')
        cam_bp.set_attribute('image_size_y', '900')
        cam_bp.set_attribute('fov', fov)
        # cam_bp.set_attribute('sensor_tick', '0.1')
        self.cam = sync.carla.world.spawn_actor(
            cam_bp, transform, attach_to=vehicle
        )
        self.cam.listen(self.process)
        self.max_size = max_size
        self.imageList = []

    def process(self, image):
        image_data = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        image_data = np.reshape(image_data, (image.height, image.width, 4))
        # image_data rgb 
        image_data = image_data[:,:,[2,1,0,3]]
        self.addImage(image_data)

    def addImage(self, image_data):
        if len(self.imageList) >= self.max_size:
            self.imageList.pop(0)
        self.imageList.append(image_data)

    def getImage(self):
        return self.imageList[-1] if self.imageList else None


    
class Arguments:
    def __init__(
            self, sumo_cfg_file: str,
            sumo_gui: bool,
            carla_host: str,
            carla_port: int,
            step_length: float,
            tls_manager: str,
            sync_vehicle_color: bool = True,
            sync_vehicle_lights: bool = True,
            ) -> None:
        self.sumo_cfg_file = sumo_cfg_file
        self.sumo_gui = sumo_gui
        self.carla_host = carla_host
        self.carla_port = carla_port
        self.step_length = step_length
        self.tls_manager = tls_manager   # ['none', 'sumo', 'carla']
        self.sync_vehicle_color = sync_vehicle_color
        self.sync_vehicle_lights = sync_vehicle_lights


def synchronization_loop(args: Arguments, ego_id):
    """
    Entry point for sumo-carla co-simulation.
    """
    sumo_simulation = SumoSimulation(args.sumo_cfg_file)
    carla_simulation = CarlaSimulation(args.carla_host, args.carla_port, args.step_length)

    synchronization = SimulationSynchronization(
        sumo_simulation, carla_simulation, ego_id, args.tls_manager,
        args.sync_vehicle_color, args.sync_vehicle_lights)

    cnt = 0
    try:
        while traci.simulation.getMinExpectedNumber():
            start = time.time()

            # 这里把 traci.simulationStep() 提取出来是为了方便后续 LimSim 和 CARLA 的
            # 协同，简单来说就是 traci.simulationStep() 以后会在 LimSim 那边调用
            traci.simulationStep()
            synchronization.tick()

            vehicle = synchronization.getEgo()
            if vehicle:
                synchronization.moveSpectator(vehicle)

            end = time.time()
            elapsed = end - start
            if elapsed < args.step_length:
                time.sleep(args.step_length - elapsed)

            cnt += 1

    except KeyboardInterrupt:
        logging.info('Cancelled by user.')

    finally:
        logging.info('Cleaning synchronization')

        synchronization.close()


def getSynchronization(
    sumo_cfg_file: str, carla_host: str, carla_port: int,
    ego_id: str, step_length: float, tls_manager: str, 
    sync_vehicle_color: bool, sync_vehicle_lights: bool
) -> SimulationSynchronization:
    """
    Initializes and returns a SimulationSynchronization object that synchronizes a SUMO simulation and a CARLA simulation.

    Caution: This function MUST be called after `model.start()`, where `model` refers to `simModel.egoTracking.model.Model` or `simModel.egoTracking.replay.Model`.

    Args:
        sumo_cfg_file (str): The path to the SUMO configuration file.
        carla_host (str): The host address of the CARLA server.
        carla_port (int): The port number of the CARLA server.
        ego_id (str): The ID of the ego vehicle in CARLA.
        step_length (float): The time step length for simulation synchronization.
        tls_manager (str): The name of the traffic light system manager.
        sync_vehicle_color (bool): Whether to synchronize the color of vehicles between SUMO and CARLA.
        sync_vehicle_lights (bool): Whether to synchronize the lights of vehicles between SUMO and CARLA.

    Returns:
        SimulationSynchronization: A SimulationSynchronization object that manages the synchronization between SUMO and CARLA simulations.
    """
    sumo_simulation = SumoSimulation(sumo_cfg_file)
    carla_simulation = CarlaSimulation(
        carla_host, carla_port, step_length
    )
    # carla_simulation.client.reload_world()
    # time.sleep(1)
    synchronization = SimulationSynchronization(
        sumo_simulation, carla_simulation, 
        ego_id, tls_manager,
        sync_vehicle_color, 
        sync_vehicle_lights
    )

    return synchronization


if __name__ == '__main__':
    debug = 0

    arguments = Arguments(
        sumo_cfg_file='./networkFiles/CarlaTown06/Town06.sumocfg',
        sumo_gui=True,
        carla_host='127.0.0.1',
        carla_port=2000,
        step_length=0.1,
        tls_manager='sumo',
        sync_vehicle_color=False,
        sync_vehicle_lights=True
    )

    if debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    traci.start(
        [
            'sumo-gui' if arguments.sumo_gui else 'sumo',
            '-n', './networkFiles/CarlaTown06/Town06.net.xml',
            '-r', './networkFiles/CarlaTown06/carlavtypes.rou.xml,./networkFiles/CarlaTown06/Town06.rou.xml',
            '--step-length', str(arguments.step_length),
            '--lateral-resolution', '10',
            '--start', '--quit-on-end',
            '-W', '--collision.action', 'remove',
        ], port=8813
    )

    synchronization_loop(arguments, '10')

