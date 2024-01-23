from datetime import datetime
import re
import time
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.callbacks import get_openai_callback

import cv2
import base64
import numpy as np
from rich import print
from typing import Dict, List


from simInfo.EnvDescriptor import EnvDescription
from trafficManager.traffic_manager import TrafficManager, LaneChangeException
from simModel.Model import Model
from simModel.MPGUI import GUI

from simModel.DataQueue import QuestionAndAnswer

import logger, logging
from trafficManager.common.vehicle import Behaviour
from simInfo.CustomExceptions import (
    CollisionException, LaneChangeException, 
    CollisionChecker, record_result
)

decision_logger = logger.setup_app_level_logger(logger_name="LLMAgent", file_name="llm_decision.log")
LLM_logger = logging.getLogger("LLMAgent").getChild(__name__)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def NPImageEncode(npimage: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', npimage)
    npimage_base64 = base64.b64encode(buffer).decode('utf-8')
    return npimage_base64
    

def getText(filePath: str) -> str:
    with open(filePath, 'r') as f:
        res = f.read()
        return res
    

def addTextPrompt(content: List, textPrompt: str) -> Dict[str, str]:
    textPrompt = {
        "type": "text",
        "text": textPrompt
    }
    content.append(textPrompt)

def addImagePrompt(content: List, imagePath: str):
    base64_image = encode_image(imagePath)
    imagePrompt = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
            "detail": "low"
        }
    }
    content.append(imagePrompt)

def addImageBase64(content: List, image_base64: str):
    imagePrompt = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_base64}",
            "detail": "low"
        }
    }
    content.append(imagePrompt)



class VLMAgent:
    def __init__(self, vlm: ChatOpenAI) -> None:
        self.vlm = vlm

    def str2behavior(self, decision: str) -> Behaviour:
            if decision == 'IDLE':
                return Behaviour.IDLE
            elif decision == 'AC':
                return Behaviour.AC
            elif decision == 'DC':
                return Behaviour.DC
            elif decision == 'LCR':
                return Behaviour.LCR
            elif decision == 'LCL':
                return Behaviour.LCL
            else:
                errorStr = f'The decision `{decision}` is not implemented yet!'
            raise NotImplementedError(errorStr)


    def makeDecision(self, content: List[Dict[str, str]]):
        message = HumanMessage(content=content)
        start = time.time()
        with get_openai_callback() as cb:
            ans = self.vlm.invoke([message])
        end = time.time()
        timeCost = end - start
        print('[green]GPT-4V: {}[/green]'.format(ans))
        match = re.search(r'## Decison\n(.*)\n', ans)
        behavior = None
        if match:
            decision = match.group(1)
            behavior = self.str2behavior(decision)
        else:
            raise ValueError('GPT-4V did not return a valid decision')
        return behavior, ans, cb, timeCost
    

SYSTEM_PROMPT = """
You are GPT-4V(ision), a large multi-modal model trained by OpenAI. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios. You'll receive some images from the onboard camera. You'll need to make driving inferences and decisions based on the information in the images. At each decision frame, you receive navigation information and a collection of actions. You will perform scene description, and reasoning based on the navigation information and the front-view image. Eventually you will select the appropriate action output from the action set.
Make sure that all of your reasoning is output in the `## Reasoning` section, and in the `## Decision` section you should only output the name of the action, e.g. `AC`, `IDLE` etc.

Your answer should follow this format:
## Description
Your description of the front-view image.
## Reasoning
reasoning based on the navigation information and the front-view image.
## Decison
one of the actions in the action set.(SHOULD BE exactly same and no other words!)
"""
if __name__ == '__main__':
    ego_id = '139'
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

    stringTimestamp = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
    database = './experiments/zeroshot/gpt4v/' + stringTimestamp + '.db'

    # init LLMDriver
    model = Model(
        egoID=ego_id, netFile=sumo_net_file, rouFile=sumo_rou_file,
        cfgFile=sumo_cfg_file, dataBase=database, SUMOGUI=sumo_gui,
        CARLACosim=False, carla_host=carla_host, carla_port=carla_port
    )
    planner = TrafficManager(model)
    descriptor = EnvDescription(planner.config)
    collision_checker = CollisionChecker()
    model.start()

    gui = GUI(model)
    gui.start()

    gpt4v = VLMAgent(
        ChatOpenAI(
            temperature=0, 
            model="gpt-4-vision-preview", 
            max_tokens=1024
            )
    )

    total_start_time = time.time()
    try:
        while not model.tpEnd:
            model.moveStep()
            collision_checker.CollisionCheck(model)
            if model.timeStep % 10 == 0:
                roadgraph, vehicles = model.exportSce()
                if model.tpStart and roadgraph:
                    actionInfo = descriptor.availableActionsDescription(roadgraph)
                    naviInfo = descriptor.getNavigationInfo(roadgraph)
                    egoInfo = descriptor.getEgoInfo()
                    TotalInfo = '## Available actions\n\n' + actionInfo + '\n\n' + '## Navigation information\n\n' + naviInfo + egoInfo
                    images = model.getCARLAImage(1, 1)
                    front_img = images[-1].CAM_FRONT
                    front_img = images[-1].CAM_FRONT
                    front_left_img = images[-1].CAM_FRONT_LEFT
                    front_right_img = images[-1].CAM_FRONT_RIGHT
                    if isinstance(front_img, np.ndarray):
                        content = []
                        addTextPrompt(content, SYSTEM_PROMPT)
                        addTextPrompt(content, '## Camera Images\n\nThe next three images are images captured by the left front, front, and right front cameras.\n')
                        addImageBase64(content, NPImageEncode(front_left_img))
                        addImageBase64(content, NPImageEncode(front_img))
                        addImageBase64(content, NPImageEncode(front_right_img))
                        addTextPrompt(content, f'\nThe current frame information is:\n{TotalInfo}')
                        behaviour, ans, cb, timecost = gpt4v.makeDecision(content)
                        print('[blue]behavior: {}[/blue]'.format(behaviour))
                        model.putQA(
                            QuestionAndAnswer(
                                '', naviInfo+egoInfo, actionInfo, '', 
                                ans, cb.prompt_tokens,
                                cb.completion_tokens, cb.total_tokens, 
                                timecost, int(behaviour)
                            )
                        )

            model.updateVeh()
    except (CollisionException, LaneChangeException) as e:
        record_result(model, total_start_time, False, str(e))
        model.dbBridge.commitData()
    except Exception as e:
        model.dbBridge.commitData()
        raise e
    else:
        record_result(model, total_start_time, True, None)
    finally:
        model.destroy()
        gui.terminate()
        gui.join()