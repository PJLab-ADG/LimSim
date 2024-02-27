# This script is a running example of Co-simulation

from datetime import datetime
import re
import cv2
import base64
import numpy as np
import requests
import traci

from simModel.Model import Model
from simModel.MPGUI import GUI
from trafficManager.traffic_manager import TrafficManager
from simInfo.Informer import Informer
from trafficManager.common.vehicle import Behaviour
from simModel.DataQueue import QuestionAndAnswer

ego_id = '10'
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
database = './results/' + stringTimestamp + '.db'

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

def request_openai(content, max_tokens: int = 4000):
        import os
        api_key = os.environ.get('OPENAI_API_KEY')

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": max_tokens
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        return response.json()

def str2behavior(decision) -> Behaviour:
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

if __name__ == '__main__':
    model = Model(
        egoID=ego_id, netFile=sumo_net_file, rouFile=sumo_rou_file,
        cfgFile=sumo_cfg_file, dataBase=database, SUMOGUI=sumo_gui,
        CARLACosim=True, carla_host=carla_host, carla_port=carla_port
    )
    model.start()
    planner = TrafficManager(model)
    informer = Informer()

    gui = GUI(model)
    gui.start()

    while not model.tpEnd:
        model.moveStep()

        if model.timeStep %20 ==0:# 20 frame
            
            roadgraph, vehicles = model.exportSce()

            if model.tpStart and roadgraph:
                actionInfo = informer.getActionInfo(vehicles, roadgraph)
                naviInfo = informer.getNaviInfo(vehicles)
                if naviInfo:
                    information = actionInfo + naviInfo
                else:
                    information = actionInfo
                print('information: ', information)

                images = model.getCARLAImage(5,1)
                front_img = images[-1].CAM_FRONT
                front_left_img = images[-1].CAM_FRONT_LEFT
                front_right_img = images[-1].CAM_FRONT_RIGHT

                if isinstance(front_img, np.ndarray):
                    _, buffer = cv2.imencode('.png', front_img)
                    front_image_base64 = base64.b64encode(buffer).decode('utf-8')
                    # print('front_image_base64: ', front_image_base64)

                    message=[]
                    message.append({
                        "type":"text",
                        "text": SYSTEM_PROMPT,
                    })
                    message.append({
                        "type":"text",
                        "text": "The current frame information is:\n"+information,
                    })
                    message.append({    
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{front_image_base64}",
                            "detail": "low"
                            }
                    })
                    response = request_openai(message)
                    ans = response['choices'][0]['message']['content']
                    print("GPT-4V:",ans)


                    match = re.search(r'## Decision\n(.*)', ans)
                    behavior = None
                    if match:
                        decision = match.group(1)
                        behavior = str2behavior(decision)
                    else:
                        print("ERROR: cannot find proper decision")
                    print ('behavior: ', behavior)
                    
                    qa = QuestionAndAnswer()
                    qa.navigation = naviInfo
                    qa.actions = actionInfo
                    qa.response = ans
                    model.putQA(qa)

                if behavior is not None:
                    trajectories = planner.plan(
                                        model.timeStep * 0.1, roadgraph, vehicles, behavior)
                    model.setTrajectories(trajectories)


        model.updateVeh()
    
    model.destroy()
    gui.join()
    gui.terminate()