
import os
import textwrap
import time
from rich import print
from typing import List, Tuple
from datetime import datetime
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from DriverAgent.Memory import DrivingMemory
from DriverAgent.EnvDescriptor import EnvDescription
from simModel.Model import Model
from simModel.MPGUI import GUI
from trafficManager.traffic_manager import TrafficManager, LaneChangeException
from langchain.callbacks.openai_info import OpenAICallbackHandler

from simModel.DataQueue import QuestionAndAnswer

import logger, logging
from trafficManager.common.vehicle import Behaviour

from utils.trajectory import Rectangle, RecCollide

import sqlite3

decision_logger = logger.setup_app_level_logger(logger_name="LLMAgent", file_name="llm_decision.log")
LLM_logger = logging.getLogger("LLMAgent").getChild(__name__)


class CollisionException(Exception):
    def __init__(self, ErrorInfo: str) -> None:
        super().__init__(self)
        self.errorinfo = ErrorInfo
    
    def __str__(self) -> str:
        return self.errorinfo
    
class BrainDeadlockException(Exception):
    def __init__(self) -> None:
        super().__init__(self)
        self.errorinfo = "Your reasoning and decision-making result is in deadlock."

    def __str__(self) -> str:
        return self.errorinfo
    
class TimeOutException(Exception):
    def __init__(self) -> None:
        super().__init__(self)
        self.errorinfo = "You failed to complete the route within 100 seconds, exceeding the allotted time."

    def __str__(self) -> str:
        return self.errorinfo

class CollisionChecker:
    def __init__(self):
        pass
    
    # 应该在model里面，实施检测碰撞，还有没有路径，或者两个地方都搞一个也行
    def CollisionCheck(self, model: Model) -> bool:
        # vehicle trajectory collision need to be checked in every frame
        for key, value in model.ms.vehINAoI.items():
            if value.id == model.ms.ego.id:
                continue
            recA = Rectangle([model.ms.ego.x, model.ms.ego.y],
                                model.ms.ego.length, model.ms.ego.width, model.ms.ego.yaw)
            recB = Rectangle([value.x, value.y],
                                value.length, value.width, value.yaw)
            rc = RecCollide(recA, recB)
            # if the car collide, stop the simulation
            if rc.isCollide():
                raise CollisionException("you have a collision with vehicle {}".format(key))
        return False

class LLMAgent:
    def __init__(
        self, use_memory: bool = True, delimiter: str = "####"
    ) -> None:
        oai_api_type = os.getenv("OPENAI_API_TYPE")
        if oai_api_type == "azure":
            print("Using Azure Chat API")
            self.llm = AzureChatOpenAI(
                deployment_name="wrz", #"GPT-16"
                temperature=0,
                max_tokens=2000,
                request_timeout=60,
            )
        elif oai_api_type == "openai":
            self.llm = ChatOpenAI(
                temperature=0,
                model_name= 'gpt-4',
                max_tokens=2000,
                request_timeout=60,
            )
        db_path = os.path.dirname(os.path.abspath(__file__)) + "/db/" + "decision_mem/"
        self.agent_memory = DrivingMemory(db_path=db_path)
        self.few_shot_num = 3

        self.use_memory = use_memory
        self.delimiter = delimiter

        self.llm_source = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0
        }

    def getLLMSource(self, cb: OpenAICallbackHandler, init: bool):
        if init:
            self.llm_source["prompt_tokens"] = cb.prompt_tokens
            self.llm_source["completion_tokens"] = cb.completion_tokens
            self.llm_source["total_tokens"] = cb.total_tokens
            self.llm_source["cost"] = cb.total_cost
        else:
            self.llm_source["prompt_tokens"] += cb.prompt_tokens
            self.llm_source["completion_tokens"] += cb.completion_tokens
            self.llm_source["total_tokens"] += cb.total_tokens
            self.llm_source["cost"] += cb.total_cost
        return 

    def makeDecision(self, information = None, image_base64 = None, descriptions: list = None) -> Tuple[int, str, str, str, dict]:
        # step1. get the scenario description
        scenario_description = descriptions[0]
        available_actions = descriptions[1]
        driving_intensions = descriptions[2]

        # step2. get the few shot examples
        if self.use_memory:
            key = "## Driving scenario description:\n" + scenario_description + "\n## Navigation instruction:\n" + driving_intensions
            key = key.replace("'", "")
            fewshot_results = self.agent_memory.retriveMemory(
                key, self.few_shot_num)
            fewshot_messages = []
            fewshot_answers = []
            fewshot_actions = []
            for fewshot_result in fewshot_results:
                fewshot_messages.append(fewshot_result["human_question"])
                fewshot_answers.append(fewshot_result["LLM_response"])
                fewshot_actions.append(fewshot_result["action"])

                print(fewshot_actions)

            if len(fewshot_actions) == 0:
                LLM_logger.warning("There is no memory!")
        else:
            fewshot_messages = []
        # step3. get system message and make prompt
        with open(os.path.dirname(os.path.abspath(__file__)) + "/text_example/system_message_v-1.txt", "r") as f:
            system_message = f.read()

        with open(os.path.dirname(os.path.abspath(__file__)) + "/text_example/example_QA1.txt", "r") as f:
            example = f.read()
            example_message = example.split("======")[0]
            example_answer = example.split("======")[1]

        human_message = textwrap.dedent(f"""\
        Here is the current scenario:
        ## Driving scenario description:
        {scenario_description}

        ## Navigation instruction:
        {driving_intensions}

        ## Available actions:
        {available_actions}

        ## 
        Remember to follow the format instructions.
        You can stop reasoning once you have a valid action to take. 
        """)

        messages = [
            SystemMessage(content=system_message.format(delimiter = self.delimiter)),
            HumanMessage(content=example_message),
            AIMessage(content=example_answer),
        ]

        if len(fewshot_messages) > 0:
            for i in range(len(fewshot_messages)):
                messages.append(
                    HumanMessage(content=fewshot_messages[i])
                )
                messages.append(
                    AIMessage(content=fewshot_answers[i])
                )
                messages.append(HumanMessage(content="Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario."))
        messages.append(
            HumanMessage(content=human_message)
        )

        # step4. get the response
        with get_openai_callback() as cb:
            response = self.llm(messages)
            self.getLLMSource(cb, True)

        # step5. get the decision and check the output
        decision_action = response.content.split(self.delimiter)[-1]
        try:
            result = int(decision_action)
            if result not in [1, 2, 3, 4, 8]:
                raise ValueError
        except ValueError:
            LLM_logger.warning("--------- Output is not available, checking the output... ---------")
            check_message = f"""
            You are a output checking assistant who is responsible for checking the output of another agent.
            
            The output you received is: {decision_action}

            Your should just output the right int type of action_id, with no other characters or delimiters.
            i.e. :
            | Action_id | Action Description                                     |
            |--------|--------------------------------------------------------|
            | 3      | Turn-left: change lane to the left of the current lane |
            | 8      | IDLE: remain in the current lane with current speed   |
            | 4      | Turn-right: change lane to the right of the current lane|
            | 1      | Acceleration: accelerate the vehicle                 |
            | 2      | Deceleration: decelerate the vehicle                 |


            You answer format would be:
            {self.delimiter} <correct action_id within [3,8,4,1,2]>
            """
            messages = [
                HumanMessage(content=check_message),
            ]
            with get_openai_callback() as cb:
                check_response = self.llm(messages)
                LLM_logger.info(f"Checking question: {check_message}")
                LLM_logger.info(f"Output checking result: {check_response.content}")
                self.getLLMSource(cb, False)
            result = int(check_response.content.split(self.delimiter)[-1])

        few_shot_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_store += "human question: \n" + fewshot_messages[i] + "\nResponse:\n" + fewshot_answers[i] + \
                "\n---------------\n"
        print("Result:", result)
        
        if result not in [1, 2, 3, 4, 8]:
            result = 8
            LLM_logger.error(f"Output is still not available")
        else:
            LLM_logger.info(f"Output is available, the decision is {result}")
            
        LLM_logger.info(f"Tokens Used: {self.llm_source['total_tokens']}\n"
            f"\tPrompt Tokens: {self.llm_source['prompt_tokens']}\n"
            f"\tCompletion Tokens: {self.llm_source['completion_tokens']}\n"
            f"Total Cost (USD): ${self.llm_source['cost']}")
        
        return result, response.content, human_message, few_shot_store, self.llm_source

def record_result(model: Model, start_time: float, result: bool, reason: str = "", error: Exception = None) -> None:
    conn = sqlite3.connect(model.dataBase)
    cur = conn.cursor()
    # add result data
    cur.execute(
        """INSERT INTO resultINFO (
            egoID, result, total_score, complete_percentage, drive_score, use_time, fail_reason
            ) VALUES (?,?,?,?,?,?,?);""",
        (
            model.ms.ego.id, result, 0, 0, 0, time.time() - start_time, reason
        )
    )
    conn.commit()
    conn.close()
    return 

if __name__ == "__main__":
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
    database = './results/' + stringTimestamp + '.db'

    # init LLMDriver
    model = Model(
        egoID=ego_id, netFile=sumo_net_file, rouFile=sumo_rou_file,
        cfgFile=sumo_cfg_file, dataBase=database, SUMOGUI=sumo_gui,
        CARLACosim=False, carla_host=carla_host, carla_port=carla_port
    )
    planner = TrafficManager(model)
    agent = LLMAgent(use_memory=False)
    descriptor = EnvDescription(planner.config)
    collision_checker = CollisionChecker()
    model.start()

    gui = GUI(model)
    gui.start()

    action_list = []
    total_start_time = time.time()
    try:
        while not model.tpEnd:
            model.moveStep()
            
            # check collision
            # TODO: current lane更新慢10s,是因为状态更新不及时，应该拉到0.5s更新一次
            collision_checker.CollisionCheck(model)
            if model.timeStep % 10 == 0:
                roadgraph, vehicles = model.exportSce()
                if model.tpStart and roadgraph:
                    LLM_logger.info(f"--------------- timestep is {model.timeStep} ---------------")
                    descriptions = descriptor.getDescription(
                        roadgraph, vehicles, planner, model.timeStep * 0.1)
                    start_time = time.time()
                    ego_behaviour, response, human_question, fewshot, llm_cost = agent.makeDecision("", "", descriptions)
                    # if "Change" in descriptions[2]:
                    #     ego_behaviour = 3
                    # else:
                    #     ego_behaviour = 8
                    descriptor.decision = ego_behaviour
                    current_QA = QuestionAndAnswer(descriptions[0], descriptions[2], descriptions[1], fewshot, response, llm_cost["prompt_tokens"], llm_cost["completion_tokens"], llm_cost["total_tokens"], time.time()-start_time, ego_behaviour)

                    model.putQA(current_QA)
                    trajectories = planner.plan(
                        model.timeStep * 0.1, roadgraph, vehicles, Behaviour(ego_behaviour), other_plan=False
                    )
                    action_list.append(ego_behaviour)
                    if len(action_list) > 10:
                        last_10_actions = action_list[-10::]
                        last_10_actions.sort()
                        if last_10_actions[0] == last_10_actions[-1]:
                            raise BrainDeadlockException()
                    if len(action_list) > 100:
                        raise TimeoutError()
                    model.setTrajectories(trajectories)
                else:
                    model.ego.exitControlMode()

            
            model.updateVeh()

    except (CollisionException, LaneChangeException, BrainDeadlockException, TimeoutError) as e:
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

    