
import os
import textwrap
import time
from rich import print
from typing import List

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from DriverAgent.GT.loadConfig import load_openai_config
from DriverAgent.GT.vectorStore import DrivingMemory

from DriverAgent.GT.promptDB import DBBridge
import json
from trafficManager.common.vehicle import Behaviour

USE_MEMORY = False
delimiter = "####"
example_message = textwrap.dedent(f"""\
        {delimiter} Driving scenario description:
        You are driving on a road with 4 lanes in your direction, and you are currently driving in the number 4 lane from the left. The length of the current lane is 483.2 $m$.The limit speed of the current lane is 13.89 $m/s$. You can drive on the lane you are currently driving. 
        Your current position is '(-11.2,497.147)', speed is 13.90 $m/s$, acceleration is 0.0 $m/s^2$, and lane position is 2.85 $m$. 
        There are other vehicles driving around you, and below is their basic information:
        - Vehicle '8' is driving on the same lane as you and is ahead of you. The position of it is (-10.525,454.37), speed is 12.861 $m/s$, acceleration is 0.0 $m/s^2$, and lane position is 12.35 $m$. The distance between you and vehicle '8' is 9.5 $m$. 

        {delimiter} Your available actions:
        IDLE - remain in the current lane with current speed Action_id: 8
        Turn-left - change lane to the left of the current lane Action_id: 3
        Turn-right - change lane to the right of the current lane Action_id: 4
        Acceleration - accelerate the vehicle Action_id: 1
        Deceleration - decelerate the vehicle Action_id: 2
        """)
example_answer = textwrap.dedent(f"""\
        I have five actions to choose from. Next, I need to consider what action I should choose in terms of whether or not I will be in a collision, whether or not I need to change lanes, whether or not I am approaching an junction, and whether or not I am approaching the speed limit on the road.
        
        - First, I need to avoid a collision. I need to consider the vehicles in front of me and the potential for collision. Attention, the vehicles behind me is no need to consider. In the current scenario, vehical `8` is in front of me and I need to specifically analyze the likelihood of a collision and how I can avoid it. First of all, when I am less than 10 meters from the vehicle in front of me, I need to consider whether a collision will occur. Since Vehicle 8 is 9.5 m in front of me, which is less than the judgment distance, I should consider Vehicle 8. Since my speed is faster than Vehicle 8's speed by 13.90 - 12.861 = 1.039 m/s, if I keep my current speed, I will collide with Vehicle 8 after 9.5 / 1.039 = 9.14 s. So I should decelerate.
        Great, I can make my decision now. Decision: Deceleration
        
        Response to user:#### 2
        """)

ENVSTATE = {
    0: "normal lane",
    1: "approaching junction without traffic light",
    2: "approaching junction with traffic light",
    3: "in junction",
    4: "change lane"
}


class LLMAgent:
    def __init__(
        self
    ) -> None:
        load_openai_config()
        oai_api_type = os.getenv("OPENAI_API_TYPE")
        if oai_api_type == "azure":
            print("Using Azure Chat API")
            self.llm = AzureChatOpenAI(
                # streaming=True,
                # callbacks=[
                #     StreamingStdOutCallbackHandler(),
                #     OpenAICallbackHandler()
                # ],
                deployment_name="wrz",
                temperature=0,
                max_tokens=2000,
                request_timeout=60,
            )
        elif oai_api_type == "openai":
            self.llm = ChatOpenAI(
                temperature=0,
                model_name= 'gpt-3.5-turbo-16k', #'gpt-4',  'gpt-3.5-turbo-16k'# or any other model with 8k+ context
                max_tokens=2000,
                request_timeout=60,
            )
        db_path = os.path.dirname(os.path.abspath(__file__)) + "/db/" + "chroma_5_shot_20_mem/"
        self.agent_memory = DrivingMemory(db_path=db_path)
        self.few_shot_num = 3

        self.database = "egoTrackingTest.db"
        self.dbBridge = DBBridge(self.database)
        self.dbBridge.createTable()

        with open(os.path.dirname(os.path.abspath(__file__)) + "/GT/few_shot.json", "r", encoding="utf-8") as f:
            self.few_shot_json = json.load(f)

    def makeDecision(self, information = None, image_base64 = None, descriptions: list = None):
        # for template usage refer to: https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/

        scenario_description = descriptions[0]
        available_actions = descriptions[1]
        driving_intensions = descriptions[2]
        state = descriptions[3]

        if USE_MEMORY:
            fewshot_results = self.agent_memory.retriveMemory(
                scenario_description, self.few_shot_num)
            fewshot_messages = []
            fewshot_answers = []
            fewshot_actions = []
            for fewshot_result in fewshot_results:
                fewshot_messages.append(fewshot_result["human_question"])
                fewshot_answers.append(fewshot_result["LLM_response"])
                fewshot_actions.append(fewshot_result["action"])
                mode_action = max(
                    set(fewshot_actions), key=fewshot_actions.count)
                mode_action_count = fewshot_actions.count(mode_action)
            if len(fewshot_actions) == 0:
                print("fewshot_actions ERRORS!: ", fewshot_actions)
                exit(1)
            
        else:
            fewshot_messages = []
            fewshot_answers = []
            for item in self.few_shot_json["init_few_shot"][ENVSTATE[state]]:
                fewshot_messages.append(item["human_question"])
                fewshot_answers.append(item["response"])

        system_message = textwrap.dedent(f"""\
        You are ChatGPT, a large language model trained by OpenAI. Now you act as a mature driving assistant, who can give accurate and correct advice for human driver in complex urban driving scenarios.
        You will be given a detailed description of the driving scenario of current frame along with your history of previous decisions. You will also be given the available actions you are allowed to take. All of these elements are delimited by {delimiter}.

        Your response should use the following format:
        <reasoning>
        <reasoning>
        <repeat until you have a decision>
        Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`> 

        Make sure to include {delimiter} to separate every step.
        """)

        human_message = f"""\
        Above messages are some examples of how you make a decision successfully in the past. Those scenarios are similar to the current scenario. You should refer to those examples to make a decision for the current scenario. P.S. Be careful of examples which decision is change lanes, since change lanes is not a frequent action, you think twice and reconfirm before you change lanes.

        Here is the current scenario:
        {delimiter} Driving scenario description:
        {scenario_description}
        {delimiter} Driving Intensions:
        {driving_intensions}
        {delimiter} Available actions:
        {available_actions}

        You can stop reasoning once you have a valid action to take. 
        """
        human_message = human_message.replace("        ", "")

        if fewshot_messages is None:
            raise ValueError("fewshot_message is None")
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=example_message),
            AIMessage(content=example_answer),
        ]
        for i in range(len(fewshot_messages)):
            messages.append(
                HumanMessage(content=fewshot_messages[i])
            )
            messages.append(
                AIMessage(content=fewshot_answers[i])
            )
        messages.append(
            HumanMessage(content=human_message)
        )

        start_time = time.time()

        response = self.llm(messages)

        print("Time used: ", time.time() - start_time)

        decision_action = response.content.split(delimiter)[-1]
        try:
            result = int(decision_action)
            if result not in [1, 2, 3, 4, 8]:
                raise ValueError
        except ValueError:
            print("Output is not available, checking the output...")
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
            {delimiter} <correct action_id within [3,8,4,1,2]>
            """
            messages = [
                HumanMessage(content=check_message),
            ]
            with get_openai_callback() as cb:
                check_response = self.llm(messages)
            result = int(check_response.content.split(delimiter)[-1])

        few_shot_store = ""
        for i in range(len(fewshot_messages)):
            few_shot_store += "human question: \n" + fewshot_messages[i] + "\nResponse:\n" + fewshot_answers[i] + \
                "\n---------------\n"
        print("Result:", result)
        
        if result not in [1, 2, 3, 4, 8]:
            result = 8
            decision_make = False
        else:
            decision_make = True
        return Behaviour(result), response.content, human_message, few_shot_store, decision_make

