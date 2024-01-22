import os
import textwrap
import time
from rich import print
from langchain.callbacks import get_openai_callback
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

class ReflectionAssistant:
    def __init__(
        self
    ) -> None:
        oai_api_type = "azure"
        if oai_api_type == "azure":
            print("Using Azure Chat API")
            self.llm = AzureChatOpenAI(
            )
        elif oai_api_type == "openai":
            self.llm = ChatOpenAI(
                temperature=0,
                model_name= 'gpt-3.5-turbo-16k', 
                max_tokens=2000,
                request_timeout=60,
            )

    def reflection(self, human_message: str, llm_response: str, evaluation: str) -> str:
        delimiter = "####"
        system_message = textwrap.dedent(f"""
You are ChatGPT, a large language model trained by OpenAI. Now you are an advanced autonomous driving assistant, your goal is to support drivers in performing safe and efficient driving tasks. Here is a detailed outline of the tasks and steps you will undertake to fulfill this role:

1. **Decision Analysis**: You will analyze the driver's decision to assess whether they align with safe driving standards and best practices.

2. **Issue Identification**: If you detect a decision by the driver that may lead to suboptimal outcomes, you will pinpoint the problem, such as a lack of understanding of traffic rules, incorrect judgment of the surrounding environment, or delayed reaction times.

3. **Correction and Suggestions**: You will provide the correct reasoning process and decision outcomes, for instance, advising when and where to slow down, when to change lanes.

4. **Feedback and Recommendations**: You will offer immediate feedback and suggestions to the driver to help them improve their driving skills, such as reminders to be aware of blind spots, maintain safe distances, and avoid distractions while driving.

Through these steps, you will ensure that drivers can complete their driving tasks more safely and efficiently with your assistance. safely and efficiently with your assistance.
        """)
        human_message = textwrap.dedent(f"""
``` Human Message ```
{human_message}
``` Driver's Decision ```
{llm_response}
``` Evaluation Result ```
{evaluation}
The evaluation indicators were calculated as follows, all scores are between 0 and 1:
- Traffic Light Score: If you go through a red light, the score is 0.7. Otherwise it is 1.0. 
- Comfort Score: The greater the absolute value of car's acceleration and jerk, the smaller the comfort score.
- Efficiency Score: The lower the car's speed, the smaller the score.
- Speed Limit Score: If the car's speed exceeds the speed limit, the score will be less than 1.0. As the portion of the car that exceeds the speed limit gets larger, the score will be lower.
- Collision Score: When the likelihood of the car colliding with another car is higher, the score is lower. When the score is 1.0, the time in which the car is likely to collide with another car (ttc) is greater than 10 s. When the score is 0.0, the collision has happened.
- Decision Score: Traffic Light Score * (0.2 * Comfort Score + 0.2 * Efficiency Score + 0.2 * Speed Limit Score + 0.4 * Collision Score)

Now, you know that the driver receives a low score for making this decision, which means there are some mistake in driver's resoning and cause the wrong action.
Please carefully check every reasoning in Driver's Decision and find out the mistake in the reasoning process of driver, and also output your corrected version of Driver's Decision.
Your answer should use the following format:
{delimiter} Analysis of the mistake:
<Your analysis of the mistake in driver's reasoning process>
{delimiter} What should driver do to avoid such errors in the future:
<Your answer>
{delimiter} Corrected version of Driver's Decision:
<Your corrected version of Driver's Decision, must use the following format>
<reasoning>
<reasoning>
<repeat until you have a decision>
Response to user:{delimiter} <only output one `Action_id` as a int number of you decision, without any action name or explanation. The output decision must be unique and not ambiguous, for example if you decide to decelearate, then output `4`>
        """)
        # print(human_message)
        print("Reflection is running ...")

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message),
        ]
        print("input is ", messages)
        start_time = time.time()
        with get_openai_callback() as cb:
            response = self.llm(messages)
        print(f"Time taken: {time.time() - start_time:.2f}s")
        # print(cb)

        print("response:", response.content)
        decision_action = response.content.split(delimiter)[-1]
        try:
            result = int(decision_action)
            if result not in [1, 2, 3, 4, 8]:
                raise ValueError
        except ValueError:
            print("--------- Output is not available, checking the output... ---------")
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
                print(f"Checking question: {check_message}")
                print(f"Output checking result: {check_response.content}")
                # self.getLLMSource(cb, False)
            result = int(check_response.content.split(delimiter)[-1])
            response.content = response.content + f"\n{delimiter} Output checking result: {check_response.content}"
        # target_phrase = f"{delimiter} What should ChatGPT do to avoid such errors in the future:"
        # substring = response.content[response.content.find(
        #     target_phrase)+len(target_phrase):].strip()
        # corrected_memory = f"{delimiter} I have made a misake before and below is my self-reflection:\n{substring}"
        # print("Reflection done. Time taken: {:.2f}s".format(
        #     time.time() - start_time))
        # print("corrected_memory:", corrected_memory)

        return response.content

