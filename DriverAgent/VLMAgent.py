import base64
import json
import os
import re
import textwrap
from typing import Dict

import requests
import yaml
from openai import OpenAI
from rich import print

from trafficManager.common.vehicle import Behaviour


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def getText(filePath: str) -> str:
    with open(filePath, 'r') as f:
        res = f.read()
        return res

class PromptsWrap:
    def __init__(self, API_KEY: str = os.environ.get('OPENAI_API_KEY')) -> None:
        self.content = []
        self.api_key = API_KEY

    def addTextPrompt(self, textPrompt: str) -> Dict[str, str]:
        textPrompt = {
            "type": "text",
            "text": textPrompt
        }
        self.content.append(textPrompt)

    def addImagePrompt(self, imagePath: str):
        base64_image = encode_image(imagePath)
        imagePrompt = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "low"
            }
        }
        self.content.append(imagePrompt)

    def addImageBase64(self, image_base64: str):
        imagePrompt = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}",
                "detail": "low"
            }
        }
        self.content.append(imagePrompt)

    def request(self, max_tokens: int = 4000):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": self.content
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
    
class VLMAgent:
    def __init__(self, api_key: str) -> None:
        self.lastDecision: Behaviour = None
        self.api_key = api_key


    def str2behavior(self, decision) -> Behaviour:
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

    def generatePrompt(self, information: str, image_base64: str) -> PromptsWrap:
        prompts = PromptsWrap(self.api_key)
        prompts.addTextPrompt(getText('./DriverAgent/initialPrompts/Texts/SystemMessage.txt'))

        # add few-shots message
        # todo: memory retrive
        few_shot_num = 4
        for i in range(few_shot_num):
            prompts.addTextPrompt(
                getText(f'./DriverAgent/initialPrompts/Texts/Information_{i}.txt')
            )
            prompts.addImagePrompt(
                f'./DriverAgent/initialPrompts/Images/Fig_{i}.jpg'
            )
            prompts.addTextPrompt(
                getText(f'./DriverAgent/initialPrompts/Texts/Decision_{i}.txt')
            )

        prompts.addTextPrompt(textwrap.dedent(
            "The example ends here, please start describing, reasoning and making decision."
        ))
        prompts.addTextPrompt(information)
        prompts.addImageBase64(image_base64)
        
        return prompts

    def makeDecision(self, information: str, image_base64: str):
        prompts = self.generatePrompt(information, image_base64)
        response = prompts.request()
        print(response)
        ans = response['choices'][0]['message']['content']
        match = re.search(r'## Decision\n(.*)', ans)
        if match:
            decision = match.group(1)
            behavior = self.str2behavior(decision)
            self.lastDecision = behavior
            return response, behavior
        else:
            return response, None


if __name__ == '__main__':
    # OpenAI API Key
    # use system environment variable
    API_KEY = os.environ.get('OPENAI_API_KEY')

    client = OpenAI(api_key=API_KEY)

    testPrompts = PromptsWrap(API_KEY)
    testPrompts.addTextPrompt(getText('./DriverAgent/initialPrompts/Texts/SystemMessage.txt'))

    # add few-shots message
    for i in [0, 1, 2, 3]:
        testPrompts.addTextPrompt(
            getText(f'./DriverAgent/initialPrompts/Texts/Information_{i}.txt')
        )
        testPrompts.addImagePrompt(
            f'./DriverAgent/initialPrompts/Images/Fig_{i}.jpg'
        )
        testPrompts.addTextPrompt(
            getText(f'./DriverAgent/initialPrompts/Texts/Decision_{i}.txt')
        )

    testPrompts.addTextPrompt(textwrap.dedent(
        "The example ends here, please start describing, reasoning and making decision."
    ))
    testPrompts.addTextPrompt(getText('./DriverAgent/initialPrompts/Texts/Information_4.txt'))
    testPrompts.addImagePrompt(f'./DriverAgent/initialPrompts/Images/Fig_4.jpg')

    response = testPrompts.request()

    print(response)
