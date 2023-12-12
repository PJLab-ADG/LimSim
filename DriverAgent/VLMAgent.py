import os
import re
import json
import yaml
import base64
import requests
import textwrap
from rich import print
from openai import OpenAI
from typing import Dict, List, Union

from DriverAgent.Informer import Informer

from trafficManager.common.vehicle import Behaviour


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def getText(filePath: str) -> str:
    with open(filePath, 'r') as f:
        res = f.read()
        return res

class PrompsWrap:
    def __init__(self, API_KEY: str) -> None:
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
    def __init__(self) -> None:
        self.lastDecision: Behaviour = None

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

    def makeDecision(self, prompts: PrompsWrap) -> Behaviour:
        response = prompts.request()
        ans = response['choices']['message']['content']
        match = re.search(r'## Decision\n(.*)', ans)
        if match:
            decision = match.group(1)
            behavior = self.str2behavior(decision)
            self.lastDecision = behavior
            return behavior


if __name__ == '__main__':
    # OpenAI API Key
    CONFIG = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
    API_KEY = CONFIG['API_KEY']

    client = OpenAI(api_key=API_KEY)

    testPrompts = PrompsWrap(API_KEY)
    testPrompts.addTextPrompt(getText('./initialPrompts/Texts/systemMessage.txt'))

    # add few-shots message
    for i in [0, 1, 2, 3]:
        testPrompts.addTextPrompt(
            getText(f'./initialPrompts/Texts/Information_{i}.txt')
        )
        testPrompts.addImagePrompt(
            f'./initialPrompts/Images/Fig_{i}.jpg'
        )
        testPrompts.addTextPrompt(
            getText(f'./initialPrompts/Texts/Decision_{i}.txt')
        )

    testPrompts.addTextPrompt(textwrap.dedent(
        "The example ends here, please start describing, reasoning and making decision."
    ))
    testPrompts.addTextPrompt(getText('./initialPrompts/Texts/Information_4.txt'))
    testPrompts.addImagePrompt(f'./initialPrompts/Images/Fig_4.jpg')

    response = testPrompts.request()

    print(response)
