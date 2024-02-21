# LimSim++

[![Custom badge](https://img.shields.io/badge/paper-Arxiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2402.01246)
[![Custom badge](https://img.shields.io/badge/Project-page-green?logo=document)](https://pjlab-adg.github.io/limsim_plus/)


LimSim++: an extended version of LimSim designed for the application of Multimodal Large Language Models ((M)LLMs) in autonomous driving. 

The following video shows the process of autonomous driving with VLM.  The top of the GUI is the image information in carla, the left is the information description of the current scene, and the right is the reasoning process of LLM. 

<video src="https://pjlab-adg.github.io/limsim_plus/static/videos/zeroshot_gpt4v.mp4" controls="controls" width="500" height="300"></video>


To run this project in minutes, check [Getting Started](#Getting-Started).

## Set Up

- This project uses [conda](https://github.com/conda/conda) to manage the environment
- [SUMO](https://www.eclipse.org/sumo/) >= 1.15.0 
- [Carla](https://github.com/carla-simulator/carla) == 9.15.0 (?)

After configuring the runtime environment, download the LimSim++ source code to your local machine:

```powershell
git clone https://github.com/Fdarco/LimSimLLM.git
```

Finally, you need to create environments using the ``environment.yml``:

```powershell
cd LimSimLLM
conda env create -f environment.yml
```

Now, the local installation and deployment of LimSim++ is complete.

## Getting Started

### 1. Using (M)LLMs in autonomous driving 🚙

- #### LimSim++ supports Large-Language Models, such as GPT-3.5, GPT-4, etc. 
To experience it, run the following command:

```bash
# use openai
export OPENAI_API_TYPE="openai"
export OPENAI_API_VERSION="2020-11-07"
export OPENAI_API_KEY='your openai key'
# use azure
export OPENAI_API_TYPE="azure"
export OPENAI_API_KEY='your azure key'
export OPENAI_API_BASE="your azure node"
export OPENAI_API_VERSION="2023-07-01-preview"
export EMBEDDING_MODEL="ada-002"
 
python LLMAgent_closeloop.py 
```

- #### LimSim++ supports Vision-Language Models, such as GPT-4V. 
To experience it, You should open two terminal.

```bash
# Terminal 1
cd path-to-carla/
./CarlaUE4.sh
```

Now you should see CARLA GUI. Then switch to another terminal:

```bash
# Termnial 2
cd path-to-carla/
cd PythonAPI/util/
python3 config.py --map Town06
```

Now the map in Carla is successfully switch to Town06 map.

```bash
# Termnial 2
export OPENAI_API_KEY='your openai key'
cd path-to-LimSimLLM/
python VLMAgentCloseLoop.py
```


### 2. Simulation replay 🎥
In the root directory, running the following command will demonstrate the (M)LLMs decision-making process :

```powershell
python ReplayExample.py
```

### 3. Decisions evaluate 📝
After the (M)LLMs' autonomous driving task is completed, running the following code will evaluate the (M)LLMs' decision results :

```bash
python Evaluator.py
```

Then you can see the scores for each frame of the decision in the database, and the evaluation result of the whole route in the file ``llm_decision_result.log``.

### 4. Reflection & Memory 🧐

- #### Auto-add memory
The following command allows the LLM to self-reflect on this autonomous driving task and automatically add items to the memory library:

```bash
python simInfo/Memory.py
```

The memory database will be created in ``db/decision_mem``

- #### Maunal-add memory

TODO


**Use reflection module:**

To activate the memory module, set ``use_memory`` to True in ``LLMAgent_closeloop.py``. The default setting uses 3-shots . You can modify this in the code.

## License and Citation
All assets and code in this repository are under the Apache 2.0 license. If you use LimSim++ in your research , please use the following BibTeX entry.
```
@article{fu2024limsim++,
  title={LimSim++: A Closed-Loop Platform for Deploying Multimodal LLMs in Autonomous Driving},
  author={Fu, Daocheng and Lei, Wenjie and Wen, Licheng and Cai, Pinlong and Mao, Song and Dou, Min and Shi, Botian and Qiao, Yu},
  journal={arXiv preprint arXiv:2402.01246},
  year={2024}
}
```

