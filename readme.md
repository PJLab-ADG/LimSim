# LimSim Light: A lightweight LimSim simulator that parses OpenDrive standard format files

Although LimSim, leveraging SUMO as a simulation engine, facilitates joint simulation with CARLA, we have developed LimSimLight to address users' needs focusing on traffic flow simulation and trajectory planning. This new version targets local road networks rather than the urban-scale networks SUMO supports. LimSimLight is a lightweight alternative, driven by a custom simulation engine, abandoning complex cross-platform bridging and intricate road structures designed for specific trajectory planning algorithms. It directly parses OpenDrive-formatted road network files, offering comprehensive topological and geometric information for road networks. It supports trajectory planning for any vehicle on various road types. This version is better suited for scenario simulation in research on controllable traffic flow generation and trajectory planning methods based on deep learning.


## Features

- **OpenDrive Parser**: Analyzing the road network files in the OpenDrive standard format to obtain basic road information and network topology.
- **Simulation Engine**: Customizing a parallel processing simulation engine with a visual GUI, supporting real-time simulation and record-based replay.
- **Baseline Traffic Flow**: Providing baseline traffic flow simulation models for route planning, following, and lane changing, and also supports users in replacing any module.


## Installation

Download the LimSim source code to your local machine:

```powershell
git clone https://github.com/PJLab-ADG/LimSim.git
```

Finally, you need to install the required Python extensions:

```powershell
cd LimSim
git checkout LimSimLight
pip install -r requirements.txt
```

Now, the local installation and deployment of LimSim-Light are complete.


## Start Simulation

- Run the real-time simulation:
```powershell
python ModelExample.py
```

- Run the relayed simulation (Make sure ``simulationDatabase.db`` exists):

```powershell
python ModelExample.py replay
```

- Choose another OpenDrive file in ``ModelExample.py`` (optional): 

```python
net_file = file_paths["CarlaTown02"]
```
OpenDrive files are available in ``networkFiles/CarlaTown`` got from [CARLA](https://github.com/carla-simulator/carla).

- Modify the traffic demands (optional):
```powershell
vi demands.txt
```

## Module Introduction

### 1. Road Network Construction  (``simModel/networkbuild.py``)

From OpenDrive standard files, this module can extract the fundamental elements of a road network: roads and junctions. To ascertain the boundaries and centerlines of the roads, it is necessary to generate them based on the geometric curve functions of the roads. The network topology encompasses the connectivity of roads and lanes, which may have varying requirements for different trajectory planning algorithms. Specifically, to construct a fully connected form between the entry and exit lanes of junctions, additional internal connecting roads within junctions have been introduced. The transformation between Cartesian and Frenet coordinate systems is crucial for trajectory planning. Furthermore, this module also defines the traffic control signals for each junction.

### 2. Simulation Engine (``simModel/model.py``)

Upon initiation of the simulation engine, it first generates traffic flows between various origins and destinations based on a traffic demand file. The route planning among these traffic flows is determined by baseline methods, or can be customized by the user. Vehicles that arrive continuously are generated according to a Poisson distribution, based on the generated traffic flows. At each simulation step, the status of traffic lights and vehicles is synchronized and updated. The simulation engine can operate in a vehicle tracking mode or a fixed perspective mode, similar to the definitions in previous versions of LimSim. Especially, LimSimLight supports flexible mouse-switching modes and the ability to change the ego vehicle， allowing for the flexible selection of the area of interest. The dynamic traffic flow and the status of traffic lights are recorded in real-time to a peewee database, facilitating the replay of the simulation.

### 3. Baseline traffic flow (``trafficManager/planning.py``)

Implement the shortest path algorithm based on Dijkstra, considering the topological connection of roads or lanes. Trajectory planning includes using the Intelligent Driver Model (IDM) as a car-following algorithm, combined with custom lane-changing logic to plan lane-changing trajectories. The baseline trajectory planning is universal for roads and junctions, but it needs to consider the status of traffic signals to determine whether the junction can be passed. The planned trajectory is first completed in the Frenet coordinate system and then transformed into Cartesian coordinate system trajectory points. Collision detection for different vehicles follows a set of rule-based conditional logic.

### 4. Driving Performance Evaluation (``trafficManager/evaluation.py``)

The evaluation of driving performance encompasses multiple dimensions, such as collision detection, drivable area detection, time to collision (TTC), comfort, and driving efficiency, which are collectively considered through a weighted sum approach. This module evaluates both past and planned trajectories. Collision detection determines whether the geometric contours of two vehicles intersect. Drivable area detection ascertains if the vehicle's contour falls within the geometric area of the road boundaries. Time to collision assesses the potential collision time by calculating the time difference it takes for vehicles to reach the intersection point between their respective travel trajectories, based on their current speed and direction. Comfort considers whether acceleration, jerk, steering angular velocity, and angular acceleration are within a reasonable range. Driving efficiency simply calculates the ratio of the current speed to the vehicle's expected speed.

### 5. Visual GUI and Interaction (``visualization/plotEngine.py``)

The simulation diagram is presented using the dearpygui module, which includes the road network, dynamic scenes, ego vehicle basic information, ego vehicle speed and acceleration curves, and ego vehicle driving performance evaluation. The GUI runs in a parallel thread, enabling high-frequency refreshing and supporting mouse signal input: scrolling (scene zoom), dragging (adjusting the scene view center), and double-clicking (within the scene—change the area of interest, other areas—pause and play). The scene display content includes road topology, planned trajectories of the ego vehicle and other vehicles, traffic signals, etc.


## License and Citation

All assets and code in this repository are under the Apache 2.0 license. If you use LimSimLight in your research , please use the following BibTeX entry.

```
@inproceedings{wenl2023limsim,
  title={LimSim: A long-term interactive multi-scenario traffic simulator},
  author={Wenl, Licheng and Fu, Daocheng and Mao, Song and Cai, Pinlong and Dou, Min and Li, Yikang and Qiao, Yu},
  booktitle={26th International Conference on Intelligent Transportation Systems (ITSC)},
  pages={1255--1262},
  year={2023}
}
 
@article{fu2024limsim++,
  title={LimSim++: A Closed-Loop Platform for Deploying Multimodal LLMs in Autonomous Driving},
  author={Fu, Daocheng and Lei, Wenjie and Wen, Licheng and Cai, Pinlong and Mao, Song and Dou, Min and Shi, Botian and Qiao, Yu},
  journal={arXiv preprint arXiv:2402.01246},
  year={2024}
}
```
