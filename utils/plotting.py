'''
Author: Licheng Wen
Date: 2022-11-11 16:13:21
Description: 
Copyright (c) 2022 by PJLab, All Rights Reserved. 
'''
from __future__ import annotations
from abc import ABC, abstractmethod
import logger
logging = logger.get_logger(__name__)

from math import inf
from matplotlib import pyplot as plt
import numpy as np
from typing import TYPE_CHECKING

from cubic_spline import Spline2D
from roadgraph import RoadGraph, TrafficLightStatus


def plot_roadgraph(roadgraph: RoadGraph) -> None:
    edges = roadgraph.edges
    lanes = roadgraph.lanes
    fig, ax = plt.subplots()
    for edge in edges.values():
        for lane_index in range(edge.lane_num):
            lane_id = edge.id + '_' + str(lane_index)
            lane = lanes[lane_id]

            lane.center_line, lane.left_bound, lane.right_bound = [], [], []
            s = np.linspace(0, lane.course_spline.s[-1], num=50)
            for si in s:
                lane.center_line.append(lane.course_spline.calc_position(si))
                lane.left_bound.append(
                    lane.course_spline.frenet_to_cartesian1D(
                        si, lane.width / 2)
                )
                lane.right_bound.append(
                    lane.course_spline.frenet_to_cartesian1D(
                        si, -lane.width / 2)
                )
            ax.plot(*zip(*lane.center_line), "w:", linewidth=1.5)
            plt.arrow(
                lane.center_line[0][0],
                lane.center_line[0][1],
                lane.center_line[2][0] - lane.center_line[0][0],
                lane.center_line[2][1] - lane.center_line[0][1],
                shape='full',
                width=0.3,
                length_includes_head=False,
                zorder=2,
                color="w",
            )
            if lane_index == edge.lane_num - 1:
                ax.plot(*zip(*lane.left_bound), "k", linewidth=1.5)
            else:
                ax.plot(*zip(*lane.left_bound), "k--", linewidth=1)
            if lane_index == 0:
                ax.plot(*zip(*lane.right_bound), "k", linewidth=1.5)

            if edge.from_junction == None:
                ax.plot(
                    [lane.left_bound[0][0], lane.right_bound[0][0]],
                    [lane.left_bound[0][1], lane.right_bound[0][1]],
                    "r",
                    linewidth=2,
                )
            if edge.to_junction == None:
                ax.plot(
                    [lane.left_bound[-1][0], lane.right_bound[-1][0]],
                    [lane.left_bound[-1][1], lane.right_bound[-1][1]],
                    "r",
                    linewidth=2,
                )

    for lane in lanes.values():
        for junctionlane, dir in lane.next_lanes.values():
            s = np.linspace(0, junctionlane.course_spline.s[-1], num=50)
            junctionlane.center_line = []
            for si in s:
                junctionlane.center_line.append(
                    junctionlane.course_spline.frenet_to_cartesian1D(si, 0)
                )
            if roadgraph.traffic_lights.lights[junctionlane] == TrafficLightStatus.GREEN:
                color = "green"
            elif roadgraph.traffic_lights.lights[junctionlane] == TrafficLightStatus.RED:
                color = "red"
            else:
                color = "yellow"

            ax.plot(*zip(*junctionlane.center_line),
                    ":", color=color, linewidth=1.5)

    ax.set_facecolor("lightgray")
    ax.grid(True)
    ax.axis("equal")
    plt.show()
