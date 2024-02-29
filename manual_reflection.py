import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sqlite3, os
import matplotlib.pyplot as plt
from typing import List, Tuple
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.text import Text
from simModel.Replay import ReplayModel
import io
from simInfo.Memory import DrivingMemory, MemoryItem

db_path = "experiments/zeroshot/gpt4v/exp_0.db"
replay = ReplayModel(db_path)
memory_agent = DrivingMemory()
# get data
conn = sqlite3.connect(db_path)
eval_df = pd.read_sql_query('''SELECT * FROM evaluationINFO''', conn)
qa_df = pd.read_sql_query('''SELECT * FROM QAINFO''', conn)
conn.close()

def getVehShape(
        posx: float, posy: float,
        heading: float, length: float, width: float
    ):
    radian = np.pi - heading
    rotation_matrix = np.array(
        [
            [np.cos(radian), -np.sin(radian)],
            [np.sin(radian), np.cos(radian)]
        ]
    )
    half_length = length / 2
    half_width = width / 2
    vertices = np.array(
        [
            [half_length, half_width], [half_length, -half_width],
            [-half_length, -half_width], [-half_length, half_width]
        ]
    )
    rotated_vertices = np.dot(vertices, rotation_matrix)
    position = np.array([posx, posy])
    translated_vertices = rotated_vertices + position
    return translated_vertices.tolist()

def plotSce(decisionFrame: int, replay:ReplayModel) -> str:
    fig, ax = plt.subplots()
    replay.timeStep = decisionFrame
    replay.getSce()
    roadgraphRenderData, VRDDict = replay.exportRenderData()
    if roadgraphRenderData and VRDDict:
        egoVRD = VRDDict['egoCar'][0]
        ex = egoVRD.x
        ey = egoVRD.y
        ego_shape = getVehShape(egoVRD.x, egoVRD.y, egoVRD.yaw, egoVRD.length, egoVRD.width)
        vehRectangle = Polygon(ego_shape, closed=True, facecolor="r", alpha=1)
        vehText = Text(ex, ey, 'ego', fontsize='x-small')
        ax.plot(list(egoVRD.trajectoryXQ)[:len(egoVRD.trajectoryXQ)//2], list(egoVRD.trajectoryYQ)[:len(egoVRD.trajectoryXQ)//2], 'r', linewidth=1, alpha=0.8)
        ax.add_patch(vehRectangle)
        ax.add_artist(vehText)
        for car in VRDDict["carInAoI"]:
            av_shape = getVehShape(car.x, car.y, car.yaw, car.length, car.width)
            vehRectangle = Polygon(av_shape, facecolor='#47C5FF', alpha=1)
            vehText = Text(car.x, car.y, car.id, fontsize='x-small')
            ax.plot(list(car.trajectoryXQ)[:len(car.trajectoryXQ)//2], list(car.trajectoryYQ)[:len(car.trajectoryXQ)//2], '#419CCD', linewidth=1, alpha=0.8)
            ax.add_patch(vehRectangle)
            ax.add_artist(vehText)
    replay.sr.plotScene(ax)

    ax.set_xlim(ex-60, ex+60)
    ax.set_ylim(ey-50, ey+50)

    plt.axis('off')
    buffer = io.BytesIO()
    plt.savefig(buffer,format = 'png')
    plt.close()

    image = replay.exportImageData()
    return buffer, image

def on_slider_change():
    st.session_state.x_position = st.session_state.slider_key
    get_memory()

def on_button():
    st.session_state.current_mem = memory_agent.getReflection(st.session_state.current_mem, False)

def get_memory():
    st.session_state.current_mem.reflection = ""
    st.session_state.current_mem.set_description(qa_df.loc[st.session_state.x_position])
    st.session_state.current_mem.set_score(eval_df.loc[st.session_state.x_position])

def last_frame():
    st.session_state.x_position = st.session_state.x_position - 1 if st.session_state.x_position > 1 else eval_df.shape[0]-1
    get_memory()

def next_frame():
    st.session_state.x_position = st.session_state.x_position + 1 if st.session_state.x_position < eval_df.shape[0]-1 else 0
    get_memory()

if __name__ == "__main__":
    st.set_page_config(
    page_title="Result Analysis",
    page_icon=":memo:",
    layout="wide",
    initial_sidebar_state="expanded")

    st.title(":memo: Result Analysis")
    # 1. choose the frame
    if "x_position" not in st.session_state:
        st.session_state.x_position = 0
    
    if "current_mem" not in st.session_state:
        st.session_state.current_mem = MemoryItem()

    with st.sidebar:
        st.title("Choose the Frame")
        st.markdown("Use the slider to select the frame you want to analyze.")
        # 2. Create a Plotly figure with scatter plot
        # 2.1 function button
        frame_col1, frame_col2 = st.columns(2)

        with frame_col1:
            frame_button1 = st.button("Last Frame", key="last_frame", on_click=last_frame)

        with frame_col2:
            frame_button2 = st.button("Next Frame", key="next_frame", on_click=next_frame)
        
        # 2.2 Slider for controlling the x-axis position
        slider_key = st.slider("#### Select Frame", min_value=0, max_value=eval_df.shape[0]-1, value=st.session_state.x_position, key="slider_key", on_change=on_slider_change)

        # 2.3 Create a Plotly figure with scatter plot
        fig = go.Figure()
        scatter = go.Scatter(
            x=eval_df['frame'] - 10,
            y=eval_df['decision_score'],
            mode='markers',
            marker=dict(size=8, color='#FFB6C1')
        )

        # Highlight the selected point with red color
        scatter.marker.line.width = 3
        scatter.marker.line.color = 'red'
        scatter.selectedpoints = [slider_key]

        line = go.Line(
            x=eval_df['frame'] - 10,
            y=eval_df['decision_score']
        )
        # Add trace to figure
        fig.add_trace(scatter)
        fig.add_trace(line)
        fig.update_traces(showlegend=False)

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

        # 2.4 Display information about the selected point
        st.write(f"""#### **Selected frame is {int(eval_df['frame'][slider_key])}, the score is {round(eval_df['decision_score'][slider_key], 2)}. The detail score is as follows:**\n
        - comfort score: {round(eval_df['comfort_score'][slider_key], 2)}\n
        - safety score: {round(eval_df['collision_score'][slider_key], 2)}\n
        - efficiency score: {round(eval_df['efficiency_score'][slider_key], 2)}\n""".replace("   ", ""))

        if eval_df["caution"][slider_key] != "":
            caution = ""
            for line in eval_df['caution'][slider_key].split("\n"):
                if line != "":
                    caution += f"- {line}\n"

            st.write(f"#### **:red[There are a few things to note in this frame:]**\n{caution}")

    text_col, image_col = st.columns(2)

    # 3. QA pairs
    with text_col:
        question = f"## Driving scenario description:\n" + qa_df["description"][slider_key] + "\n## Navigation instruction\n" + qa_df["navigation"][slider_key] + "\n## Available actions:\n" + qa_df["actions"][slider_key]
        
        st.write(f"#### Current Description")

        st.text_area(value=question, label="## Current Description", height=600, label_visibility="collapsed")
        st.write(f"#### Reasoning from LLM")
        st.text_area(value=qa_df["response"][slider_key], label="## Reasoning from LLM", height=350, label_visibility="collapsed")
    
    # 4. image
    with image_col:
        buffer, image = plotSce(int(qa_df['frame'][slider_key]), replay)
        st.write("### **BEV**")
        st.image(buffer, use_column_width=True)
        buffer.close()
        if image:
            st.write("#### Camera")
            left_col, front_col, image_col = st.columns(3)
            with left_col:
                st.image(image.ORI_CAM_FRONT_LEFT, use_column_width=True)
                st.image(image.ORI_CAM_BACK_LEFT, use_column_width=True)
            with front_col:
                st.image(image.ORI_CAM_FRONT, use_column_width=True)
                st.image(image.ORI_CAM_BACK, use_column_width=True)
            with image_col:
                st.image(image.ORI_CAM_FRONT_RIGHT, use_column_width=True)
                st.image(image.ORI_CAM_BACK_RIGHT, use_column_width=True)

    # 5. memory moudle & function button
    col0, _, _, _, col1, col2 = st.columns(6)
    with col0:
        st.write("#### Reflection: ")
    with col1:
        button1 = st.button("Reflection", key="reflection", on_click=on_button)

    with col2:
        button2 = st.button("Add to Memory", key="add_to_memory")

    # reflection
    if st.session_state.current_mem.reflection == "":
        default_value = ""
    else:
        default_value = f"#### The correct result is:\n {st.session_state.current_mem.response} \n#### The suggestion is:\n {st.session_state.current_mem.reflection}"
    user_input = st.text_area("#### Reflection: ", value=default_value, height=150, label_visibility='collapsed')

    # add to memory
    if button2:
        st.session_state.current_mem.set_reflection(user_input, "", user_input.split("####")[1])
        memory_agent.addMemory(st.session_state.current_mem)

