"""This file contains the drawing functions and front-end presentation functions to build a web page to view decision scores, decision processes, bevs, and ringviews. Besides, the web page can get the user's reflection for a selected frame, and add it to the memory database.
"""

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
import io, base64
from simInfo.Memory import DrivingMemory, MemoryItem
import textwrap
import atexit, time, copy

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

def plotSce(decisionFrame: int) -> str:
    fig, ax = plt.subplots()
    st.session_state.replay.timeStep = decisionFrame
    st.session_state.replay.getSce()
    roadgraphRenderData, VRDDict = st.session_state.replay.exportRenderData()
    # plot roadgraph
    st.session_state.replay.sr.plotScene(ax)
    # plot car
    if roadgraphRenderData and VRDDict:
        egoVRD = VRDDict['egoCar'][0]
        ex = egoVRD.x
        ey = egoVRD.y
        ego_shape = getVehShape(egoVRD.x, egoVRD.y, egoVRD.yaw, egoVRD.length, egoVRD.width)
        vehRectangle = Polygon(ego_shape, closed=True, facecolor="#D35400", alpha=1)
        vehText = Text(ex, ey, 'ego', fontsize='x-small')
        ax.plot(list(egoVRD.trajectoryXQ)[:len(egoVRD.trajectoryXQ)//2], list(egoVRD.trajectoryYQ)[:len(egoVRD.trajectoryXQ)//2], '#CD84F1', linewidth=1.5, alpha=1)
        ax.add_patch(vehRectangle)
        ax.add_artist(vehText)
        for car in VRDDict["carInAoI"]:
            av_shape = getVehShape(car.x, car.y, car.yaw, car.length, car.width)
            vehRectangle = Polygon(av_shape, facecolor='#2980B9', alpha=1)
            vehText = Text(car.x, car.y, car.id, fontsize='x-small')
            if car.trajectoryXQ != None:
                ax.plot(list(car.trajectoryXQ)[:len(car.trajectoryXQ)//2], list(car.trajectoryYQ)[:len(car.trajectoryXQ)//2], '#CD84F1', linewidth=1.5, alpha=1)
            ax.add_patch(vehRectangle)
            ax.add_artist(vehText)

    ax.set_xlim(ex-50, ex+50)
    ax.set_ylim(ey-30, ey+30)

    plt.axis('off')
    ax.set_aspect('equal', adjustable='box')
    buffer = io.BytesIO()
    plt.savefig(buffer,format = 'png', dpi=600, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    image = st.session_state.replay.exportImageData()
    return buffer, image

def on_slider_change():
    st.session_state.x_position = st.session_state.slider_key
    get_memory()

def on_button():
    memroy = MemoryItem()
    memroy.set_description(st.session_state.qa_df.loc[st.session_state.x_position])
    memroy.set_score(st.session_state.eval_df.loc[st.session_state.x_position])
    memroy = st.session_state.memory_agent.getReflection(memroy, False)
    st.session_state.current_mem = copy.deepcopy(memroy)

def get_memory():
    st.session_state.current_mem.reflection = ""
    st.session_state.current_mem.set_description(st.session_state.qa_df.loc[st.session_state.x_position])
    st.session_state.current_mem.set_score(st.session_state.eval_df.loc[st.session_state.x_position])

def last_frame():
    st.session_state.x_position = st.session_state.x_position - 1 if st.session_state.x_position > 0 else st.session_state.eval_df.shape[0]-1
    get_memory()

def next_frame():
    st.session_state.x_position = st.session_state.x_position + 1 if st.session_state.x_position < st.session_state.eval_df.shape[0]-1 else 0
    get_memory()

def add_memory():
    try:
        result = st.session_state.user_input.split("#### Corrected version of Driver's Decision:")[1].split("#### What should driver do to avoid such errors in the future:")[0]
        comment = st.session_state.user_input.split("#### What should driver do to avoid such errors in the future:")[1]
        action = int(result.split("####")[-1])
    except:
        st.error("Please make sure your reflection is formatted correctly.")
        return
    st.session_state.current_mem.set_reflection(result, comment, action)
    try:
        st.session_state.memory_agent.addMemory(None)
        st.session_state.memory_agent.scenario_memory.persist()
        st.success("Reflection added successfully.")
    except Exception as e:
        st.error(f"Reflection added failed, the reason is: {e}")
    
def cleanup():
    print("stopping")

if __name__ == "__main__":
    atexit.register(cleanup)

    st.set_page_config(
        page_title="Result Analysis",
        page_icon=":memo:",
        layout="wide",
        initial_sidebar_state="expanded")

    st.title(":memo: Result Analysis")
    # 0. init the variant
    if "replay" not in st.session_state:
        db_path = "results/2024-03-04_17-46-55.db" # the database need to analysis
        # get data
        conn = sqlite3.connect(db_path)
        st.session_state.eval_df = pd.read_sql_query('''SELECT * FROM evaluationINFO''', conn)
        st.session_state.qa_df = pd.read_sql_query('''SELECT * FROM QAINFO''', conn)
        conn.close()
        st.session_state.replay = ReplayModel(db_path)

    if "memory_agent" not in st.session_state:
        memory_path = os.path.dirname(os.path.abspath(__file__)) + "/db/" + "memory_library/" # the database for memory
        st.session_state.memory_agent = DrivingMemory(memory_path)
    
    if "x_position" not in st.session_state:
        st.session_state.x_position = 0
    
    if "current_mem" not in st.session_state:
        st.session_state.current_mem = MemoryItem()
        get_memory()

    # 1. choose the frame
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
        slider_key = st.slider("#### Select Frame", min_value=0, max_value=st.session_state.eval_df.shape[0]-1, value=st.session_state.x_position, key="slider_key", on_change=on_slider_change)

        # 2.3 Create a Plotly figure with scatter plot
        fig = go.Figure()
        scatter = go.Scatter(
            x=st.session_state.eval_df['frame'] - 10,
            y=st.session_state.eval_df['decision_score'],
            mode='markers',
            marker=dict(size=8, color='#FFB6C1')
        )

        # Highlight the selected point with red color
        scatter.marker.line.width = 3
        scatter.marker.line.color = 'red'
        scatter.selectedpoints = [slider_key]

        line = go.Line(
            x=st.session_state.eval_df['frame'] - 10,
            y=st.session_state.eval_df['decision_score']
        )
        # Add trace to figure
        fig.add_trace(scatter)
        fig.add_trace(line)
        fig.update_traces(showlegend=False)

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

        # 2.4 Display information about the selected point
        st.write(f"""#### **Selected frame is {int(st.session_state.eval_df['frame'][slider_key])}, the score is {round(st.session_state.eval_df['decision_score'][slider_key], 2)}. The detail score is as follows:**\n
        - comfort score: {round(st.session_state.eval_df['comfort_score'][slider_key], 2)}\n
        - safety score: {round(st.session_state.eval_df['collision_score'][slider_key], 2)}\n
        - efficiency score: {round(st.session_state.eval_df['efficiency_score'][slider_key], 2)}\n""".replace("   ", ""))

        if st.session_state.eval_df["caution"][slider_key] != "":
            caution = ""
            for line in st.session_state.eval_df['caution'][slider_key].split("\n"):
                if line != "":
                    caution += f"- {line}\n"

            st.write(f"#### **:red[There are a few things to note in this frame:]**\n{caution}")

    text_col, image_col = st.columns(2)

    # 3. QA pairs
    with text_col:
        question = f"## Driving scenario description:\n" + st.session_state.qa_df["description"][slider_key] + "\n## Navigation instruction\n" + st.session_state.qa_df["navigation"][slider_key] + "\n## Available actions:\n" + st.session_state.qa_df["actions"][slider_key]
        
        st.write(f"#### Current Description")

        st.text_area(value=question, label="## Current Description", height=400, label_visibility="collapsed")
        st.write(f"#### Reasoning from LLM")
        st.text_area(value=st.session_state.qa_df["response"][slider_key], label="## Reasoning from LLM", height=350, label_visibility="collapsed")
    
    # 4. image
    with image_col:
        buffer, image = plotSce(int(st.session_state.qa_df['frame'][slider_key]))
        st.write("### **BEV**")
        # show the image
        image_data = buffer.getvalue() 
        encoded_image = base64.b64encode(image_data).decode() 

        custom_css = """
            <style> 
                .image-container { 
                    width: auto;
                    text-align: center;
                    justify-content: center;
                    align-items: center; } 
                .image-with-border { 
                    border: 1px solid #C0C0C0;
                    padding: 5px;
                    max-width: auto;
                    max-height: 390px; 
                    object-fit: contain;
                } 
            </style> 
        """

        st.markdown(custom_css, unsafe_allow_html=True) 
        st.markdown( f""" <div class="image-container"> <img src="data:image/png;base64,{encoded_image}" class="image-with-border"> </div> """, unsafe_allow_html=True )
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
    st.write("#### Reflection: ")
    
    # reflection
    if st.session_state.current_mem.reflection == "":
        default_value = ""
    else:
        default_value = st.session_state.current_mem.response + st.session_state.current_mem.reflection
    user_input = st.text_area("#### Reflection: ", value=default_value, height=150, label_visibility='collapsed', key="user_input")

    col1, col2 = st.columns([1,7])
    with col1:
        button1 = st.button("Reflection", key="reflection", on_click=on_button)

    with col2:
        button2 = st.button("Add to Memory", key="add_to_memory", on_click=add_memory)

    st.text_area(label="##### Make sure your reflection is formatted as follows:", value=textwrap.dedent(f'''\
        #### Corrected version of Driver's Decision:
        {{Results of the reflection}}
        Response to user:#### {{action_id}}
        #### What should driver do to avoid such errors in the future:
        {{Suggestions for this scenario}}
        '''), height=150)
