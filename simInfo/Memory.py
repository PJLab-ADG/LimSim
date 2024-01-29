import os
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import sqlite3
import pandas as pd
from simInfo.Reflection import ReflectionAssistant
from typing import List, Tuple

class MemoryItem():
    def __init__(self, score = 0, detail_score = {}, description = "", navigation = "", response = "", action = 0, cautious = "") -> None:
        self.score = score
        self.detail_score = detail_score
        self.description = description
        self.navigation = navigation
        self.response = response
        self.action = action
        self.cautious = cautious
        self.available_action = ""
        self.reflection = ""

    def set_score(self, score_df):
        self.score = score_df["decision_score"]
        self.detail_score = {
            "traffic_light_score": score_df["traffic_light_score"],
            "comfort_score": score_df["comfort_score"],
            "efficiency_score": score_df["efficiency_score"],
            "speed_limit_score": score_df["speed_limit_score"],
            "safety_score": score_df["collision_score"]
        }
        self.cautious = score_df["caution"]

    def set_description(self, QA_df):
        self.description = QA_df["description"]
        self.navigation = QA_df["navigation"]
        self.available_action = QA_df["actions"]
        self.response = QA_df["response"]
        self.action = QA_df["choose_action"]

    def set_reflection(self, result, comment, action):
        self.reflection = comment
        self.response = result
        self.action = action


class DrivingMemory:
    def __init__(self, db_path=None) -> None:
        if os.environ["OPENAI_API_TYPE"] == 'azure':
                self.embedding = OpenAIEmbeddings(
                    deployment=os.environ['EMBEDDING_MODEL'], chunk_size=1)
        elif os.environ["OPENAI_API_TYPE"] == 'openai':
            self.embedding = OpenAIEmbeddings()
        else:
            raise ValueError(
                "Unknown OPENAI_API_TYPE: should be azure or openai")
        
        db_path = os.path.join(
            './db', 'decision_mem/') if db_path is None else db_path
        self.scenario_memory = Chroma(
            embedding_function=self.embedding,
            persist_directory=db_path
        )

        print("==========Loaded Memory, Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.==========")

        self.reflector = ReflectionAssistant()


    def retriveMemory(self, query_scenario, top_k: int = 5):
        similarity_results = self.scenario_memory.similarity_search_with_score(
            query_scenario, k=top_k)
        fewshot_results = []
        for idx in range(0, len(similarity_results)):
            # print(f"similarity score: {similarity_results[idx][1]}")
            fewshot_results.append(similarity_results[idx][0].metadata)
        return fewshot_results

    def addMemory(self, memory: MemoryItem):
        key = "## Driving scenario description:\n" + memory.description + "\n## Navigation instruction:\n" + memory.navigation
        key = key.replace("'", "")
        human_question = "## Driving scenario description:\n" + memory.description + "\n## Navigation instruction:\n" + memory.navigation + "\n## Available actions:\n" + memory.available_action
        # https://docs.trychroma.com/usage-guide#using-where-filters
        get_results = self.scenario_memory._collection.get(
            where_document={
                "$contains": key
            }
        )

        if len(get_results['ids']) > 0:
            # already have one
            id = get_results['ids'][0]
            self.scenario_memory._collection.update(
                ids=id, metadatas={
                    "human_question": human_question,
                    'LLM_response': memory.response, 'action': str(memory.action),
                    'comments': memory.reflection
                }
            )
            print("Modify a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
        else:
            doc = Document(
                page_content=key,
                metadata={
                    "human_question": human_question.replace("'", ""),
                    'LLM_response': memory.response.replace("'", ""), 'action': str(memory.action),
                    'comments': memory.reflection.replace("'", "")
                }
            )
            id = self.scenario_memory.add_documents([doc])
            print("Add a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    def deleteMemory(self, ids):
        self.scenario_memory._collection.delete(ids=ids)
        print("Delete", len(ids), "memory items. Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")

    def combineMemory(self, other_memory):
        other_documents = other_memory.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        current_documents = self.scenario_memory._collection.get(
            include=['documents', 'metadatas', 'embeddings'])
        for i in range(0, len(other_documents['embeddings'])):
            if other_documents['embeddings'][i] in current_documents['embeddings']:
                print("Already have one memory item, skip.")
            else:
                self.scenario_memory._collection.add(
                    embeddings=other_documents['embeddings'][i],
                    metadatas=other_documents['metadatas'][i],
                    documents=other_documents['documents'][i],
                    ids=other_documents['ids'][i]
                )
        print("Merge complete. Now the database has ", len(
            self.scenario_memory._collection.get(
                include=['embeddings'])['embeddings']), " items.")

    # def checkResponse(self, memory: MemoryItem) -> tuple[bool, str]:
    #     pass
    
    def getReflection(self, memory: MemoryItem) -> MemoryItem:
        delimiter = "####"
        human_message = "## Driving scenario description:\n" + memory.description + "\n## Navigation instruction:\n" + memory.navigation + "\n## Available actions:\n" + memory.available_action
        detail_score = ""
        for key, value in memory.detail_score.items():
            detail_score += f"{key}: {value}, "

        evaluation = f"The current score is {memory.score}, the detail score is {detail_score.strip(', ')}."
        caution = "There are the current decision's cautions:\n{memory.cautious}"
        
        LLM_response, action = self.reflector.reflection(human_message, memory.response, evaluation, memory.action, caution)
        
        if LLM_response == None:
            return None
        
        decision_phrase = f"{delimiter} Corrected version of Driver's Decision:"
        correct_answer = LLM_response.partition(decision_phrase)[2].strip()

        comment_phrase = f"{delimiter} What should driver do to avoid such errors in the future:"
        comment = LLM_response.partition(comment_phrase)[2].partition(decision_phrase)[0].strip()
        # substring = LLM_response[LLM_response.find(
            # target_phrase)+len(target_phrase):].strip()
        # comment = f"{delimiter} I have made a misake before and below is my self-reflection:\n{comment_ori}"
        # action = int(LLM_response.split(delimiter)[-1].strip(" "))
        memory.set_reflection(correct_answer, comment, action)
        return memory
    
    def divideBasedOnScore(self, database: str) -> Tuple[List[MemoryItem], List[MemoryItem]]:
        """_summary_

        Args:
            database (str): _description_

        Returns:
            Tuple[List[MemoryItem], List[MemoryItem]]: _description_
        """
        good_memory = []
        bad_memory = []

        conn = sqlite3.connect(database)
        eval_df = pd.read_sql_query('''SELECT * FROM evaluationINFO''', conn)
        QA_df = pd.read_sql_query('''SELECT * FROM QAINFO''', conn)

        result_df = pd.read_sql_query('''SELECT * FROM resultINFO''', conn)
        
        if result_df["result"][0]:
            # eval_df.sort_values(by=['decision_score'], inplace=True)
            # eval_df.reset_index(drop=True, inplace=True)
            # 找到中位数
            middle_score = eval_df["decision_score"].quantile(q = 0.2, interpolation="linear")

            # 找到高/低于中位数的数据
            low_index = eval_df[eval_df["decision_score"] <= middle_score].index.tolist()
            high_index = eval_df[eval_df["decision_score"] > middle_score].index.tolist()
            # 对index基于不连续判断进行分段
            low_split = self.split(low_index)
            high_split = self.split(high_index)
            bad_mem_index = []
            good_mem_index = []
            # 对每一段进行判断
            for i in low_split:
                item = eval_df.loc[i].copy(True)
                item.sort_values(by=['decision_score'], ascending=True, inplace=True)
                bad_index = item.index.tolist()[0]
                if bad_index > 0:
                    bad_mem_index.append(bad_index-1)
                if bad_index < len(eval_df)-1:
                    bad_mem_index.append(bad_index+1)
                bad_mem_index.append(bad_index)
            for i in high_split:
                item = eval_df.loc[i].copy(True)
                item.sort_values(by=['decision_score'], ascending=False, inplace=True)
                good_mem_index.append(item.index.tolist()[0])

            # 筛选出前20%的数据
            good_eval_df = eval_df.loc[good_mem_index]
            bad_eval_df = eval_df.loc[bad_mem_index]

            # 找出对应的QAINFO
            good_QA_df = QA_df.loc[good_mem_index]
            bad_QA_df = QA_df.loc[bad_mem_index]

            # 整理成list
            for i in good_eval_df.index.to_list():
                memory = MemoryItem()
                memory.set_description(good_QA_df.loc[i])
                memory.set_score(good_eval_df.loc[i])
                good_memory.append(memory)
            for i in bad_eval_df.index.tolist():
                memory = MemoryItem()
                memory.set_description(bad_QA_df.loc[i])
                memory.set_score(bad_eval_df.loc[i])
                bad_memory.append(memory)
        
        # 找出导致失败的数据，把最后五次决策的结果进行reflection
        else:
            # if "collision" in result_df["fail_reason"][0]:
            #     # 找到最后一个choose_action不是2的frame
            #     QA_df = pd.read_sql_query('''SELECT * FROM QAINFO''', conn)
            #     QA_df.loc[QA_df["choose_action"] != 2, ["choose_action"]] = -1
            #     QA_df.sort_values(by=['frame', 'choose_action'], ascending=[False, True], inplace=True)
            #     frame = QA_df["frame"][0]
            #     memory = MemoryItem()
            #     memory.set_description(QA_df[QA_df["frame"] == frame])
            #     memory.cautious = "you will have a collision with other vehicle"
            #     bad_memory.append(memory)
            for i in range(0, min(5, len(eval_df))):
                memory = MemoryItem()
                memory.set_description(QA_df.loc[len(eval_df) - i -1])
                memory.set_score(eval_df.loc[len(eval_df) - i - 1])
                fail_info = f"The result of this one decision could lead to the following problem {min(5, len(eval_df)) - i - 1} seconds later: {result_df['fail_reason'][0]}\n"
                memory.cautious = fail_info + memory.cautious
                bad_memory.append(memory)


        return good_memory, bad_memory

    def split(self, data) -> List[list]:
        res = []
        for i in range(len(data)):
            if not res:
                res.append([data[i]])
            elif data[i - 1] + 1 == data[i]:
                res[-1].append(data[i])
            else:
                res.append([data[i]])
        return res

if __name__ == "__main__":
    memory = DrivingMemory(db_path = "db/decision_mm/")
    
    good_mem, bad_mem = memory.divideBasedOnScore("experiments/zeroshot/GPT-4/2024-01-23_21-04-29.db")
    for mem_item in bad_mem:
        reflection_memory = memory.getReflection(mem_item)
        if reflection_memory == None:
            continue
        memory.addMemory(reflection_memory)
    for mem_item in good_mem:
        memory.addMemory(mem_item)
    # memory.deleteMemory(["168fa4b4-bc54-11ee-9e1f-87c41a6883a3"])

    # import json
    # with open("/home/PJLAB/leiwenjie/LimSimLLM/reflection.json", "r", encoding="utf-8") as f:
    #     reflection = json.load(f)
    #     reflection = reflection["reflection_data"]

    # for item in reflection:
    #     if item["add_memory"]:
    #         f = "\n## Available actions:\n"
    #         b = "## Driving scenario description:\n"
    #         key = "## Driving scenario description:\n" + item["human_question"].partition(b)[2].partition(f)[0]
    #         key = key.replace("'", "")

    #         target_phrase = f"#### Corrected version of Driver's Decision:"

    #         comment_phrase = f"#### What should driver do to avoid such errors in the future:"
    #         '''提取字符串s中，字符串f和b的中间部分'''
    #         comment = item["reflection"].partition(comment_phrase)[2].partition(target_phrase)[0]
    #         print("comment: ", comment)
    #         LLM_response = item["reflection"].partition(target_phrase)[2].strip()
    #         print("LLM_response: ", LLM_response)
    #         action = item["reflection_action"]
    #         target_phrase = "``` Human Message ```"
    #         huaman_phrase = "\n``` Driver's Decision ```"
    #         human_question = item["human_question"].partition(target_phrase)[2].partition(huaman_phrase)[0]
    #         print("human_question: ", human_question)
        
    #         get_results = memory.scenario_memory._collection.get(
    #             where_document={
    #                 "$contains": key
    #             }
    #         )

    #         if len(get_results['ids']) > 0:
    #             # already have one
    #             id = get_results['ids'][0]
    #             memory.scenario_memory._collection.update(
    #                 ids=id, metadatas={
    #                     "human_question": human_question,
    #                     'LLM_response': LLM_response,
    #                     'action': str(action),
    #                     'comments': comment
    #                 }
    #             )
    #             print("Modify a memory item. Now the database has ", len(
    #                 memory.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
#     memory_item = MemoryItem()
#     memory_item.description = """### Current lane description:
# You are driving on a road with 5 lanes in your direction, and you are currently driving in the number 4 lane from the left. The length of the current lane is 171.476 m. The limit speed of the current lane is 13.89 m/s. 
# ### Next lane description:
# The next lane is too far to consider. 
# ### Your current state:
# Your current position is '(733.813,354.536)', speed is 14.186 m/s, acceleration is -1.401 m/s^2, and lane position is 67.6 m. 
# ### Nearby vehicles description:
# There are no other vehicles driving near you, so you can drive completely according to your own ideas.
# ### Last decision:
# The last action you made 1.0 s ago was 2. 
# """
#     memory_item.navigation = """Stay in the current lane and continue straight ahead."""
#     memory_item.available_action = """Your available actions are: 
# Acceleration - accelerate the vehicle Action_id: 1
# IDLE - remain in the current lane with current speed Action_id: 8
# Deceleration - decelerate the vehicle Action_id: 2
# Turn-left - change lane to the left of the current lane Action_id: 3
# Turn-right - change lane to the right of the current lane Action_id: 4
# """
#     memory_item.response = """The current scenario is relatively simple, with no other vehicles nearby and no need to change lanes. The main considerations are maintaining a safe speed and following the navigation instructions.

# - First, I need to consider the speed limit. The current speed limit is 13.89 m/s, and my current speed is 14.186 m/s, which is slightly over the limit. Therefore, I need to decelerate to comply with the speed limit.
# - Second, I need to follow the navigation instructions. The instructions are to stay in the current lane and continue straight ahead, which aligns with my current trajectory. Therefore, there is no need to change lanes.

# Given these considerations, the best course of action is to decelerate to comply with the speed limit.

# Response to user:#### 2"""
#     memory_item.action = 2
#     memory.addMemory(memory_item)

    