import os
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import json
# from loadConfig import load_openai_config
import sqlite3
import pandas as pd
from DriverAgent.Reflection import ReflectionAssistant
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
            "collision_score": score_df["collision_score"]
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
        if "azure" != 'azure':
            self.embedding = OpenAIEmbeddings()
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
                    'LLM_response': memory.response, 'action': memory.action,
                    'comments': memory.reflection
                }
            )
            print("Modify a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
        else:
            doc = Document(
                page_content=key,
                metadata={
                    "human_question": human_question,
                    'LLM_response': memory.response, 'action': memory.action,
                    'comments': memory.reflection
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

        evaluation = f"The current score is {memory.score}, the detail score is {detail_score.strip(', ')}. There are the current decision's cautions:\n{memory.cautious}"
        
        LLM_response = self.reflector.reflection(human_message, memory.response, evaluation)
        
        target_phrase = f"{delimiter} Corrected version of Driver's Decision:"
        correct_answer = LLM_response[LLM_response.find(
            target_phrase)+len(target_phrase):].strip()
        
        target_phrase = f"{delimiter} What should driver do to avoid such errors in the future:"
        substring = LLM_response[LLM_response.find(
            target_phrase)+len(target_phrase):].strip()
        comment = f"{delimiter} I have made a misake before and below is my self-reflection:\n{substring}"
        action = int(LLM_response.split(delimiter)[-1].strip(" "))
        memory.set_reflection(correct_answer, comment, action)
        return memory
    
    def divideBasedOnScore(self, database: str) -> Tuple[List[MemoryItem], List[MemoryItem]]:

        good_memory = []
        bad_memory = []

        conn = sqlite3.connect(database)
        eval_df = pd.read_sql_query('''SELECT * FROM evaluationINFO''', conn)
        QA_df = pd.read_sql_query('''SELECT * FROM QAINFO''', conn)

        result_df = pd.read_sql_query('''SELECT * FROM resultINFO''', conn)
            
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
        
        # 找出导致失败的数据
        if not result_df["result"][0]:
            if "collision" in result_df["fail_reason"][0]:
                # 找到最后一个choose_action不是2的frame
                QA_df = pd.read_sql_query('''SELECT * FROM QAINFO''', conn)
                QA_df.loc[QA_df["choose_action"] != 2, ["choose_action"]] = -1
                QA_df.sort_values(by=['frame', 'choose_action'], ascending=[False, True], inplace=True)
                frame = QA_df["frame"][0]
                memory = MemoryItem()
                memory.set_description(QA_df[QA_df["frame"] == frame])
                memory.cautious = "you will have a collision with other vehicle"
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
    from DriverAgent.test.loadConfig_memory import load_openai_config
    load_openai_config()
    memory = DrivingMemory()
    
    good_mem, bad_mem = memory.divideBasedOnScore("./results/2024-01-18_16-12-17.db")
    memory.addMemory(good_mem[0])
    for mem_item in bad_mem:
        reflection_memory = memory.getReflection(mem_item)
        memory.addMemory(reflection_memory)
    for mem_item in good_mem:
        memory.addMemory(reflection_memory)

    # memory = DrivingMe
    