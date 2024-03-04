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
        ## For azure user, if you want to use azure key in this project, you need to check if your azure has deployed embedding model
        # if os.environ["OPENAI_API_TYPE"] == 'azure':
        #     self.embedding = OpenAIEmbeddings(
        #             deployment=os.environ['EMBEDDING_MODEL'], chunk_size=1)
        # elif os.environ["OPENAI_API_TYPE"] == 'openai':
        #     self.embedding = OpenAIEmbeddings()
        # else:
        #     raise ValueError(
        #         "Unknown OPENAI_API_TYPE: should be azure or openai")

        self.embedding = OpenAIEmbeddings() # openai embedding model
        
        db_path = os.path.join(
            './db', 'memory_library/') if db_path is None else db_path
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

    
    def getReflection(self, memory: MemoryItem, method: bool) -> MemoryItem:
        """use the reflection assistant to get the reflection of the memory item

        Args:
            memory (MemoryItem): the memory item
            method (bool): decide to auto reflection or manual reflection
                            if True, use the auto reflection
                            if False, use the manual reflection

        Returns:
            MemoryItem: the memory item with reflection
        """
        delimiter = "####"
        human_message = "## Driving scenario description:\n" + memory.description + "\n## Navigation instruction:\n" + memory.navigation + "\n## Available actions:\n" + memory.available_action
        detail_score = ""
        for key, value in memory.detail_score.items():
            detail_score += f"{key}: {value}, "

        evaluation = f"The current score is {memory.score}, the detail score is {detail_score.strip(', ')}."
        caution = "There are the current decision's cautions:\n{cautious}".format(cautious=memory.cautious if memory.cautious != "" else "None\n")
        
        LLM_response, action = self.reflector.reflection(human_message, memory.response, evaluation, memory.action, caution, method)
        
        if LLM_response == None:
            return None
        
        if method:
            decision_phrase = f"{delimiter} Corrected version of Driver's Decision:"
            correct_answer = LLM_response.partition(decision_phrase)[2].strip()

            comment_phrase = f"{delimiter} What should driver do to avoid such errors in the future:"
            comment = LLM_response.partition(comment_phrase)[2].partition(decision_phrase)[0].strip()
            memory.set_reflection(correct_answer, comment, action)
        else:
            memory.set_reflection("", LLM_response, action) # manual reflection, no need to format the response
        return memory
    
    def divideBasedOnScore(self, database: str) -> Tuple[List[MemoryItem], List[MemoryItem]]:
        """After finishing a route, the quality of decision is evaluated according to the decision score of each frame obtained by the evaluation module. First, the decision curve is segmented by the median. Then, the decision with the highest score in each continuous curve above the median is taken as a good decision. The decision result with the lowest score in each continuous curve less than the median is taken as the decision to be corrected.

        Args:
            database (str): Database storing decisions and evaluation scores made by LLM Driver in a route.

        Returns:
            Tuple[List[MemoryItem], List[MemoryItem]]: Make the good decision and bad decision as memory item, return the good memories and bad memories.
        """
        good_memory = []
        bad_memory = []

        conn = sqlite3.connect(database)
        eval_df = pd.read_sql_query('''SELECT * FROM evaluationINFO''', conn)
        QA_df = pd.read_sql_query('''SELECT * FROM QAINFO''', conn)

        result_df = pd.read_sql_query('''SELECT * FROM resultINFO''', conn)
        
        if result_df["result"][0]:
            # find median
            middle_score = eval_df["decision_score"].quantile(q = 0.2, interpolation="linear")

            # segmented by median
            low_index = eval_df[eval_df["decision_score"] <= middle_score].index.tolist()
            high_index = eval_df[eval_df["decision_score"] > middle_score].index.tolist()
            # segment the decision curve
            low_split = self.split(low_index)
            high_split = self.split(high_index)
            bad_mem_index = []
            good_mem_index = []
            # judge each segment
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

            good_eval_df = eval_df.loc[good_mem_index]
            bad_eval_df = eval_df.loc[bad_mem_index]

            # fine the QAINFO
            good_QA_df = QA_df.loc[good_mem_index]
            bad_QA_df = QA_df.loc[bad_mem_index]

            # convert to memory item
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
        
        # If the route fails, the results of the last five decisions are taken as the decision to be corrected.
        else:
            # if "collision" in result_df["fail_reason"][0]:
            #     # find the last frame which the choose action is not deceleration
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
    memory = DrivingMemory(db_path = "db/decision_mem/")
    
    good_mem, bad_mem = memory.divideBasedOnScore("results/2024-02-21_18-46-19.db")
    for mem_item in bad_mem:
        reflection_memory = memory.getReflection(mem_item, True)
        if reflection_memory == None:
            continue
        memory.addMemory(reflection_memory)
    for mem_item in good_mem:
        memory.addMemory(mem_item)
    