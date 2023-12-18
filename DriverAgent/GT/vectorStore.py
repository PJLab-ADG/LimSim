import os
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import json
# from loadConfig import load_openai_config

class DrivingMemory:
    def __init__(self, encode_type='sce_language', db_path=None) -> None:
        self.encode_type = encode_type
        if encode_type == 'sce_language':
            if os.environ["OPENAI_API_TYPE"] == 'azure':
                self.embedding = OpenAIEmbeddings(
                    deployment=os.environ['EMBEDDING_MODEL'], chunk_size=1)
            elif os.environ["OPENAI_API_TYPE"] == 'openai':
                self.embedding = OpenAIEmbeddings()
            else:
                raise ValueError(
                    "Unknown OPENAI_API_TYPE: should be azure or openai")
            db_path = os.path.join(
                './db', 'chroma_5_shot_20_mem/') if db_path is None else db_path
            self.scenario_memory = Chroma(
                embedding_function=self.embedding,
                persist_directory=db_path
            )
        else:
            raise ValueError(
                "Unknown ENCODE_TYPE: should be `sce_language`")
        print("==========Loaded Memory, Now the database has ", len(
            self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.==========")

    def retriveMemory(self, query_scenario, top_k: int = 5):
        similarity_results = self.scenario_memory.similarity_search_with_score(
            query_scenario, k=top_k)
        fewshot_results = []
        for idx in range(0, len(similarity_results)):
            # print(f"similarity score: {similarity_results[idx][1]}")
            fewshot_results.append(similarity_results[idx][0].metadata)
        return fewshot_results

    def addMemory(self, sce_descrip: str, human_question: str, response: str, action: int, comments: str = ""):
        sce_descrip = sce_descrip.replace("'", '')
        # https://docs.trychroma.com/usage-guide#using-where-filters
        get_results = self.scenario_memory._collection.get(
            where_document={
                "$contains": sce_descrip
            }
        )

        if len(get_results['ids']) > 0:
            # already have one
            id = get_results['ids'][0]
            self.scenario_memory._collection.update(
                ids=id, metadatas={
                    "human_question": human_question,
                    'LLM_response': response, 'action': action,
                    'comments': comments
                }
            )
            print("Modify a memory item. Now the database has ", len(
                self.scenario_memory._collection.get(include=['embeddings'])['embeddings']), " items.")
        else:
            doc = Document(
                page_content=sce_descrip,
                metadata={
                    "human_question": human_question,
                    'LLM_response': response, 'action': action,
                    'comments': comments
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


if __name__ == "__main__":
    load_openai_config()
    db = DrivingMemory()
    with open("few_shot.json", "r", encoding="utf-8") as f:
        few_shot = json.load(f)
        few_shot_list = few_shot["init_few_shot"]
    for i in range(0, len(few_shot_list)):
        db.addMemory(few_shot_list[i]['sce_descrip'], few_shot_list[i]['human_question'], few_shot_list[i]['response'], few_shot_list[i]['action'])
