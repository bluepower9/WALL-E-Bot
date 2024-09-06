import chromadb
from util import read_configs


class VectorDB():
    def __init__(self, threshold=1.5) -> None:
        self.configs = read_configs(filename='config.json')
        self.client = chromadb.PersistentClient(path = self.configs['database']['chromadb'])
        self.threshold = threshold


    def add_file(self, userid: int, docid: int, excerpts: list):
        collection = self.client.get_or_create_collection(f'user-{userid}')
        
        documents, ids = [], []
        for e in excerpts:
            documents.append(e['excerpt'])
            ids.append(str(e['excerpt_id'])) 

        collection.add(
            documents=documents,
            metadatas=[{'docid': docid} for i in range(len(documents))],
            ids=ids
        )

    
    def delete(self, userid:int, docid: int):
        try:
            collection = self.client.get_collection(f'user-{userid}')
            collection.delete(where={'docid': docid})

        except ValueError as e:
            print('Could not find collection. error: ' + str(e))
            return False


    def query(self, s: str, userid: int):
        userid = f'user-{userid}'
        try:
            collection = self.client.get_collection(userid)

        except ValueError as e:
            print('Could not find collection. error: ' + str(e))
            return []

        excerpts = collection.query(query_texts=s)
        result = []

        excerpts = zip(excerpts['ids'][0], excerpts['distances'][0], excerpts['documents'][0])

        for id, d, text in excerpts:
            if d <= self.threshold:
                result.append({'id': id, 'distance': d, 'text': text})

        return result
