import chromadb
import datetime
import time
from util import read_configs


class VectorDB():
    def __init__(self, threshold=1.3) -> None:
        self.configs = read_configs(filename='config.json')
        self.client = chromadb.PersistentClient(path = self.configs['database']['chromadb'])
        self.threshold = threshold


    def add_message(self, userid: int, msgid: int, message: str, speaker: str, timestamp:int=None):
        '''
        Adds new message to the vectorDB for the user. Should match with the SQL database.

        :param userid: userid of the user.
        :param msgid: message id of the message in the SQL database.
        :param message: message that was spoken or generated.
        :param speaker: 0 for user or 1 for bot for who spoke the message.
        '''
        collection = self.client.get_or_create_collection(f'user-{userid}')
        
        if timestamp is None:
            timestamp = time.time()
        timestamp = int(timestamp)

        collection.add(
            documents=[message],
            metadatas=[{'speaker': speaker, 'timestamp': timestamp}],
            ids=[str(msgid)]
        )

    
    def delete(self, userid:int, msgid: int):
        try:
            collection = self.client.get_collection(f'user-{userid}')
            collection.delete(where={'id': msgid})

        except ValueError as e:
            print('Could not find collection. error: ' + str(e))
            return False


    def query(self, s: str, userid: int):
        userid = f'user-{userid}'
        try:
            collection = self.client.get_collection(userid)

        except ValueError as e:
            print('Error querying VectorDB. Could not find collection. error: ' + str(e))
            return []

        excerpts = collection.query(query_texts=s)
        result = []

        excerpts = zip(excerpts['ids'][0], excerpts['distances'][0], excerpts['documents'][0], excerpts['metadatas'][0])

        for id, d, text, metadata in excerpts:
            if d <= self.threshold:
                result.append({'id': id, 'distance': d, 'text': text})

        return result
