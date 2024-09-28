from AI.persistent_memory.vectordb import VectorDB
from AI.persistent_memory.sqldb import SQLDatabase
from util import load_people_encodings
import time
import datetime

class Memory:
    def __init__(self):
        self.vectordb = VectorDB()
        self.sqldb = SQLDatabase()
        self.reload()


    def reload(self):
        self.people = load_people_encodings()
        for k, v in self.people.items():
            del v['face_encoding']
            del v['voice_encoding']

    
    def add_utterance(self, userid:int, msg: str, speaker: int):
        '''
        Adds the utterance from a conversation into the databases.

        :param userid: userid of the conversation participant.
        :param msg: what was spoken.
        :param speaker: 0 for user 1 for bot
        '''
        timestamp = int(time.time())
        id = self.sqldb.insert_message(userid, msg, speaker, timestamp=timestamp)
        self.vectordb.add_message(userid, id, msg, speaker, timestamp=timestamp)


    def search_memory(self, userid:int, msg:str, count:int=10) -> list:
        '''
        Searches for relevant memories given        
        '''
        result = []
        msgs = {}

        closest = self.vectordb.query(msg, userid)
        for m in closest:
            id = int(m['id'])
            data = self.get_context_messages(userid, id)
            
            for d in data:
                msgid = d['message_id']
                del d['message_id']
                msgs[msgid] = d
        
        return msgs
  
    
    def create_chat_string(self, messages:dict):
        '''
        Takes in a dict of the chat messages from search_memory and returns a string representation of the chat.      
        '''
        keys = sorted(list(messages.keys()))
        result = []
        prevdate = ''

        for k in keys:
            msg = messages[k]
            dateobj = datetime.datetime.fromtimestamp(msg.get('timestamp', None))

            date = dateobj.strftime('--- %b %d, %Y ---')
            msgtime = dateobj.strftime('%H:%M:%S')

            # adds new date above messages when the date changes
            if prevdate != date:
                result.append('\n' + date)
                prevdate = date
            
            # gets the speaker of the message
            spkr = msg.get('speaker', -1)
            if spkr == 0:
                name = self.people[msg['user_id']]['name']
            elif spkr == 1:
                name = 'Bot'
            else:
                name = 'Unknown'
        
            s = f'[{msgtime}] {name}: {msg["message"]}'
            result.append(s)
        
        return '\n'.join(result)



    def get_context_messages(self, userid: int, id: int, count:int=5):
        prev = 'SELECT * FROM messages WHERE user_id=? and message_id<? ORDER BY message_id DESC'
        nxt = 'SELECT * FROM messages WHERE user_id=? and message_id>? ORDER BY message_id ASC'

        res = []

        res.extend(self.sqldb.fetch(prev, args=(userid, id), count=count))
        res.extend(self.sqldb.fetch(nxt, args=(userid, id), count=count))

        return res


        


