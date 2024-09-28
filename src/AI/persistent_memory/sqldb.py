import sqlite3 as sq
import os
import time


class SQLDatabase:
    def __init__(self, dbpath='./data/databases/conversations.db') -> None:
        self.dbpath = dbpath
        createdb = not os.path.isfile(self.dbpath)
        self.con = sq.connect(self.dbpath)
        self.con.row_factory = sq.Row
        self.cur = self.con.cursor()
        if createdb:
            self.setupDB()
        
    
    def setupDB(self):
        try:
            with open('./AI/persistent_memory/dbsetup.sql', 'r') as file:
                script = file.read()
            
            print('creating new database...')
            self.cur.executescript(script)
            self.con.commit()
        
        except FileNotFoundError:
            print('SQL setup script not found.')

    
    def fetch(self, sql, args=(), count=None):
        '''
        :param sql: SQL query with '?' to denote args
        :param args: list of args to match with the '?'
        :param count: count of how many values to return. If none, returns all values found.
        '''
        try:
            res = self.cur.execute(sql, args)
            data = res.fetchall() if count is None else res.fetchmany(count)

            return [dict(d) for d in data]
            
        except sq.Error as e:
            print(e)


    def insert(self, sql, args=()):
        res = self.cur.execute(sql, args)
        self.con.commit()

    
    def insert_message(self, userid:int, message:str, speaker:int, timestamp:int=None) -> int:
        '''
        Inserts a message into the SQL database.

        :param userid: userid of who conversation is with.
        :param message: message that was spoken.
        :param speaker: 0 for user, 1 for the bot.
        :param timestamp: int of unix epoch.

        :returns msgid: Message id of the saved message. Can be used to save into vector db
        '''
        if timestamp is None:
            timestamp = time.time()
        timestamp = int(timestamp)

        sql = "INSERT INTO messages (user_id, timestamp, message, speaker) VALUES(?, ?, ?, ?)"
        self.cur.execute(sql, (userid, timestamp, message, speaker))
        self.con.commit()

        return self.cur.lastrowid




if __name__ == '__main__':
    db = SQLDatabase('test.db')
