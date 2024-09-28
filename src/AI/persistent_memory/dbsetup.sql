CREATE TABLE messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    message TEXT NOT NULL,
    speaker INTEGER NOT NULL 
);
