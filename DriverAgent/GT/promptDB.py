import sqlite3
from typing import List


class DBBridge:
    def __init__(self, database: str) -> None:
        self.database = database

    def createTable(self):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS promptsINFO(
                timeStep REAL PRIMARY KEY,
                vectorID TEXT,
                done BOOL,
                description TEXT,
                fewshots TEXT,
                thoughtsAndAction TEXT,
                editedTA TEXT,
                editTimes INT
            );"""
        )
        conn.commit()
        conn.close()

    def insertPrompts(
            self, timeStep: int, vectorID: str, done: bool,
            description: str, fewshots: str, thoughtsAndAction: str
    ):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO promptsINFO (
                timeStep, vectorID, done, description, fewshots, 
                thoughtsAndAction, editedTA, editTimes
                ) VALUES (?,?,?,?,?,?,?,?);""",
            (
                timeStep, vectorID, done, description,
                fewshots, thoughtsAndAction, None, 0
            )
        )
        conn.commit()
        conn.close()
