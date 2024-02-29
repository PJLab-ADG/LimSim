import sqlite3
from typing import Tuple


class DBBridge:
    def __init__(self, database: str) -> None:
        self.database = database
        self.commitQueue = []
        self.commitCnt = 0

    def createTable(self):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()

        cur.execute('''CREATE TABLE IF NOT EXISTS simINFO(
                        excuteTime TIMESTAMP PRIMARY KEY,
                        egoID TEXT,
                        netBoundary TEXT);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS frameINFO(
                            frame INT NOT NULL,
                            vid TEXT NOT NULL,
                            vtag TEXT NOT NULL,
                            x REAL NOT NULL,
                            y REAL NOT NULL,
                            yaw REAL NOT NULL,
                            speed REAL NOT NULL,
                            accel REAL NOT NULL,
                            laneID TEXT NOT NULL,
                            lanePos REAL NOT NULL,
                            routeIdx INT NOT NULL,
                            PRIMARY KEY (frame, vid));''')

        cur.execute('''CREATE TABLE IF NOT EXISTS vehicleINFO(
                            vid TEXT PRIMARY KEY,
                            length REAL NOT NULL,
                            width REAL NOT NULL,
                            maxAccel REAL,
                            maxDecel REAL,
                            maxSpeed REAL,
                            vTypeID TEXT NOT NULL,
                            routes TEXT NOT NULL);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS edgeINFO(
                            id TEXT RRIMARY KEY,
                            laneNumber INT NOT NULL,
                            from_junction TEXT,
                            to_junction TEXT);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS laneINFO(
                            id TEXT PRIMARY KEY,
                            rawShape TEXT,
                            width REAL,
                            maxSpeed REAL,
                            edgeID TEXT,
                            length REAL,
                            type TEXT,
                            allow TEXT,
                            disallow TEXT);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS junctionLaneINFO(
                            id TEXT PRIMARY KEY,
                            width REAL,
                            maxSpeed REAL,
                            length REAL,
                            tlsIndex INT,
                            type TEXT,
                            allow TEXT,
                            disallow TEXT);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS junctionINFO(
                            id TEXT PRIMARY KEY,
                            rawShape TEXT);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS connectionINFO(
                            fromLane TEXT NOT NULL,
                            toLane TEXT NOT NULL,
                            direction TEXT,
                            via TEXT,
                            PRIMARY KEY (fromLane, toLane));''')

        cur.execute('''CREATE TABLE IF NOT EXISTS trafficLightStates(
                            frame INT NOT NULL,
                            id TEXT NOT NULL,
                            currPhase TEXT,
                            PRIMARY KEY (frame, id));''')

        cur.execute('''CREATE TABLE IF NOT EXISTS circleObsINFO(
                            id TEXT PRIMARY KEY,
                            edgeID TEXT NOT NULL,
                            centerx REAL NOT NULL,
                            centery REAL NOT NULL,
                            radius REAL NOT NULL);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS rectangleObsINFO(
                            id TEXT PRIMARY KEY,
                            edgeID TEXT NOT NULL,
                            centerx REAL NOT NULL,
                            centery REAL NOT NULL,
                            length REAL NOT NULL,
                            width REAL NOT NULL,
                            yaw REAL NOT NULL);''')

        cur.execute('''CREATE TABLE IF NOT EXISTS geohashINFO(
                            ghx INT NOT NULL,
                            ghy INT NOT NULL,
                            edges TEXT,
                            junctions TEXT,
                            PRIMARY KEY (ghx, ghy));''')
        
        cur.execute('''CREATE TABLE IF NOT EXISTS imageINFO(
                    frame INT PRIMARY KEY,
                    CAM_FORNT BLOB,
                    CAM_FRONT_RIGHT BLOB,
                    CAM_FRONT_LEFT BLOB,
                    CAM_BACK_LEFT BLOB,
                    CAM_BACK BLOB,
                    CAM_BACK_RIGHT BLOB);''')
        
        cur.execute('''CREATE TABLE IF NOT EXISTS QAINFO(
                    frame INT PRIMARY KEY,
                    description TEXT,
                    navigation TEXT,
                    actions TEXT,
                    few_shots TEXT,
                    response TEXT,
                    prompt_tokens INT,
                    completion_tokens INT,
                    total_tokens INT,
                    total_time REAL,
                    choose_action INT);''')
        cur.execute(
            """CREATE TABLE IF NOT EXISTS resultINFO(
                egoID TEXT PRIMARY KEY,
                result BOOLEAN,
                total_score REAL,
                complete_percentage REAL,
                drive_score REAL,
                use_time REAL,
                fail_reason TEXT
            );"""
        )

        conn.commit()
        cur.close()
        conn.close()

    def commitData(self):
        conn = sqlite3.connect(self.database)
        cur = conn.cursor()
        for tableName, data in self.commitQueue:
            sql = f'INSERT INTO {tableName} VALUES' + '(' + '?,'*(len(data)-1) + '?' + ')'
            try:
                cur.execute(sql, data)
            except sqlite3.IntegrityError:
                pass

        conn.commit()
        conn.close()

    def putData(self, tableName: str,  data: Tuple):
        self.commitQueue.append((tableName, data))
        self.commitCnt += 1
        if self.commitCnt >= 100:
            self.commitData()
            self.commitQueue = []
            self.commitCnt = 0
        else:
            self.commitCnt += 1

    def close(self):
        if self.commitQueue:
            self.commitData()
        else:
            return
