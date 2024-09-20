import peewee as pw
import os
import base64
import pickle
from dataclasses import dataclass


dbName = "simulationDatabase.db"


@dataclass
class VehicleInfo:
    id: int
    length: float
    width: float
    x: float
    y: float
    scf: float
    tcf: float
    hdg: float
    deArea: float
    roadId: str
    EXPECT_VEL: float
    xQ: list[float]
    yQ: list[float]
    hdgQ: list[float]
    velQ: list[float]
    accQ: list[float]
    yawQ: list[float]
    roadIdQ: list[str]
    planXQ: list[float]
    planYQ: list[float]
    planHdgQ: list[float]
    planVelQ: list[float]
    planAccQ: list[float]
    planYawQ: list[float]
    planRoadIdQ: list[str]


class Database:
    def __init__(self) -> None:
        self.db = pw.SqliteDatabase(dbName)

    def initDB(self):
        if os.path.exists(dbName):
            os.remove(dbName)
        self.db.connect()
        self.db.create_tables([VehicleDB, tlsDB])

    def updateDB(self, vehRunning, tls, timeStep: int):
        vehs = dict()
        for veh in vehRunning.values():
            vehs[veh.id] = VehicleInfo(
                veh.id,
                veh.length,
                veh.width,
                veh.x,
                veh.y,
                veh.scf,
                veh.tcf,
                veh.hdg,
                veh.deArea,
                veh.roadId,
                veh.EXPECT_VEL,
                veh.xQ,
                veh.yQ,
                veh.hdgQ,
                veh.velQ,
                veh.accQ,
                veh.yawQ,
                veh.roadIdQ,
                veh.planXQ,
                veh.planYQ,
                veh.planHdgQ,
                veh.planVelQ,
                veh.planAccQ,
                veh.planYawQ,
                veh.planRoadIdQ,
            )

        newVeh = VehicleDB(
            timeStep=timeStep,
            info=base64.b64encode(pickle.dumps(vehs)).decode("utf-8"),
        )
        newVeh.save()
        newTls = tlsDB(
            timeStep=timeStep,
            info=base64.b64encode(pickle.dumps(tls)).decode("utf-8"),
        )
        newTls.save()

    def getRunTime(self):
        return VehicleDB.select(pw.fn.Max(VehicleDB.timeStep)).scalar()

    def getDB(self, timeStep: int):
        try:
            self.db.connect()
        except:
            pass
        vehInfo = pickle.loads(
            base64.b64decode(VehicleDB.get(timeStep=timeStep).info.encode("utf-8"))
        )
        tls = pickle.loads(
            base64.b64decode(tlsDB.get(timeStep=timeStep).info.encode("utf-8"))
        )
        return vehInfo, tls

    def closeDB(self):
        self.db.close()


class VehicleDB(pw.Model):
    timeStep = pw.IntegerField()
    info = pw.TextField()

    class Meta:
        database = pw.SqliteDatabase(dbName)


class tlsDB(pw.Model):
    timeStep = pw.IntegerField()
    info = pw.TextField()

    class Meta:
        database = pw.SqliteDatabase(dbName)
