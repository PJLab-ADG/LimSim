import numpy as np
from simModel.networkBuild import NetworkBuild
from trafficManager.vehicle import Vehicle
from simModel.common.dataQueue import DynamicInfo, EgoVehicleInfo, VehicleInfo, TlsInfo


class MovingScene:
    def __init__(self, netInfo: NetworkBuild) -> None:
        self.netInfo: NetworkBuild = netInfo
        self.vehInAoI: dict[str, Vehicle] = {}
        self.outOfAoI: dict[str, Vehicle] = {}
        self.vehInfo = None

    def updateScene(self, ego: Vehicle):
        ex, ey = ego.x, ego.y
        currGeox = int(ex // 100)
        currGeoy = int(ey // 100)
        sceGeohashIDs = (
            (currGeox - 1, currGeoy - 1),
            (currGeox, currGeoy - 1),
            (currGeox + 1, currGeoy - 1),
            (currGeox - 1, currGeoy),
            (currGeox, currGeoy),
            (currGeox + 1, currGeoy),
            (currGeox - 1, currGeoy + 1),
            (currGeox, currGeoy + 1),
            (currGeox + 1, currGeoy + 1),
        )
        NowSections: set = set()
        for sgh in sceGeohashIDs:
            try:
                geohash = self.netInfo.geoHashes[sgh]
            except KeyError:
                continue
            NowSections = NowSections | geohash.laneSections

    def updateSurroudVeh(self, ego: Vehicle, vehicle_running: dict[str, Vehicle]):
        """GetSurroundVeh will update all vehicle's attributes."""
        self.vehInAoI = {}
        self.outOfAoI = {}
        allcar = vehicle_running
        # get other cars by Euclidean distance
        for vehId in allcar:
            veh = allcar[vehId]
            roads = self.netInfo.roads
            if ego.id != veh.id:
                if np.hypot(ego.scf - veh.scf, ego.tcf - veh.tcf) < ego.deArea:
                    self.vehInAoI[vehId] = veh
                else:
                    self.outOfAoI[vehId] = veh

    def getPlotInfo(self, ego, tls):
        """Generate the plot information for the moving scene."""
        vehInAoI = {}
        for veh in self.vehInAoI.values():
            vehInfo = VehicleInfo(
                veh.id,
                veh.length,
                veh.width,
                veh.x,
                veh.y,
                veh.hdg,
                veh.planXQ,
                veh.planYQ,
            )
            vehInAoI[veh.id] = vehInfo
        outOfAoI = {}
        for veh in self.outOfAoI.values():
            vehInfo = VehicleInfo(
                veh.id,
                veh.length,
                veh.width,
                veh.x,
                veh.y,
                veh.hdg,
                veh.planXQ,
                veh.planYQ,
            )
            outOfAoI[veh.id] = vehInfo

        egoInfo = EgoVehicleInfo(
            ego.id,
            ego.length,
            ego.width,
            ego.x,
            ego.y,
            ego.hdg,
            ego.planXQ,
            ego.planYQ,
            ego.deArea,
            ego.roadId,
            ego.drivingScore,
            ego.velQ,
            ego.accQ,
            ego.planVelQ,
            ego.planAccQ,
        )

        tlsInfo = {}
        for key in tls:
            tlsInfo[key] = TlsInfo(tls[key].currentState)
        self.vehInfo = DynamicInfo(egoInfo, vehInAoI, outOfAoI, tlsInfo)
