import numpy as np
import sys
from trafficManager.vehicle import Vehicle
from simModel.networkBuild import NetworkBuild


class Planner:
    def __init__(self, ego: Vehicle, nb: NetworkBuild):
        self.ego = ego
        self.nb = nb
        self.surcar = None

    def IDM(self, v_cur: float, v_lead: float, s_cur: float, v_des=10):
        """
        - reference: https://traffic-simulation.de/info/info_IDM.html
        - The function IDM takes four inputs:
        - v_cur: the current velocity of the vehicle.
        - v_lead: the velocity of the leading vehicle (or -1 if there is no leading vehicle).
        - s_cur: the current distance to the leading vehicle (must be > 0).
        - v_des: the desired velocity of the vehicle (default is 10 m/s).
        - The function returns the acceleration of the vehicle based on the IDM model.
        """
        T = 3
        s_0 = 10.0
        acc = 1.0
        dec = 3.0
        a_max = acc
        if v_lead == -1:
            v_lead = v_des
            s_cur = sys.maxsize

        d_v = v_cur - v_lead
        if s_cur == 0:
            s_cur = 0.00001
        s_star = s_0 + max(0, (v_cur * T + (v_cur * d_v) / (2 * np.sqrt(acc * dec))))
        a = a_max * (1 - pow(v_cur / v_des, 4) - pow(s_star / s_cur, 2))
        if a > 0:
            a = min(acc, a)
        if a < 0:
            a = max(-dec, a)
        return a
        # allcars are the status before reaction time

    def getSurCar(self, vehicle_running, radius=100):
        """
        This function retrieves a dictionary of surrounding cars within a specified radius (default 100) of the ego vehicle.
        It filters cars based on the same road ID, direction, and Euclidean distance from the ego vehicle's position (scf and tcf attributes).
        """
        surCar = {}
        allcar = vehicle_running
        for other_id in allcar:
            other = allcar[other_id]
            if self.ego != other:
                if (
                    self.ego.scf >= 0
                    and other.roadId == self.ego.roadId
                    and other.direction == self.ego.direction
                    and np.hypot(self.ego.scf - other.scf, self.ego.tcf - other.tcf)
                    < radius
                ):
                    surCar[other_id] = other
        return surCar

    def turnLeft(self, straight_status: list, frequency: int) -> list:
        """
        This function determines the trajectory of a vehicle turning left.        It calculate the safety distance from the follow vehicle and the distance after 1 second when the ego vehicle enters the other lane.

        Parameters:
            straight_status: A list of states of the ego vehicle at each time step when it goes straight.
            frequency: The frequency of the simulation or time steps per second.

        Returns:
            list: states of the ego vehicle at each time step, including its
                x-coordinate, y-coordinate, s-coordinate, t-coordinate, yaw, velocity, and acceleration.
            None: cannot turn left.
        """
        cur_lane = self.nb.lanes[self.ego.laneId]
        # cannot turn left if there is no left lane
        if cur_lane.left:
            left_lane = cur_lane.left
        else:
            return
        # assume ego vehicle keeps the lane with in 1 second
        # determine the lead vehicle in the left lane
        lead_vehicle = None
        follow_vehicle = None
        for vehicle_id in self.surcar:
            vehicle = self.surcar[vehicle_id]
            if (
                vehicle.scf > self.ego.scf
                and vehicle.laneId == left_lane.id
                and (not lead_vehicle or vehicle.scf < lead_vehicle.scf)
            ):
                lead_vehicle = vehicle
            if (
                vehicle.scf <= self.ego.scf
                and vehicle.laneId == left_lane.id
                and (not follow_vehicle or vehicle.scf > follow_vehicle.scf)
            ):
                follow_vehicle = vehicle
        # safety distane from following vehicle
        if follow_vehicle:
            acc_follow = self.IDM(
                follow_vehicle.vel, self.ego.vel, abs(self.ego.scf - follow_vehicle.scf)
            )
            if acc_follow < 0:
                return
        if lead_vehicle:
            # distance after 1 second (ego vehicle enter the other lane)
            lead_scf = lead_vehicle.scf + lead_vehicle.vel * 1.0
            ego_scf = straight_status[frequency - 1][0]
            acc_calc = self.IDM(self.ego.vel, lead_vehicle.vel, abs(lead_scf - ego_scf))
        else:
            acc_calc = self.IDM(self.ego.vel, -1, -1)
        left_status = []
        timeGap = 1 / frequency
        dis = np.hypot(
            cur_lane.centerLine[0][0] - left_lane.centerLine[0][0],
            cur_lane.centerLine[0][1] - left_lane.centerLine[0][1],
        )
        # calculate tcf
        # turing period is 2.0 second
        # using sin function to smooth the lateral motion
        delta_tcf = [
            np.sin(p / (2.0 + timeGap) * np.pi)
            for p in np.arange(timeGap, 2.0 + timeGap, timeGap)
        ]
        delta_tcf = delta_tcf / sum(delta_tcf) * dis

        for t in np.arange(timeGap, self.ego.PLAN_LEN + timeGap, timeGap):
            index = round(t / timeGap)
            if t <= 1.0:
                scf = straight_status[index - 1][0]
                vel = straight_status[index - 1][3]
                acc = straight_status[index - 1][4]
            else:
                acc = min(acc_calc, self.IDM(vel, -1, -1))
                if vel + acc * timeGap >= 0:
                    scf += vel * timeGap + 0.5 * acc * timeGap**2
                    vel += acc * timeGap
                else:
                    scf += (vel + 0) / 2
                    vel = 0
            if t <= 2.0:
                tcf = straight_status[index - 1][1] - dis / 2.0 * t
            else:
                tcf = left_status[-1][1]
            if t == timeGap:
                dscf = scf - self.ego.scf
                dtcf = delta_tcf[index - 1]
                tcf = self.ego.tcf - dtcf
                yaw = np.arctan2(dtcf, dscf) * -1
            elif t <= 2.0:
                dscf = scf - left_status[-1][0]
                dtcf = delta_tcf[index - 1]
                tcf = left_status[-1][1] - dtcf
                yaw = np.arctan2(dtcf, dscf) * -1
                # yaw cannot be larger than 45 degrees
                if dscf < dtcf:
                    return
            else:
                tcf = left_status[-1][1]
                yaw = 0.0
            # cannot turn left when the vehicle arrive at the end of the lane
            if scf >= cur_lane.length:
                return
            left_status.append([scf, tcf, yaw, vel, acc])
        # cannot turn left when the vehicle do not move longer distance
        if left_status[-1][0] < straight_status[-1][0] * 1.05:
            return
        return left_status

    def turnRight(self, straight_status, frequency) -> list:
        # cannot turn right when there is no right lane
        cur_lane = self.nb.lanes[self.ego.laneId]
        if cur_lane.right:
            right_lane = cur_lane.right
        else:
            return
        # assume ego vehicle keeps the lane with in 1 second
        # determine the lead vehicle in the right lane
        lead_vehicle = None
        follow_vehicle = None
        for vehicle_id in self.surcar:
            vehicle = self.surcar[vehicle_id]
            if (
                vehicle.scf > self.ego.scf
                and vehicle.laneId == right_lane.id
                and (not lead_vehicle or vehicle.scf < lead_vehicle.scf)
            ):
                lead_vehicle = vehicle
            if (
                vehicle.scf <= self.ego.scf
                and vehicle.laneId == right_lane.id
                and (not follow_vehicle or vehicle.scf > follow_vehicle.scf)
            ):
                follow_vehicle = vehicle
        # safety distane from following vehicle
        if follow_vehicle:
            acc_follow = self.IDM(
                follow_vehicle.vel, self.ego.vel, abs(self.ego.scf - follow_vehicle.scf)
            )
            if acc_follow < 0:
                return
        if lead_vehicle:
            # distance after 1 second (ego vehicle enter the other lane)
            lead_scf = lead_vehicle.scf + lead_vehicle.vel * 1.0
            ego_scf = straight_status[frequency - 1][0]
            acc_calc = self.IDM(self.ego.vel, lead_vehicle.vel, abs(lead_scf - ego_scf))
        else:
            acc_calc = self.IDM(self.ego.vel, -1, -1)
        right_status = []
        timeGap = 1 / frequency
        dis = np.hypot(
            cur_lane.centerLine[0][0] - right_lane.centerLine[0][0],
            cur_lane.centerLine[0][1] - right_lane.centerLine[0][1],
        )
        # calculate tcf
        # turing period is 2.0 second
        # using sin function to smooth the lateral motion
        delta_tcf = [
            np.sin(p / (2.0 + timeGap) * np.pi)
            for p in np.arange(timeGap, 2.0 + timeGap, timeGap)
        ]
        delta_tcf = delta_tcf / sum(delta_tcf) * dis

        for t in np.arange(timeGap, self.ego.PLAN_LEN + timeGap, timeGap):
            index = round(t / timeGap)
            if t <= 1.0:
                scf = straight_status[index - 1][0]
                vel = straight_status[index - 1][3]
                acc = straight_status[index - 1][4]
            else:
                acc = min(acc_calc, self.IDM(vel, -1, -1))
                if vel + acc * timeGap >= 0:
                    scf += vel * timeGap + 0.5 * acc * timeGap**2
                    vel += acc * timeGap
                else:
                    scf += (vel + 0) / 2
                    vel = 0
            if t <= 2.0:
                tcf = straight_status[index - 1][1] - dis / 2.0 * t
            else:
                tcf = right_status[-1][1]
            if t == timeGap:
                dscf = scf - self.ego.scf
                dtcf = delta_tcf[index - 1]
                tcf = self.ego.tcf + dtcf
                yaw = np.arctan2(dtcf, dscf)
            elif t <= 2.0:
                dscf = scf - right_status[-1][0]
                dtcf = delta_tcf[index - 1]
                tcf = right_status[-1][1] + dtcf
                yaw = np.arctan2(dtcf, dscf)
                # yaw cannot be larger than 45 degrees
                if dscf < dtcf:
                    return
            else:
                tcf = right_status[-1][1]
                yaw = 0.0
            # cannot turn left when the vehicle arrive at the end of the lane
            if scf >= cur_lane.length:
                return
            right_status.append([scf, tcf, yaw, vel, acc])
        # using sin function to smooth the lateral motion
        if right_status[-1][0] < straight_status[-1][0] * 1.2:
            return
        return right_status

    def staightGo(self, frequency: int) -> list:
        """
        This function simulates a vehicle's straight motion based on its current state and the IDM model.

        Parameters:
            frequency: The frequency of the simulation or time steps per second.

        Returns:
            list: A list of lists, where each sublist contains the vehicle's state at a given time step.
            The state is represented as [scf, tcf, yaw, vel, acc], where scf is the longitudinal position, tcf is the lateral position, yaw is the yaw angle, vel is the velocity, and acc is the acceleration.
        """
        # speed_sign = self.get_speed_sign()
        # determine the lead vehicle
        lead_vehicle = None
        for vehicle_id in self.surcar:
            vehicle = self.surcar[vehicle_id]
            if (
                vehicle.scf > self.ego.scf
                and vehicle.laneId == self.ego.laneId
                and (not lead_vehicle or vehicle.scf < lead_vehicle.scf)
            ):
                lead_vehicle = vehicle
        if lead_vehicle:
            acc_calc = self.IDM(
                self.ego.vel, lead_vehicle.vel, abs(lead_vehicle.scf - self.ego.scf)
            )
        else:
            acc_calc = self.IDM(self.ego.vel, -1, -1)
        straight_status = []
        vel = self.ego.vel
        scf = self.ego.scf
        timeGap = 1 / frequency
        for t in np.arange(timeGap, self.ego.PLAN_LEN + timeGap, timeGap):
            acc = min(acc_calc, self.IDM(vel, -1, -1))
            if vel + acc * timeGap >= 0:
                scf += vel * timeGap + 0.5 * acc * timeGap**2
                vel += acc * timeGap
            else:
                scf += (vel + 0) / 2
                vel = 0
            straight_status.append([scf, self.ego.tcf, self.ego.yaw, vel, acc])
        return straight_status

    def planning(self, vehicle_running, frequency=10):
        """
        This function generates a plan for the ego vehicle based on the surrounding cars and the current lane.

        Parameters:
            vehicle_running: Vehisles that are currently running in the simulations.
            frequency (int): The frequency of the simulation or time steps per second. Defaults to 10.

        Returns:
            list: A list of lists, where each sublist contains the vehicle's state at a given time step.
            The state is represented as [scf, tcf, yaw, vel, acc], where scf is the longitudinal position, tcf is the lateral position, yaw is the yaw angle, vel is the velocity, and acc is the acceleration.
        """
        self.surcar = self.getSurCar(vehicle_running)
        straight_status = self.staightGo(frequency)
        left_status = self.turnLeft(straight_status, frequency)
        right_status = self.turnRight(straight_status, frequency)
        if left_status and right_status:
            if left_status[-1][0] > right_status[-1][0]:
                return left_status
            else:
                return right_status
        elif left_status:
            self.ego.laneId = self.nb.lanes[self.ego.laneId].left.id
            return left_status
        elif right_status:
            self.ego.laneId = self.nb.lanes[self.ego.laneId].right.id
            return right_status
        else:
            return straight_status
