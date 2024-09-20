import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon


# Define boundaries
boundaries = {
    "max_acc": 2.40,  # [m/s^2]
    "min_acc": -4.05,  # [m/s^2]
    "max_abs_jerk": 8.37,  # [m/s^3],
    "max_abs_yaw_rate": 0.95,  # [rad/s]
    "max_abs_yaw_acc": 1.93,  # [rad/s^2]
}

score_weight = {
    "ttc": 5,
    "c": 2,
    "ep": 5,
}

safe_ttc = 3.0


class ScoreCalculator:
    def __init__(self, ego, vehs, netInfo, frequency):
        self.ego = ego
        self.vehs = vehs
        self.netInfo = netInfo
        self.timeGap = 1 / frequency
        self.pastLen = len(self.ego.xQ)
        self.planLen = len(self.ego.planXQ)

    def create_rectangle(self, center_x, center_y, width, length, hdg):
        """Create a rectangle polygon."""
        cos_yaw = np.cos(hdg)
        sin_yaw = np.sin(hdg)
        x_offs = [length / 2, length / 2, -length / 2, -length / 2]
        y_offs = [width / 2, -width / 2, -width / 2, width / 2]
        x_pts = [
            center_x + x_off * cos_yaw - y_off * sin_yaw
            for x_off, y_off in zip(x_offs, y_offs)
        ]
        y_pts = [
            center_y + x_off * sin_yaw + y_off * cos_yaw
            for x_off, y_off in zip(x_offs, y_offs)
        ]
        return ShapelyPolygon(zip(x_pts, y_pts))

    def _calculate_no_collision(self):
        """
        Calculates the collision score for the ego vehicle in both the past and planned trajectories.

        The function checks for collisions between the ego vehicle and other vehicles in the scene.
        It returns two lists of collision scores, one for the past trajectory and one for the planned trajectory.

        The collision score is 1.0 if no collision is detected, and 0.0 if a collision is detected.

        """

        def collision(t, phase):
            for veh in self.vehs:
                if veh != self.ego:
                    if phase == "past":
                        # align the time
                        tt = t - (len(self.ego.xQ) - len(veh.xQ))
                        if tt >= 0:
                            if (
                                np.hypot(
                                    self.ego.xQ[t] - veh.xQ[tt],
                                    self.ego.yQ[t] - veh.yQ[tt],
                                )
                                > 8.0
                            ):
                                continue
                            ego_poly = self.create_rectangle(
                                self.ego.xQ[t],
                                self.ego.yQ[t],
                                self.ego.width,
                                self.ego.length,
                                self.ego.hdgQ[t],
                            )
                            obs_poly = self.create_rectangle(
                                veh.xQ[tt],
                                veh.yQ[tt],
                                veh.width,
                                veh.length,
                                veh.hdgQ[tt],
                            )
                            if ego_poly.intersects(obs_poly):
                                return 0.0  # Collision detected
                    else:
                        if t < len(veh.planXQ):
                            if (
                                np.hypot(
                                    self.ego.planXQ[t] - veh.planXQ[t],
                                    self.ego.planYQ[t] - veh.planYQ[t],
                                )
                                > 8.0
                            ):
                                continue
                            ego_poly = self.create_rectangle(
                                self.ego.planXQ[t],
                                self.ego.planYQ[t],
                                self.ego.width,
                                self.ego.length,
                                self.ego.planHdgQ[t],
                            )
                            obs_poly = self.create_rectangle(
                                veh.planXQ[t],
                                veh.planYQ[t],
                                veh.width,
                                veh.length,
                                veh.planHdgQ[t],
                            )
                            if ego_poly.intersects(obs_poly):
                                return 0.0  # Collision detected
            return 1.0

        collision_score_past = [collision(t, "past") for t in range(self.pastLen)]
        collision_score_plan = [collision(t, "plan") for t in range(self.planLen)]
        return collision_score_past, collision_score_plan

    def _calculate_drivable_area_compliance(self):
        """
        Calculates the drivable area compliance of the ego vehicle.

        This function evaluates the ego vehicle's position and orientation at each time step
        in the past and planned trajectories, and checks if it is within the drivable area
        of the road. The drivable area is defined as the area of the road that the vehicle
        can safely drive on.

        The function returns two lists of scores, one for the past trajectory and one for
        the planned trajectory. Each score represents the proportion of the vehicle's area
        that is within the drivable area of the road at each time step.

        """

        def get_road_poly(roadId):
            roadIds = (
                self.netInfo.drivingRoad[roadId]
                if roadId in self.netInfo.drivingRoad
                else [roadId]
            )
            shapes = []
            road_polys = []
            for roadId in roadIds:
                road = self.netInfo.roads[roadId]
                if road.junction:
                    junction = self.netInfo.junctions[road.junction]
                    if junction not in shapes:
                        shapes.append(junction)
                        road_polys.append(ShapelyPolygon(junction.boundary).buffer(0))
                else:
                    if road not in shapes:
                        shapes.append(road)
                        road_polys.append(ShapelyPolygon(road.boundary).buffer(0))

            return road_polys

        def is_inside(t, phase):
            if phase == "past":
                ego_poly = self.create_rectangle(
                    self.ego.xQ[t],
                    self.ego.yQ[t],
                    self.ego.width,
                    self.ego.length,
                    self.ego.hdgQ[t],
                )
                road_polys = get_road_poly(self.ego.roadIdQ[t])
            else:
                ego_poly = self.create_rectangle(
                    self.ego.planXQ[t],
                    self.ego.planYQ[t],
                    self.ego.width,
                    self.ego.length,
                    self.ego.planHdgQ[t],
                )
                road_polys = get_road_poly(self.ego.planRoadIdQ[t])
            intersect_area = sum(
                [poly.intersection(ego_poly).area for poly in road_polys]
            )
            return 1.0 if intersect_area / ego_poly.area > 0.95 else 0.0

        inroad_score_past = [is_inside(t, "past") for t in range(self.pastLen)]
        inroad_score_plan = [is_inside(t, "plan") for t in range(self.planLen)]
        return inroad_score_past, inroad_score_plan

    def _calculate_efficiency(self):
        """
        Calculates the efficiency score of the ego vehicle in both the past and planned trajectories.

        The efficiency score is calculated as the ratio of the vehicle's velocity to its expected velocity.

        """
        efficiency_score_past = [vel / self.ego.EXPECT_VEL for vel in self.ego.velQ]
        efficiency_score_plan = [vel / self.ego.EXPECT_VEL for vel in self.ego.planVelQ]
        return efficiency_score_past, efficiency_score_plan

    def _calculate_is_comfortable(self):
        """
        Check if all kinematic parameters of a trajectory are within specified boundaries.
        return: 1.0 if all parameters are within boundaries, 0.0 otherwise
        """
        comfortable_score_past = []
        jerkQ = np.diff(self.ego.accQ) / self.timeGap
        yawRateQ = np.diff(self.ego.yawQ) / self.timeGap
        yawAccQ = np.diff(yawRateQ) / self.timeGap
        for t in range(self.pastLen):
            j1 = self.ego.accQ[t] <= boundaries["max_acc"]
            j2 = self.ego.accQ[t] >= boundaries["min_acc"]
            j3, j4, j5 = True, True, True
            if t > 0:
                j3 = jerkQ[t - 1] <= boundaries["max_abs_jerk"]
                j4 = yawRateQ[t - 1] <= boundaries["max_abs_yaw_rate"]
            if t > 1:
                j5 = yawAccQ[t - 2] <= boundaries["max_abs_yaw_acc"]
            comfortable_score_past.append(1.0 if j1 & j2 & j3 & j4 & j5 else 0.0)

        comfortable_score_plan = []
        jerkQ = (
            np.diff(list(self.ego.accQ)[-1:] + list(self.ego.planAccQ)) / self.timeGap
        )
        yawRateQ = (
            np.diff(list(self.ego.yawQ)[-2:] + list(self.ego.planYawQ)) / self.timeGap
        )
        yawAccQ = np.diff(yawRateQ) / self.timeGap
        yawRateQ = yawRateQ[1:]
        for t in range(self.planLen):
            j1 = self.ego.planAccQ[t] <= boundaries["max_acc"]
            j2 = self.ego.planAccQ[t] >= boundaries["min_acc"]
            j3, j4, j5 = True, True, True
            if t > 0:
                j3 = jerkQ[t - 1] <= boundaries["max_abs_jerk"]
                j4 = yawRateQ[t - 1] <= boundaries["max_abs_yaw_rate"]
            if t > 1:
                j5 = yawAccQ[t - 2] <= boundaries["max_abs_yaw_acc"]
            comfortable_score_plan.append(1.0 if j1 & j2 & j3 & j4 & j5 else 0.0)
        return comfortable_score_past, comfortable_score_plan

    def _calculate_time_to_collision(self):
        def interactPoint(p1, d1, p2, d2):
            """
            The collision point of the two vehicles traveling along the speed direction from the current position
            """
            p1 = np.array(p1)
            d1 = np.array(d1)
            p2 = np.array(p2)
            d2 = np.array(d2)

            A = np.array([d1, -d2]).T
            b = p2 - p1

            try:
                t, s = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return None
            if t >= 0 and s >= 0:
                intersection = p1 + t * d1
                return intersection
            else:
                return None

        safe_score_past = [1.0]  # begin from t + 1
        evxQ = np.diff(self.ego.xQ) / self.timeGap
        evyQ = np.diff(self.ego.yQ) / self.timeGap
        for t in range(self.pastLen):
            if t > 0:  # begin from t + 1
                safe_score_past.append(1.0)
                ex, ey, evx, evy = (
                    self.ego.xQ[t],
                    self.ego.yQ[t],
                    evxQ[t - 1],
                    evyQ[t - 1],
                )
                for veh in self.vehs:
                    tt = t - (len(self.ego.xQ) - len(veh.xQ))  # align the time
                    vxQ = np.diff(veh.xQ) / self.timeGap
                    vyQ = np.diff(veh.yQ) / self.timeGap
                    try:
                        x, y, vx, vy = veh.xQ[tt], veh.yQ[tt], vxQ[tt - 1], vyQ[tt - 1]
                        ret = interactPoint([ex, ey], [evx, evy], [x, y], [vx, vy])
                        if ret is not None:
                            ttc = np.hypot(ex - ret[0], ey - ret[1]) / np.hypot(
                                evx, evy
                            )
                            safe_score_past[-1] = min(
                                safe_score_past[-1], round(ttc / self.ego.vQ[t], 1)
                            )
                    except:
                        pass

        safe_score_plan = [1.0]  # begin from t + 1
        evxQ = np.diff(list(self.ego.xQ)[-1:] + list(self.ego.planXQ)) / self.timeGap
        evyQ = np.diff(list(self.ego.yQ)[-1:] + list(self.ego.planYQ)) / self.timeGap
        for t in range(self.planLen):
            if t > 0:  # begin from t + 1
                safe_score_plan.append(1.0)
                ex, ey, evx, evy = (
                    self.ego.planXQ[t],
                    self.ego.planYQ[t],
                    evxQ[t - 1],
                    evyQ[t - 1],
                )
                for veh in self.vehs:
                    vxQ = np.diff(list(veh.xQ)[-1:] + list(veh.planXQ)) / self.timeGap
                    vyQ = np.diff(list(veh.yQ)[-1:] + list(veh.planYQ)) / self.timeGap
                    try:
                        x, y, vx, vy = veh.xQ[t], veh.yQ[t], vxQ[t - 1], vyQ[t - 1]
                        ret = interactPoint([ex, ey], [evx, evy], [x, y], [vx, vy])
                        if ret is not None:
                            ttc = np.hypot(ex - ret[0], ey - ret[1]) / np.hypot(
                                evx, evy
                            )
                            safe_score_plan[-1] = min(
                                safe_score_plan[-1], round(ttc / self.ego.vQ[t], 1)
                            )
                    except:
                        pass
        return safe_score_past, safe_score_plan

    def calculate(self):
        """
        Calculates the driving scores for past and planned trajectories.

        The function combines scores from various metrics, including no collision,
        drivable area compliance, time to collision, comfort, and efficiency. The
        scores are weighted and averaged to produce a final safety score for each
        time step in the past and planned trajectories.

        """
        score_past = []
        score_plan = []
        score_nc = self._calculate_no_collision()
        score_dac = self._calculate_drivable_area_compliance()
        score_ttc = self._calculate_time_to_collision()
        score_c = self._calculate_is_comfortable()
        score_ep = self._calculate_efficiency()
        for t in range(self.pastLen):
            score_past.append(
                round(
                    score_nc[0][t]
                    * score_dac[0][t]
                    * (
                        score_weight["ttc"] * score_ttc[0][t]
                        + score_weight["c"] * score_c[0][t]
                        + score_weight["ep"] * score_ep[0][t]
                    )
                    / (score_weight["ttc"] + score_weight["c"] + score_weight["ep"]),
                    1,
                )
            )
        for t in range(self.planLen):
            score_plan.append(
                round(
                    score_nc[1][t]
                    * score_dac[1][t]
                    * (
                        score_weight["ttc"] * score_ttc[1][t]
                        + score_weight["c"] * score_c[1][t]
                        + score_weight["ep"] * score_ep[1][t]
                    )
                    / (score_weight["ttc"] + score_weight["c"] + score_weight["ep"]),
                    1,
                )
            )
        return score_past, score_plan
