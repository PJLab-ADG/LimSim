class Junction(object):

    def __init__(self):
        self._id = None
        self._name = None
        self._connections = []
        self.boundary = []

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = str(value)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = str(value)

    @property
    def connections(self):
        return self._connections

    def addConnection(self, connection):
        if not isinstance(connection, Connection):
            raise TypeError("Has to be of instance Connection")

        self._connections.append(connection)


class Connection(object):

    def __init__(self):
        self._id = None
        self._incomingRoad = None
        self._connectingRoad = None
        self._contactPoint = None
        self._leavingRoad = None
        self._contactPointOnLeavingRoad = None
        self._laneLinks = []

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = int(value)

    @property
    def incomingRoad(self):
        return self._incomingRoad

    @incomingRoad.setter
    def incomingRoad(self, value):
        self._incomingRoad = int(value)

    @property
    def connectingRoad(self):
        return self._connectingRoad

    @connectingRoad.setter
    def connectingRoad(self, value):
        self._connectingRoad = int(value)

    @property
    def leavingRoad(self):
        return self._leavingRoad

    @leavingRoad.setter
    def leavingRoad(self, value):
        self._leavingRoad = int(value)

    @property
    def contactPoint(self):
        return self._contactPoint

    @contactPoint.setter
    def contactPoint(self, value):
        if value not in ["start", "end"]:
            raise AttributeError("Contact point can only be start or end.")

        self._contactPoint = value

    @property
    def contactPointOnLeavingRoad(self):
        return self._contactPointOnLeavingRoad

    @contactPointOnLeavingRoad.setter
    def contactPointOnLeavingRoad(self, value):
        if value not in ["start", "end"]:
            raise AttributeError("Contact point can only be start or end.")

        self._contactPointOnLeavingRoad = value

    @property
    def laneLinks(self):
        return self._laneLinks

    def addLaneLink(self, laneLink):
        if not isinstance(laneLink, LaneLink):
            raise TypeError("Has to be of instance LaneLink")

        self._laneLinks.append(laneLink)


class LaneLink(object):

    def __init__(self):
        self._from = None
        self._to = None

    def __str__(self):
        return str(self._from) + " > " + str(self._to)

    @property
    def fromId(self):
        return self._from

    @fromId.setter
    def fromId(self, value):
        self._from = int(value)

    @property
    def toId(self):
        return self._to

    @toId.setter
    def toId(self, value):
        self._to = int(value)
