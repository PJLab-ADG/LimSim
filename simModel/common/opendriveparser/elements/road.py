from ..elements.roadPlanView import PlanView
from ..elements.roadLink import Link
from ..elements.roadLanes import Lanes


class Road:

    def __init__(self):
        self._id = None
        self._name = None
        self._junction = None
        self.length = None

        self._link = Link()
        self._types = []
        self._planView = PlanView()
        self._lanes = Lanes()
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
    def junction(self):
        return self._junction

    @junction.setter
    def junction(self, value):
        if not isinstance(value, str) and value is not None:
            raise TypeError("Property must be a str or NoneType")

        if value == -1:
            value = None

        self._junction = value

    @property
    def link(self):
        return self._link

    @property
    def types(self):
        return self._types

    @property
    def planView(self):
        return self._planView

    @property
    def lanes(self):
        return self._lanes
