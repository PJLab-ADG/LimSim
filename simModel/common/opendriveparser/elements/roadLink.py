class Link(object):

    def __init__(self):
        self._id = None
        self._predecessor = None
        self._successor = None

    def __str__(self):
        return " > link id " + str(self._id) + " | successor: " + str(self._successor)


    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = int(value)

    @property
    def predecessor(self):
        return self._predecessor

    @predecessor.setter
    def predecessor(self, value):
        if not isinstance(value, Predecessor):
            raise TypeError("Value must be Predecessor")

        self._predecessor = value

    @property
    def successor(self):
        return self._successor

    @successor.setter
    def successor(self, value):
        if not isinstance(value, Successor):
            raise TypeError("Value must be Successor")

        self._successor = value


class Predecessor(object):

    def __init__(self):
        self._elementType = None
        self._elementId = None
        self._contactPoint = None

    def __str__(self):
        return str(self._elementType) + " with id " + str(self._elementId) + " contact at " + str(self._contactPoint)

    @property
    def elementType(self):
        return self._elementType

    @elementType.setter
    def elementType(self, value):
        if value not in ["road", "junction"]:
            raise AttributeError("Value must be road or junction")

        self._elementType = value

    @property
    def elementId(self):
        return self._elementId

    @elementId.setter
    def elementId(self, value):
        self._elementId = int(value)

    @property
    def contactPoint(self):
        return self._contactPoint

    @contactPoint.setter
    def contactPoint(self, value):
        if value not in ["start", "end"] and value is not None:
            raise AttributeError("Value must be start or end")

        self._contactPoint = value

class Successor(Predecessor):
    pass