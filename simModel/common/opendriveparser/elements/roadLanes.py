class Lanes(object):

    def __init__(self):
        self._laneOffsets = []
        self._laneSections = []

    @property
    def laneOffsets(self):
        self._laneOffsets.sort(key=lambda x: x.sPos)
        return self._laneOffsets

    @property
    def laneSections(self):
        self._laneSections.sort(key=lambda x: x.sPos)
        return self._laneSections

    def getLaneSection(self, laneSectionIdx):
        for laneSection in self.laneSections:
            if laneSection.idx == laneSectionIdx:
                return laneSection

        return None

    def getLastLaneSectionIdx(self):
        """Returns the index of the last lane section of the road"""

        numLaneSections = len(self.laneSections)

        if numLaneSections > 1:
            return numLaneSections - 1

        return 0


class LaneOffset(object):

    def __init__(self):
        self._sPos = None
        self._a = None
        self._b = None
        self._c = None
        self._d = None

    @property
    def sPos(self):
        return self._sPos

    @sPos.setter
    def sPos(self, value):
        self._sPos = float(value)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = float(value)

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = float(value)

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        self._c = float(value)

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        self._d = float(value)

    @property
    def coeffs(self):
        """Array of coefficients for usage with numpy.polynomial.polynomial.polyval"""
        return [self._a, self._b, self._c, self._d]


class LaneSection(object):

    def __init__(self):
        self._idx = None
        self._sPos = None
        self._singleSide = None
        self._leftLanes = LeftLanes()
        self._centerLanes = CenterLanes()
        self._rightLanes = RightLanes()
        self.affGridIDs = set()

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, value):
        self._idx = int(value)

    @property
    def sPos(self):
        return self._sPos

    @sPos.setter
    def sPos(self, value):
        self._sPos = float(value)

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = float(value)

    @property
    def singleSide(self):
        return self._singleSide

    @singleSide.setter
    def singleSide(self, value):
        if value not in ["true", "false"] and value is not None:
            raise AttributeError("Value must be true or false.")

        self._singleSide = value == "true"

    @property
    def leftLanes(self):
        """Get list of sorted lanes always starting in the middle (lane id -1)"""
        return self._leftLanes.lanes

    @property
    def centerLanes(self):
        return self._centerLanes.lanes

    @property
    def rightLanes(self):
        """Get list of sorted lanes always starting in the middle (lane id 1)"""
        return self._rightLanes.lanes

    @property
    def allLanes(self):
        """Attention! lanes are not sorted by id"""
        return self._leftLanes.lanes + self._centerLanes.lanes + self._rightLanes.lanes

    def getLane(self, laneId):
        for lane in self.allLanes:
            if lane.id_int == laneId:
                return lane
        return None


class LeftLanes(object):

    sort_direction = False

    def __init__(self):
        self._lanes = []

    @property
    def lanes(self):
        self._lanes.sort(key=lambda x: x.id_int, reverse=self.sort_direction)
        return self._lanes


class CenterLanes(LeftLanes):
    pass


class RightLanes(LeftLanes):
    sort_direction = True


class Lane(object):

    laneTypes = [
        "none",
        "driving",
        "stop",
        "shoulder",
        "biking",
        "sidewalk",
        "border",
        "restricted",
        "parking",
        "bidirectional",
        "median",
        "special1",
        "special2",
        "special3",
        "roadWorks",
        "tram",
        "rail",
        "entry",
        "exit",
        "offRamp",
        "onRamp",
    ]

    def __init__(self):
        self._id_int = None
        self._type = None
        self._level = None
        self._link = LaneLink()
        self._widths = []
        self._borders = []
        self._leftBorders = None
        self._rightBorders = None
        self.centerLine = None
        self.left = None
        self.right = None
        self.next = []
        self.last = []
        self.direction = None
        self.id = None
        self.roadId = None
        self.laneSection_idx = None
        self.length = None

    @property
    def id_int(self):
        return self._id_int

    @id_int.setter
    def id_int(self, value):
        self._id_int = int(value)

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        if value not in self.laneTypes:
            raise Exception()

        self._type = str(value)

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        if value not in ["true", "false"] and value is not None:
            raise AttributeError("Value must be true or false.")

        self._level = value == "true"

    @property
    def link(self):
        return self._link

    @property
    def widths(self):
        self._widths.sort(key=lambda x: x.sOffset)
        return self._widths

    def getWidth(self, widthIdx):
        for width in self._widths:
            if width.idx == widthIdx:
                return width

        return None

    def getLastLaneWidthIdx(self):
        """Returns the index of the last width sector of the lane"""

        numWidths = len(self._widths)

        if numWidths > 1:
            return numWidths - 1

        return 0

    def setBorders(self, value, direction):
        if direction == "left":
            self._leftBorders = value
        elif direction == "right":
            self._rightBorders = value

    @property
    def leftBorders(self):
        return self._leftBorders

    @property
    def rightBorders(self):
        return self._rightBorders

    @property
    def borders(self):
        return self._borders


class LaneLink(object):

    def __init__(self):
        self._predecessor = None
        self._successor = None

    @property
    def predecessorId(self):
        return self._predecessor

    @predecessorId.setter
    def predecessorId(self, value):
        self._predecessor = int(value)

    @property
    def successorId(self):
        return self._successor

    @successorId.setter
    def successorId(self, value):
        self._successor = int(value)


class LaneWidth(object):

    def __init__(self):
        self._idx = None
        self._sOffset = None
        self._a = None
        self._b = None
        self._c = None
        self._d = None

    @property
    def idx(self):
        return self._idx

    @idx.setter
    def idx(self, value):
        self._idx = int(value)

    @property
    def sOffset(self):
        return self._sOffset

    @sOffset.setter
    def sOffset(self, value):
        self._sOffset = float(value)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = float(value)

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = float(value)

    @property
    def c(self):
        return self._c

    @c.setter
    def c(self, value):
        self._c = float(value)

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, value):
        self._d = float(value)

    @property
    def coeffs(self):
        """Array of coefficients for usage with numpy.polynomial.polynomial.polyval"""
        return [self._a, self._b, self._c, self._d]


class LaneBorder(LaneWidth):
    pass
