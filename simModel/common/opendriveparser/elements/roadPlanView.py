import abc
import numpy as np
import math
from ..elements.eulerspiral import EulerSpiral


class PlanView(object):

    def __init__(self):
        self._geometries = []

    def addLine(self, startPosition, heading, length):
        self._geometries.append(Line(startPosition, heading, length))

    def addSpiral(self, startPosition, heading, length, curvStart, curvEnd):
        self._geometries.append(
            Spiral(startPosition, heading, length, curvStart, curvEnd)
        )

    def addArc(self, startPosition, heading, length, curvature):
        self._geometries.append(Arc(startPosition, heading, length, curvature))

    def addParamPoly3(
        self, startPosition, heading, length, aU, bU, cU, dU, aV, bV, cV, dV, pRange
    ):
        self._geometries.append(
            ParamPoly3(
                startPosition, heading, length, aU, bU, cU, dU, aV, bV, cV, dV, pRange
            )
        )

    def addBezier(self, controlPoints):
        self._geometries.append(
            Bezier(
                controlPoints,
            )
        )

    @property
    def geometries(self):
        return self._geometries

    def getLength(self):
        """Get length of whole plan view"""

        length = 0

        for geometry in self._geometries:
            length += geometry.getLength()

        return length

    def calc(self, sPos):
        """Calculate position and tangent at sPos"""

        for geometry in self._geometries:
            if geometry.getLength() < sPos and not np.isclose(
                geometry.getLength(), sPos
            ):
                sPos -= geometry.getLength()
                continue
            return geometry.calcPosition(sPos)

        raise Exception(
            "Tried to calculate a position outside of the borders of the trajectory by s="
            + str(sPos)
        )


class Geometry(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getGeoType(self):
        """Returns the type of geometry"""
        return

    @abc.abstractmethod
    def getStartPosition(self):
        """Returns the overall geometry length"""
        return

    @abc.abstractmethod
    def getLength(self):
        """Returns the overall geometry length"""
        return

    @abc.abstractmethod
    def calcPosition(self, s):
        """Calculates the position of the geometry as if the starting point is (0/0)"""
        return


class Line(Geometry):

    def __init__(self, startPosition, heading, length):
        self.startPosition = np.array(startPosition)
        self.heading = heading
        self.length = length
        self.geoType = "Line"

    def getGeoType(self):
        return self.geoType

    def getStartPosition(self):
        return self.startPosition

    def getLength(self):
        return self.length

    def calcPosition(self, s):
        pos = self.startPosition + np.array(
            [s * np.cos(self.heading), s * np.sin(self.heading)]
        )
        hdg = self.heading

        return (pos, hdg)


class Arc(Geometry):

    def __init__(self, startPosition, heading, length, curvature):
        self.startPosition = np.array(startPosition)
        self.heading = heading
        self.length = length
        self.curvature = curvature
        self.geoType = "Arc"

    def getGeoType(self):
        return self.geoType

    def getStartPosition(self):
        return self.startPosition

    def getLength(self):
        return self.length

    def calcPosition(self, s):
        c = self.curvature
        hdg = self.heading - np.pi / 2

        a = 2 / c * np.sin(s * c / 2)
        alpha = (np.pi - s * c) / 2 - hdg

        dx = -1 * a * np.cos(alpha)
        dy = a * np.sin(alpha)

        pos = self.startPosition + np.array([dx, dy])
        hdg = self.heading + s * self.curvature

        return (pos, hdg)

    def getCurv(self):
        return self.curvature


class Spiral(Geometry):

    def __init__(self, startPosition, heading, length, curvStart, curvEnd):
        self._startPosition = np.array(startPosition)
        self._heading = heading
        self._length = length
        self._curvStart = curvStart
        self._curvEnd = curvEnd

        self._spiral = EulerSpiral.createFromLengthAndCurvature(
            length, curvStart, curvEnd
        )
        self.geoType = "Spiral"

    def getGeoType(self):
        return self.geoType

    def getStartPosition(self):
        return self._startPosition

    def getLength(self):
        return self._length

    def calcPosition(self, s):
        (x, y, t) = self._spiral.calc(
            s,
            self._startPosition[0],
            self._startPosition[1],
            self._curvStart,
            self._heading,
        )
        return (np.array([x, y]), t)


class Poly3(Geometry):

    def __init__(self, startPosition, heading, length, a, b, c, d):
        self._startPosition = np.array(startPosition)
        self._heading = heading
        self._length = length
        self._a = a
        self._b = b
        self._c = c
        self._d = d
        self.geoType = "Poly3"

        raise NotImplementedError()

    def getGeoType(self):
        return self.geoType

    def getStartPosition(self):
        return self._startPosition

    def getLength(self):
        return self._length

    def calcPosition(self, s):
        # TODO untested

        # Calculate new point in s/t coordinate system
        coeffs = [self._a, self._b, self._c, self._d]

        t = np.polynomial.polynomial.polyval(s, coeffs)

        # Rotate and translate
        srot = s * np.cos(self._heading) - t * np.sin(self._heading)
        trot = s * np.sin(self._heading) + t * np.cos(self._heading)

        # Derivate to get heading change
        dCoeffs = coeffs[1:] * np.array(np.arange(1, len(coeffs)))
        tangent = np.polynomial.polynomial.polyval(s, dCoeffs)

        return (self._startPosition + np.array([srot, trot]), self._heading + tangent)


class ParamPoly3(Geometry):

    def __init__(
        self, startPosition, heading, length, aU, bU, cU, dU, aV, bV, cV, dV, pRange
    ):
        self._startPosition = np.array(startPosition)
        self._heading = heading
        self._length = length

        self._aU = aU
        self._bU = bU
        self._cU = cU
        self._dU = dU
        self._aV = aV
        self._bV = bV
        self._cV = cV
        self._dV = dV
        self.geoType = "ParamPoly3"

        if pRange is None:
            self._pRange = 1.0
        else:
            self._pRange = pRange

    def getGeoType(self):
        return self.geoType

    def getStartPosition(self):
        return self._startPosition

    def getLength(self):
        return self._length

    def calcPosition(self, s):

        # Position
        pos = (s / self._length) * self._pRange

        coeffsU = [self._aU, self._bU, self._cU, self._dU]
        coeffsV = [self._aV, self._bV, self._cV, self._dV]

        x = np.polynomial.polynomial.polyval(pos, coeffsU)
        y = np.polynomial.polynomial.polyval(pos, coeffsV)

        xrot = x * np.cos(self._heading) - y * np.sin(self._heading)
        yrot = x * np.sin(self._heading) + y * np.cos(self._heading)

        # Tangent is defined by derivation
        dCoeffsU = coeffsU[1:] * np.array(np.arange(1, len(coeffsU)))
        dCoeffsV = coeffsV[1:] * np.array(np.arange(1, len(coeffsV)))

        dx = np.polynomial.polynomial.polyval(pos, dCoeffsU)
        dy = np.polynomial.polynomial.polyval(pos, dCoeffsV)

        tangent = np.arctan2(dy, dx)

        return (self._startPosition + np.array([xrot, yrot]), self._heading + tangent)


class Bezier(Geometry):
    def __init__(self, controlPoints):
        self._startPosition = np.array(controlPoints[0])
        self._controlPoints = controlPoints
        self._tRange = 1.0
        self.geoType = "Bezier"
        self.setLength()

    def getGeoType(self):
        return self.geoType

    def getStartPosition(self):
        return self._startPosition

    def getLength(self):
        return self._length

    def setLength(self):
        points = []
        for t in np.arange(0, 1 + 1 / 100, 1 / 100):
            points.append(self.getPos(t))
        length = 0
        for i in range(len(points) - 1):
            length += np.hypot(
                points[i][0] - points[i + 1][0], points[i][1] - points[i + 1][1]
            )
        self._length = length

    def getPos(self, t):
        def B(n, i, t):
            return (
                math.factorial(n)
                / (math.factorial(i) * math.factorial(n - i))
                * t**i
                * (1 - t) ** (n - i)
            )

        n = len(self._controlPoints) - 1
        x, y = 0, 0
        for i in range(n + 1):
            x += B(n, i, t) * self._controlPoints[i][0]
            y += B(n, i, t) * self._controlPoints[i][1]
        return x, y

    def calcPosition(self, s):
        # Position
        t = (s / self._length) * self._tRange
        t = min(t, 1.0)
        x, y = self.getPos(t)
        t_near = t - 0.1 if t > 0.1 else t + 0.1
        t_min, t_max = min(t, t_near), max(t, t_near)
        x_min, y_min = self.getPos(t_min)
        x_max, y_max = self.getPos(t_max)
        tangent = np.arctan2(y_max - y_min, x_max - x_min)
        return np.array([x, y]), tangent
