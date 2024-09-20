import numpy as np
from scipy.special import fresnel

class EulerSpiral(object):

    def __init__(self, gamma):
        self._gamma = gamma

    @staticmethod
    def createFromLengthAndCurvature(length, curvStart, curvEnd):
        return EulerSpiral(1 * (curvEnd - curvStart) / length)

    def calc(self, s, x0=0, y0=0, kappa0=0, theta0=0):

        C0 = x0 + 1j * y0

        if self._gamma == 0 and kappa0 == 0:
            # Straight line
            Cs = C0 + np.exp(1j * theta0) * s

        elif self._gamma == 0 and kappa0 != 0:
            # Arc
            Cs = C0 + np.exp(1j * theta0) / kappa0 * (np.sin(kappa0 * s) + 1j * (1 - np.cos(kappa0 * s)))

        else:
            #fresnel integrals
            Sa, Ca = fresnel((kappa0 + self._gamma * s) / np.sqrt(np.pi * np.abs(self._gamma)))
            Sb, Cb = fresnel(kappa0 / np.sqrt(np.pi * np.abs(self._gamma)))

            # Euler Spiral
            Cs1 = np.sqrt(np.pi / np.abs(self._gamma)) * np.exp(1j * (theta0 - kappa0**2 / 2 / self._gamma))
            Cs2 = np.sign(self._gamma) * (Ca - Cb) + 1j * Sa - 1j * Sb

            Cs = C0 + Cs1 * Cs2

        #Tangent at each point
        theta = self._gamma * s**2 / 2 + kappa0 * s + theta0

        return (Cs.real, Cs.imag, theta)