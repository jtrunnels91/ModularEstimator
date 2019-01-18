import numpy as np

# Simple library of functions for computing x-ray background flux
#
# All energies should be passed in keV!!!
#
# Everything in this library is based on the following paper:
#
# The Spectrum of Diffuse Cosmic Hard X-Rays Measured with HEAO 1
#
# D.E. Gruber, J.L. Matteson, L.E. Peterson


def xRayBackground(E):
    if (type(E) is np.ndarray) or (type(E) is list):
        results = np.zeros(len(E))
        for i in range(len(results)):
            results[i] = xRayBackground(E[i])

        return results
    else:
        return(singleXRayBackground(E))


def singleXRayBackground(E):
    background = None
    if E < 1:
        raise ValueError(
            'Photon energies less than 2 keV are below the regime of this equation')
    elif (E >= 1 and E <= 60):
        background = (
            7.877 * np.power(E, -0.29) * np.exp(-E / 41.13)
        )
    elif E > 60:
        background = (
            0.0259 * np.power(E / 60, -5.5) +
            0.504 * np.power(E / 60, -1.58) +
            0.0288 * np.power(E / 60, -1.05)
        )

    return background


# Perform an euler numerical integration to get the total background count
# rates over an energy range
def backgroundFluxPerSR(lowerE,
                        upperE,
                        resolution=None):

    energyBand = upperE - lowerE
    if resolution is None:
        nSteps = 100
        resolution = energyBand / (nSteps - 1)
    else:
        nSteps = (energyBand / resolution) + 1

    energySpectrum = np.linspace(lowerE, upperE, nSteps)

    backgroundFlux = 0
    for i in range(len(energySpectrum)):
        backgroundFlux = (
            backgroundFlux +
            xRayBackground(energySpectrum[i]) * resolution
        )
    return backgroundFlux


def radianFOVToSR(radian):
    return 2 * np.pi * (1 - np.cos(radian))


def degreeFOVToSR(degree):
    radian = degree * np.pi / (180)
    return radianFOVToSR(radian)


def KEVbackgroundFlux(lowerE,
                      upperE,
                      FOVDegrees):
    return backgroundFluxPerSR(lowerE, upperE) * degreeFOVToSR(FOVDegrees)


def ERGbackgroundFlux(lowerE,
                      upperE,
                      FOVDegrees):
    return (1.60218e-9 *
            backgroundFluxPerSR(lowerE, upperE) *
            degreeFOVToSR(FOVDegrees))


def chandraBackgroundERG(FOVDegrees):
    return 3.8e-12 * np.square(FOVDegrees*2)
