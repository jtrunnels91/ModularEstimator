## @file staticxraypointsource.py
# @brief This file contains a class which models static xray sources
#
# @details This file contains the StaticXRayPointSource class.

import numpy as np
from math import factorial
import matplotlib.pyplot as plt

from . import pointsource
from . import poissonsource
from .. utils import spacegeometry as sg
from .. utils import xrayphotons as xp
from .. utils import physicalconstants as pc


class StaticXRayPointSource(
        pointsource.PointSource,
        poissonsource.StaticPoissonSource
):

    def __init__(
            self,
            RA,
            DEC,
            photonCountRate=None,
            photonEnergyFlux=None,
            energyRangeKeV=[2, 10],
            detectorArea=1,
            detectorFOV=np.pi,
            attitudeStateName='attitude',
            name=None,
            startTime=0,
            extent=0,
            useUnitVector=True
    ):
        if photonCountRate is None and photonEnergyFlux is None:
            raise ValueError(
                'Must pass either photon count rate or photon energy flux ' +
                'to initialize a point source.'
            )
        if photonCountRate is not None:
            self.FOV = None
            self.detectorArea = None

        else:
            photonsPerSqCm = photonEnergyFlux * pc.electronVoltPerErg/pc.electronVoltPerPhoton
            
            photonCountRate = photonsPerSqCm * detectorArea
            self.FOV = detectorFOV
            self.detectorArea = detectorArea
            
        pointsource.PointSource.__init__(
            self, RA, DEC, extent=extent, attitudeStateName=attitudeStateName, useUnitVector=useUnitVector
        )
        poissonsource.StaticPoissonSource.__init__(self, photonCountRate, startTime=startTime)
        self.peakPhotonFlux = photonCountRate

        if name is None:
            self.name = self.signalID()
        else:
            self.name = name

        return

    def computeAssociationProbability(
            self,
            measurement,
            stateDict,
            validationThreshold=0
    ):
        anglePR = pointsource.PointSource.computeAssociationProbability(
            self,
            measurement,
            stateDict,
            validationThreshold
        )
        poisPR = poissonsource.StaticPoissonSource.computeAssociationProbability(
            self,
            measurement
            )

        #return (anglePR * poisPR * self.flux)
        return anglePR * self.flux
    
    def generatePhotonArrivals(
            self,
            tMax,
            t0=0,
            attitude=None,
            FOV=None,
            AOA_StdDev=None,
            TOA_StdDev=None
    ):
        poissonEvents = self.generateEvents(tMax, t0=t0)
        measurements = []
        for event in poissonEvents:
            if TOA_StdDev:
                event = event + np.random.normal(scale=TOA_StdDev)
                nextMeasurement = {
                    't': {
                        'value': event,
                        'var': np.square(TOA_StdDev)
                    },
                    'name': self.name
                }
            else:
                nextMeasurement = {
                    't': {'value': event},
                    'name': self.name
                }
                
            if attitude is not None:
                nextMeasurement.update(
                    self.generateArrivalVector(attitude(event), AOA_StdDev)
                    )
            measurements.append(nextMeasurement)
        return(measurements)
