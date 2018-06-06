## @file uniformnoisexraysource.py
# @brief This file contains the UniformNoiseXRaySource class

import numpy as np

from . import poissonsource
from .. utils import spacegeometry as sg
from .. utils import xrayphotons as xp
from .. utils import physicalconstants as pc


class UniformNoiseXRaySource(poissonsource.StaticPoissonSource):
    def __init__(
            self,
            photonFlux=None,
            energyRangeKeV=[2,10],
            detectorFOV=180,
            detectorArea=1
    ):
        if photonFlux is not None:
            self.photonFlux = photonFlux
            self.FOV = None
            self.detectorArea = None
        else:
            photonsPerSqCm = xp.ERGbackgroundFlux(
                energyRangeKeV[0],
                energyRangeKeV[1],
                detectorFOV # function expects FOV in degrees
            ) * pc.electronVoltPerErg/pc.electronVoltPerPhoton
            self.photonFlux = photonsPerSqCm * detectorArea
            self.FOV = detectorFOV
            self.detectorArea = detectorArea
            
        super().__init__(self.photonFlux)
        
        return

    def computeAssociationProbability(
            self,
            measurement,
            stateDict,
            validationThreshold=0):

        anglePR = 1/(4 * np.pi)

        poisPR = super().computeAssociationProbability(measurement)
        
        totalPR = anglePR * poisPR * self.photonFlux

        return totalPR
    
    def generatePhotonArrivals(
            self,
            tMax,
            t0=0,
            position=None,
            attitude=None,
            FOV=None
            ):
        poissonEvents = self.generateEvents(tMax, t0=t0)
        arrivalVectors = self.generateUniformArrivalVectors(len(poissonEvents), FOV)

        photonMeasurements = []
        for photonIndex in range(len(poissonEvents)):
            arrivalVector = arrivalVectors[photonIndex]
            arrivalTime = poissonEvents[photonIndex]
            
            # Generate a uniformly distributed random arrival vector
            Ra, Dec = sg.unitVector2RaDec(arrivalVector)
            measurementDict = {
                't': {'value': arrivalTime},
                'unitVec': {'value': arrivalVector},
                'RA': {'value': Ra},
                'DEC': {'value': Dec},
                'name': 'background'
            }

            photonMeasurements.append(measurementDict)
        
        return photonMeasurements
    
    def generateUniformArrivalVectors(
            self,
            nVectors,
            FOV=None
    ):
        if FOV is None:
            if self.FOV is None:
                FOV = np.pi
            else:
                FOV = self.FOV * np.pi/180.0
        
        theta = np.random.uniform(low=np.cos(FOV), high=1.0, size=nVectors)
        phi = np.random.uniform(low=0, high=np.pi * 2, size=nVectors)

        oneMinusThetaSquare = np.sqrt(1 - np.square(theta))
        cosPhi = np.cos(phi)
        sinPhi = np.sin(phi)

        v = np.array(
            [
                theta,
                oneMinusThetaSquare * cosPhi,
                oneMinusThetaSquare * sinPhi
            ]
        )

        return np.transpose(v)
