import numpy as np
from scipy.stats import multivariate_normal
from . import pointsource
from . import poissonsource 

class StaticXRayPointSource(pointsource.PointSource, poissonsource.StaticPoissonSource):

    def __init__(
            self,
            RA,
            DEC,
            peakPhotonFlux,
            attitudeStateName='attitude'
    ):

        pointsource.PointSource.__init__(self, RA, DEC, attitudeStateName)
        poissonsource.StaticPoissonSource.__init__(self, peakPhotonFlux)
        self.peakPhotonFlux = peakPhotonFlux

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

        return (anglePR * poisPR * self.peakPhotonFlux)


class UniformNoiseXRaySource(poissonsource.StaticPoissonSource):
    def __init__(
            self,
            photonFlux
    ):
        super().__init__(photonFlux)
        
        self.photonFlux = photonFlux

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
