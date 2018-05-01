import numpy as np
# from scipy.stats import multivariate_normal
from . import signalsource
from abc import ABCMeta, abstractmethod


class PoissonSource(signalsource.SignalSource):
    def __init__(
            self,
            flux
            ):
        signalsource.SignalSource.__init__(self)
        self.lastTime = 0
        self.flux = flux
        return

    def computeAssociationProbability(
            self,
            currentFlux,
            measurement
            ):
        time = measurement['t']['value']
        dT = time - self.lastTime
        self.lastTime = time
        return np.exp(-self.flux * dT) * currentFlux


class StaticPoissonSource(PoissonSource):
    def __init__(
            self,
            flux
            ):
        super().__init__(flux)

    def computeAssociationProbability(
            self,
            measurement
            ):
        poissonProb = super().computeAssociationProbability(
            self.flux,
            measurement
        )
        return(poissonProb)


class DynamicPoissonSource(PoissonSource):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            maxFlux,
            correlationStateName='correlation'
    ):
        self.correlationStateName = correlationStateName
        PoissonSource.__init__(self, maxFlux)
        return

    @abstractmethod
    def getSignal(
            self,
            t,
            tVar=None,
            state=None
    ):
        raise NotImplementedError(
            "The getSignal method is not implemented in " +
            "DynamicPoissonSource, and must be overridden."
        )

    def computeAssociationProbability(
            self,
            measurement,
            state
            ):
        
        currentFlux = self.getSignal(
            measurement['t']['value'],
            tVar=measurement['t']['var'],
            state=self.correlationStateName
        )
        print(currentFlux)

        poissonProb = super().computeAssociationProbability(
            currentFlux,
            measurement
        )

        print(poissonProb)
        return(poissonProb)
