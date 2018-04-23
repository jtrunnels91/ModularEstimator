import numpy as np
from scipy.stats import multivariate_normal
from . import signalsource


class PoissonSource(signalsource.SignalSource):
    def __init__(
            self,
            flux
            ):
        super().__init__()
        self.lastTime = 0
        self.flux = flux
        return

    def computeAssociationProbability(
            self,
            currentFlux,
            measurement
            ):
        time = measurement['time']
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

    
class PeriodicPoissonSource(PoissonSource):
    def __init__(
            self,
            peakFlux,
            fluxProfile,
            nonPulsedFlux
    ):
        self.peakFlux = peakFlux
        self.fluxProfile = fluxProfile
        self.nonPulsedFlux = nonPulsedFlux
        return
