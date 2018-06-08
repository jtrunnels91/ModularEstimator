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

    def generateEvents(
            self,
            tMax,
            t0=0
            ):
        nCandidates = np.int((tMax - t0) * self.flux * 1.1)

        print(nCandidates)
        # Generate a batch of candidate arrival times (more efficient than generating on the fly)
        arrivalTimeArray = np.random.exponential(1.0/self.flux, nCandidates)

        poissonEvents = []

        tLastEvent = t0
        eventIndex = 0
        
        while tLastEvent < tMax:
            # Draw the next arrival time and selection variable from our
            # pre-generated arrays
            if eventIndex < len(arrivalTimeArray):
                nextEvent = arrivalTimeArray[eventIndex]
            # If we run out, generate more on the fly
            else:
                nextEvent = np.random.exponential(1.0/self.flux)
                # print('Generating on the fly!')
            tNextEvent = (
                tLastEvent +
                nextEvent
                )
            if tNextEvent < tMax:
                poissonEvents.append(tNextEvent)
            tLastEvent = tNextEvent
            eventIndex = eventIndex+1
        return poissonEvents

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
            stateDict
            ):

        state = None

        if self.correlationStateName in stateDict:
            state = stateDict[self.correlationStateName]['stateObject']
            state = state.getStateVector()
        #print('Current TDOA std: %.2e' %np.sqrt(state['TDOAVar']))
        if 'TDOAVar' in state:
            measuredTVar = measurement['t']['var'] + state['TDOAVar']
        else:
            measuredTVar = measurement['t']['var']
            
        currentFlux = self.getSignal(
            measurement['t']['value'],
            tVar=measuredTVar,
            #state=state
        )
        #print(currentFlux)

        poissonProb = super().computeAssociationProbability(
            currentFlux,
            measurement
        )

        #print(poissonProb)
        return(poissonProb)
