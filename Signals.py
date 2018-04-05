import numpy as np
from scipy.stats import multivariate_normal

class SignalSource():
    nextSignalID = 0
    
    def __init__(
            self,
    ):
        self.__signalID__ = SignalSource.nextSignalID
        SignalSource.nextSignalID += 1
        return

    def signalID(self):
        return self.__signalID__

    
class PointSource(SignalSource):
    def __init__(
            self,
            RA,
            DEC,
            attitudeStateName='attitude'
    ):
        SignalSource.__init__(self)
        self.__RA__ = RA
        self.__DEC__ = DEC
        self.__RaDec__ = {'RA': RA, 'DEC': DEC}
        self.attitudeStateName = attitudeStateName

        return

    def RaDec(self):
        return(self.__RaDec__)

    def computeAssociationProbability(
            self,
            measurement,
            stateDict,
            validationThreshold=0
    ):

        if (
                ('RA' in measurement) and
                ('DEC' in measurement) and
                ('attitude' in stateDict)
        ):
            attitudeState = stateDict[self.attitudeStateName]['stateObject']

            measurementMatrices = attitudeState.getMeasurementMatrices(
                measurement,
                source=self
            )
            P = attitudeState.covariance()

            H = measurementMatrices['H']['unitVector']

            R = measurementMatrices['R']['unitVector']

            dY = measurementMatrices['dY']['unitVector']
            
            residualVariance = H.dot(P).dot(H.transpose()) + R
            try:
                uniformProbability = 1/(4 * np.pi)
                maxProb = 1/np.sqrt(np.linalg.det(2 * np.pi * residualVariance))
                if maxProb < uniformProbability:
                    print("using uniform probability")
                    probability = uniformProbability
                else:
                    probability = multivariate_normal.pdf(dY, cov=residualVariance)

            except:
                print('P:')
                print(P)
                print('H:')
                print(H)
                print('R:')
                print(R)
                print('S:')                
                print(residualVariance)
                print('dY:')
                print(dY)
                raise ValueError(
                    'Error computing probability.'
                    )
            # print('')
        else:
            probability=0

        return(probability)


class PoissonSource(SignalSource):
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


class StaticXRayPointSource(PointSource, StaticPoissonSource):

    def __init__(
            self,
            RA,
            DEC,
            peakPhotonFlux,
            attitudeStateName='attitude'
    ):

        PointSource.__init__(self, RA, DEC, attitudeStateName)
        StaticPoissonSource.__init__(self, peakPhotonFlux)
        self.peakPhotonFlux = peakPhotonFlux

        return

    def computeAssociationProbability(
            self,
            measurement,
            stateDict,
            validationThreshold=0
    ):
        anglePR = PointSource.computeAssociationProbability(
            self,
            measurement,
            stateDict,
            validationThreshold
        )

        poisPR = StaticPoissonSource.computeAssociationProbability(
            self,
            measurement
            )

        return (anglePR * poisPR * self.peakPhotonFlux)


class UniformNoiseXRaySource(StaticPoissonSource):
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
