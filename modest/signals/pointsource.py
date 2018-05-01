import numpy as _np
from scipy.stats import multivariate_normal as _mvn
from . import signalsource


class PointSource(signalsource.SignalSource):
    def __init__(
            self,
            RA,
            DEC,
            attitudeStateName='attitude'
    ):
        signalsource.SignalSource.__init__(self)
        self.__RA__ = RA
        self.__DEC__ = DEC
        self.__RaDec__ = {'RA': RA, 'DEC': DEC}
        self.attitudeStateName = attitudeStateName

        return

    def RaDec(self):
        return(self.__RaDec__)

    def unitVec(
            self,
            RaDec=None):
        if RaDec is None:
            RaDec = self.__RaDec__
        cosD = _np.cos(RaDec['DEC'])
        sinD = _np.sin(RaDec['DEC'])
        cosRA = _np.cos(RaDec['RA'])
        sinRA = _np.sin(RaDec['RA'])

        return _np.array([cosD * cosRA, cosD * sinRA, sinD])
    
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
                uniformProbability = 1/(4 * _np.pi)
                maxProb = 1/_np.sqrt(_np.linalg.det(2 * _np.pi * residualVariance))
                if maxProb < uniformProbability:
                    print("using uniform probability")
                    probability = uniformProbability
                else:
                    probability = _mvn.pdf(dY, cov=residualVariance)

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
