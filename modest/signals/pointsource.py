import numpy as _np
from scipy.stats import multivariate_normal as _mvn
from . import signalsource
from .. utils import spacegeometry as sg


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
        self.lastPDF = None
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
                (self.attitudeStateName in stateDict)
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
                maxProb = (
                    1 /
                    _np.sqrt(
                        _np.power((2*_np.pi), len(dY)) *
                        _np.linalg.det(residualVariance)
                    )
                )

                maxProb = 1/_np.sqrt(_np.linalg.det(2 * _np.pi * residualVariance))
                if maxProb < uniformProbability:
#                    print("using uniform probability")
                    probability = uniformProbability
                else:
                    # if self.lastPDF:
                    #     if self.lastPDF['stateVectorID'] == stateDict['stateVectorID']:
                    #         probability = self.lastPDF['dist'].pdf(dY)
                    #     else:
                    #         self.lastPDF = {
                    #             'stateVectorID': stateDict['stateVectorID'],
                    #             'dist': _mvn(cov=residualVariance
                    probability = (
                        maxProb * 
                        _np.exp(-dY.dot(_np.linalg.inv(residualVariance)).dot(dY)/2)
                    )
                    # probability = _mvn.pdf(dY, cov=residualVariance, allow_singular=True)

            except:
                probability = 0
                print('Error computing probability; setting to zero')
                # print('P:')
                # print(P)
                # print('H:')
                # print(H)
                # print('R:')
                # print(R)
                # print('S:')                
                # print(residualVariance)
                # print('dY:')
                # print(dY)
                # raise ValueError(
                #     'Error computing probability.'
                #     )
            # print('')
        else:
            probability=0

        return(probability)

    def generateArrivalVector(
            self,
            attitudeQ,
            AOA_StdDev=None
    ):
        if hasattr(attitudeQ, '__len__'):
            measurement = []
            for attIndex in range(len(attitudeQ)):
                measurement.append(self.generateArrivalVector(attitudeQ[attIndex]))
        else:
            attitudeMatrix = attitudeQ.rotation_matrix.transpose()
            unitVecMeas = attitudeMatrix.dot(self.unitVec())
            RaMeas, DecMeas = sg.unitVector2RaDec(unitVecMeas)

            # If we were given a value for angle of arrival standard deviation,
            # then corrupt the measurements with noise.  Otherwise, simply
            # return the true values.
            if AOA_StdDev:
                RaMeas = RaMeas + _np.random.normal(scale=AOA_StdDev)
                DecMeas = DecMeas + _np.random.normal(scale=AOA_StdDev)
                measurement = {
                    'unitVec': {'value': sg.sidUnitVec(RaMeas, DecMeas)},
                    'RA': {
                        'value': RaMeas,
                        'var': _np.square(AOA_StdDev)
                    },
                    'DEC': {
                        'value': DecMeas,
                        'var': _np.square(AOA_StdDev)
                    }
                }
            else:
                measurement = {
                    'unitVec': {'value': unitVecMeas},
                    'RA': {
                        'value': RaMeas
                    },
                    'DEC': {
                        'value': DecMeas
                    }
                }
        return measurement
        
