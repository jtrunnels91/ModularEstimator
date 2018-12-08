import numpy as np
#from numpy import sin, cos, arcsin, arccos, arctan2, square, sqrt, abs, power
from . import substate
from .. utils import covarianceContainer

class oneDPositionVelocity(substate.SubState):
    def __init__(self, objectID, stateVectorHistory, covarianceStorage='covariance', biasState=True):
        
        if not isinstance(stateVectorHistory['covariance'], covarianceContainer):
            stateVectorHistory['covariance'] = covarianceContainer(
                stateVectorHistory['covariance'],covarianceStorage
            )
        self.biasState = biasState
        if biasState:
            super().__init__(stateDimension=3, stateVectorHistory=stateVectorHistory)
        else:
            super().__init__(stateDimension=2, stateVectorHistory=stateVectorHistory)
        self.stateVector = stateVectorHistory['stateVector']
        self.objectID = objectID
        self.velocityVar = 1e10
        self.positionVar = 1e10
    def storeStateVector(self, svDict):
        xPlus = svDict['stateVector']
        aPriori = svDict['aPriori']

        if aPriori is False:
            self.stateVector = xPlus
            
        self.currentPosition = xPlus[0]
        self.positionVar = svDict['covariance'].convertCovariance('covariance').value[0,0]
        self.currentVelocity = xPlus[1]
        self.velocityVar = svDict['covariance'].convertCovariance('covariance').value[1,1]

        svDict['position'] = self.currentPosition
        svDict['velocity'] = self.currentVelocity
        svDict['positionStd'] = np.sqrt(self.positionVar)
        svDict['velocityStd'] = np.sqrt(self.velocityVar)
        
        svDict['stateVector'] = self.stateVector
        super().storeStateVector(svDict)

    def timeUpdate(self, dT, dynamics=None):
        if self.biasState:
            F = np.array([[1, dT, 0],[0, 1, 0], [0, 0, np.power(1 + 1e-1, -dT)]])
        else:
            F = np.array([[1, dT],[0, 1]])
        dT2 = np.square(dT)
        dT3 = np.power(dT, 3)
        dT4 = np.power(dT, 4)
        if self.covariance().form == 'covariance':
            if self.biasState:
                Q = np.array([[dT4/4, dT3/2, 0],[dT3/2, dT2, 0], [0,0,0.0001]])
            else:
                Q = np.array([[dT4/4, dT3/2],[dT3/2, dT2]])
        elif self.covariance().form == 'cholesky':
            if self.biasState:
                Q = np.array([[dT2/2,0, 0],[dT,0, 0], [0,0,0.01]])
            else:
                Q = np.array([[dT2/2,0],[dT,0]])
            
        accelKey = self.objectID + 'acceleration'
        if dynamics is not None and accelKey in dynamics:
            acceleration = dynamics[accelKey]['value']
            accVar = dynamics[accelKey]['var']
        else:
            acceleration = 0
            accVar = 0
        if self.biasState:
            self.stateVector = F.dot(self.stateVector) + np.array([0, acceleration, 0])
        else:
            self.stateVector = F.dot(self.stateVector) + np.array([0, acceleration])
        if self.covariance().form == 'covariance':
            Q = covarianceContainer(Q * accVar, 'covariance')
        elif self.covariance().form == 'cholesky':
            Q = covarianceContainer(Q * np.sqrt(accVar), 'cholesky')
        else:
            raise ValueError('unrecougnized covariance')
        return {'F': F, 'Q': Q}

    def getMeasurementMatrices(self, measurement, source=None):
        HDict = {}
        RDict = {}
        dyDict = {}

        if 'position' in measurement:
            if self.biasState:
                H = np.array([[1, 0, 1]])
            else:
                H = np.array([[1, 0]])
            dY = measurement['position']['value'] - H.dot(self.stateVector) 
            HDict['%s position' %self.objectID] = H
            RDict['%s position' %self.objectID] = np.array(
                [[measurement['position']['var']]]
            )
            dyDict['%s position' %self.objectID] = dY
        if 'velocity' in measurement:
            if self.biasState:
                H = np.array([[0, 1, 0]])
            else:
                H = np.array([[0, 1]])
            dY = measurement['velocity']['value'] - H.dot(self.stateVector)
            HDict['%s velocity' %self.objectID] = H
            RDict['%s velocity' %self.objectID] = np.array(
                [[measurement['velocity']['var']]]
            )
            dyDict['%s velocity' %self.objectID] = dY

        if self.biasState:
            HDict['artificialBiasMeas'] = np.array([[0,0,1]])
            RDict['artificialBiasMeas'] = np.array([[1e-3]])
            dyDict['artificialBiasMeas'] = 0 - H.dot(self.stateVector)
        return {'H': HDict, 'R': RDict, 'dY': dyDict}

