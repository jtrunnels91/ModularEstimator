import numpy as np
#from numpy import sin, cos, arcsin, arccos, arctan2, square, sqrt, abs, power
from . import substate
from .. utils import covarianceContainer

class oneDPositionVelocity(substate.SubState):
    def __init__(
            self,
            objectID,
            stateVectorHistory,
            covarianceStorage='covariance',
            biasState=True,
            artificialBiasMeas=True,
            biasStateTimeConstant=0.9,
            biasStateProcessNoiseVar=1e-3,
            biasMeasVar=1,
            storeLastStateVectors=0,
    ):
        
        if not isinstance(stateVectorHistory['covariance'], covarianceContainer):
            stateVectorHistory['covariance'] = covarianceContainer(
                stateVectorHistory['covariance'],covarianceStorage
            )
        self.biasState = biasState
        if biasState:
            super().__init__(stateDimension=3, stateVectorHistory=stateVectorHistory, storeLastStateVectors=storeLastStateVectors)
        else:
            super().__init__(stateDimension=2, stateVectorHistory=stateVectorHistory,storeLastStateVectors=storeLastStateVectors)
        self.stateVector = stateVectorHistory['stateVector']
        self.objectID = objectID
        self.velocityVar = (
            stateVectorHistory['covariance'].convertCovariance('covariance').value[1,1]
        )
        self.positionVar = (
            stateVectorHistory['covariance'].convertCovariance('covariance').value[0,0]
        )
        stateVectorHistory['positionStd'] = np.sqrt(self.positionVar)
        stateVectorHistory['velocityStd'] = np.sqrt(self.velocityVar)
        self.currentPosition = 0
        self.currentVelocity=0
        self.currentBiasState = 0
        
        self.artificialBiasMeas = artificialBiasMeas

        self.biasStateProcessNoiseVar = biasStateProcessNoiseVar
        self.biasStateTimeConstant = biasStateTimeConstant
        self.artificialBiasMeasVar = biasMeasVar

    def storeStateVector(self, svDict):
        xPlus = svDict['stateVector']
        aPriori = svDict['aPriori']

        if aPriori is False:
            self.stateVector = xPlus
            
        self.currentPosition = xPlus[0]
        self.positionVar = svDict['covariance'].convertCovariance('covariance').value[0,0]
        self.currentVelocity = xPlus[1]
        self.velocityVar = svDict['covariance'].convertCovariance('covariance').value[1,1]

        if self.biasState:
            self.currentBiasState = xPlus[2]
            svDict['biasState'] = self.currentBiasState
        else:
            svDict['biasState'] = 0

        svDict['position'] = self.currentPosition
        svDict['velocity'] = self.currentVelocity
        svDict['positionStd'] = np.sqrt(self.positionVar)
        svDict['velocityStd'] = np.sqrt(self.velocityVar)
        
        svDict['stateVector'] = self.stateVector
        super().storeStateVector(svDict)

    def timeUpdate(self, dT, dynamics=None):
        if self.biasState:
            #F = np.array([[1, dT, 0],[0, 1, 0], [0, 0, np.power(1 + 1e-1, -dT)]])
            F = np.array([[1, dT, 0],[0, 1, 0], [0, 0, np.exp(-dT/self.biasStateTimeConstant)]])
        else:
            F = np.array([[1, dT],[0, 1]])
        dT2 = np.square(dT)
        dT3 = np.power(dT, 3)
        dT4 = np.power(dT, 4)
        if self.covariance().form == 'covariance':
            if self.biasState:
                Q = np.array([[dT4/4, dT3/2, 0],[dT3/2, dT2, 0], [0,0,self.biasStateProcessNoiseVar * dT2]])
            else:
                Q = np.array([[dT4/4, dT3/2],[dT3/2, dT2]])
        elif self.covariance().form == 'cholesky':
            if self.biasState:
                Q = np.array([[dT2/2,0, 0],[dT,0, 0], [0,0,0]])
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
            self.stateVector = F.dot(self.stateVector) + np.array([0, acceleration * dT, 0])
        else:
            self.stateVector = F.dot(self.stateVector) + np.array([0, acceleration * dT])
        if self.covariance().form == 'covariance':
            Q = covarianceContainer(Q * accVar, 'covariance')
            if self.biasState:
                Q[2,2] = self.biasStateProcessNoiseVar * dT*dT
        elif self.covariance().form == 'cholesky':
            Q = covarianceContainer(Q * np.sqrt(accVar), 'cholesky')
            if self.biasState:
                Q[2,2] = np.sqrt(self.biasStateProcessNoiseVar) * dT
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

        if self.biasState and self.artificialBiasMeas:
            HDict['artificialBiasMeas'] = np.array([[0,0,1]])
            RDict['artificialBiasMeas'] = np.array([[self.artificialBiasMeasVar]])
            # RDict['artificialBiasMeas'] = np.array([[1]])
            dyDict['artificialBiasMeas'] = -self.stateVector[2]
        return {'H': HDict, 'R': RDict, 'dY': dyDict}

