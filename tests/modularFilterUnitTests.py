import unittest
from context import modest as md
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal as mvn
class TestModularFilters(unittest.TestCase):
    def setUp(self):
        class simpleState(md.substates.SubState):
            def __init__(self, dimension, stateVectorHistory):
                super().__init__(stateDimension=dimension, stateVectorHistory=stateVectorHistory)
        self.simpleState = simpleState
        
        class oneDPositionVelocity(md.substates.SubState):
            def __init__(self, objectID, stateVectorHistory):
                super().__init__(stateDimension=2, stateVectorHistory=stateVectorHistory)
                self.stateVector = stateVectorHistory['stateVector']
                self.objectID = objectID

            def storeStateVector(self, svDict):
                xPlus = svDict['stateVector']
                aPriori = svDict['aPriori']

                if aPriori is False:
                    self.stateVector = xPlus
                svDict['stateVector'] = self.stateVector
                super().storeStateVector(svDict)
                
            def timeUpdate(self, dT, dynamics=None):
                F = np.array([[1, dT],[0, 1]])
                dT2 = np.square(dT)
                dT3 = np.power(dT, 3)
                dT4 = np.power(dT, 4)
                Q = np.array([[dT4/4, dT3/2],[dT3/2, dT2]])
                accelKey = self.objectID + 'acceleration'
                if dynamics is not None and accelKey in dynamics:
                    acceleration = dynamics[accelKey]['value']
                    accVar = dynamics[accelKey]['var']
                else:
                    acceleration = 0
                    accVar = 0
                self.stateVector = F.dot(self.stateVector) + np.array([0, acceleration])
                Q = Q * accVar

                return {'F': F, 'Q': Q}

            def getMeasurementMatrices(self, measurement, source=None):
                HDict = {}
                RDict = {}
                dyDict = {}
                if isinstance(source, oneDObjectMeasurement) and source.objectID == self.objectID:
                    if 'position' in measurement:
                        H = np.array([[1, 0]])
                        dY = measurement['position']['value'] - H.dot(self.stateVector) 
                        HDict['%s position' %self.objectID] = H
                        RDict['%s position' %self.objectID] = np.array(
                            [[measurement['position']['var']]]
                        )
                        dyDict['%s position' %self.objectID] = dY
                    if 'velocity' in measurement:
                        H = np.array([[0, 1]])
                        dY = measurement['velocity']['value'] - H.dot(self.stateVector)
                        HDict['%s velocity' %self.objectID] = H
                        RDict['%s velocity' %self.objectID] = np.array(
                            [[measurement['velocity']['var']]]
                        )
                        dyDict['%s velocity' %self.objectID] = dY

                return {'H': HDict, 'R': RDict, 'dY': dyDict}

        self.oneDPositionVelocity = oneDPositionVelocity

        class oneDObjectMeasurement(md.signals.SignalSource):
            def __init__(self, objectID):
                self.objectID = objectID
                return
            
            def computeAssociationProbability(self, measurement, stateDict, validationThreshold=0):
                myMeasMat = stateDict[self.objectID].getMeasurementMatrices(measurement, source=self)
                dY = None
                R = None
                H = None
                for key in myMeasMat['dY']:
                    if H is None:
                        H = myMeasMat['H'][key]
                        R = myMeasMat['R'][key]
                        dY = myMeasMat['dY'][key]
                    else:
                        H = np.vstack([H, myMeasMat['H'][key]])
                        R = block_diag(R, myMeasMat['R'][key])
                        dY = np.append(dY, myMeasMat['dY'][key])

                if dY is not None:
                    P = stateDict[self.objectID].covariance()
                    S = H.dot(P).dot(H.transpose()) + R

                    myProbability = mvn.pdf(dY, cov=S)
                else:
                    myProbability = 0
                return myProbability
        self.oneDObjectMeasurement = oneDObjectMeasurement
    
        
    def testAddStates(self):
        # Create a simple state class to add to the filter
            
        stateLength1 = np.random.randint(0, 10)
        cov1 = np.random.randint(0, 10)
        state1 = self.simpleState(
            stateLength1,
            {
                't':0,
                'stateVector':np.zeros(stateLength1),
                'covariance':np.eye(stateLength1)*cov1,
                'stateVectorID':0
            }
        )
        
        stateLength2 = np.random.randint(0, 10)
        cov2 = np.random.randint(0, 10)
        state2 = self.simpleState(
            stateLength2,
            {
                't':0,
                'stateVector':np.zeros(stateLength2),
                'covariance':np.eye(stateLength2)*cov2,
                'stateVectorID':0
            }
        )

        myFilter = md.ModularFilter()
        
        myFilter.addStates('state1', state1)
        myFilter.addStates('state2', state2)

        self.assertEqual(stateLength1 + stateLength2, myFilter.totalDimension)

        stackedCov = block_diag(np.eye(stateLength1) * cov1, np.eye(stateLength2) * cov2)
        self.assertTrue(np.all(stackedCov == myFilter.covarianceMatrix))

        with self.assertRaises(ValueError):
            myFilter.addStates('state1', state2)

    def testTimeUpdateEKF(self):
        myFilter = md.ModularFilter()
        position1 = np.random.normal(1)
        velocity1 = np.random.normal(1)
        x1 = np.array([position1, velocity1])
        cov1 = np.random.normal(np.zeros([2,2]))
        cov1 = cov1.dot(cov1.transpose())
        positionObj1 = self.oneDPositionVelocity(
            'object1',
            {'t': 0,
             'stateVector': np.array([position1, velocity1]),
             'covariance': cov1,
             'stateVectorID': 0
             }
        )
        myFilter.addStates('object1', positionObj1)

        position2 = np.random.normal(1)
        velocity2 = np.random.normal(1)
        cov2 = np.random.normal(np.zeros([2,2]))
        cov2 = cov2.dot(cov2.transpose())
        x2 = np.array([position2, velocity2])
        positionObj2 = self.oneDPositionVelocity(
            'object2',
            {'t': 0,
             'stateVector': np.array([position2, velocity2]),
             'covariance': cov2,
             'stateVectorID': 0
             }
        )
        myFilter.addStates('object2', positionObj2)

        # First, test the time update without the dynamics
        myFilter.timeUpdateEKF(1)
        FMat = np.array([[1,1],[0,1]])
        self.assertTrue(
            np.all(
                positionObj1.covariance() == FMat.dot(cov1).dot(FMat.transpose())
                )
            )
        self.assertTrue(
            np.all(
                positionObj1.getStateVector()['stateVector'] ==
                FMat.dot(np.array([position1, velocity1]))
            )
        )
        
        self.assertTrue(
            np.all(
                positionObj2.covariance() == FMat.dot(cov2).dot(FMat.transpose())
                )
            )
        self.assertTrue(
            np.all(
                positionObj2.getStateVector()['stateVector'] ==
                FMat.dot(np.array([position2, velocity2]))
            )
        )

        # Now, do an update with dynamics
        dynamics={
            'object1acceleration': {'value':0,'var':1},
            'object2acceleration': {'value':1,'var':1}
        }
        cov1 = positionObj1.covariance()
        cov2 = positionObj2.covariance()
        x1 = positionObj1.stateVector
        x2 = positionObj2.stateVector
        myFilter.timeUpdateEKF(1, dynamics=dynamics)
        Q = np.array([[1/4, 1/2],[1/2, 1]])
        self.assertTrue(
            np.all(
                positionObj1.covariance() == FMat.dot(cov1).dot(FMat.transpose()) + Q
                )
            )
        self.assertTrue(
            np.all(
                positionObj1.getStateVector()['stateVector'] ==
                FMat.dot(x1)
            )
        )
        
        self.assertTrue(
            np.all(
                positionObj2.covariance() == FMat.dot(cov2).dot(FMat.transpose()) + Q
                )
            )
        self.assertTrue(
            np.all(
                positionObj2.getStateVector()['stateVector'] ==
                FMat.dot(x2) + np.array([0,1])
            )
        )

        self.assertTrue(positionObj1.getStateVector()['aPriori'])
        self.assertTrue(positionObj2.getStateVector()['aPriori'])
        
        self.assertEqual(positionObj1.getStateVector()['t'], 2)
        self.assertEqual(positionObj2.getStateVector()['t'], 2)

    def testMeasurementUpdateEKF(self):
        myFilter = md.ModularFilter()
        position1 = np.random.normal(1)
        velocity1 = np.random.normal(1)
        x1 = np.array([position1, velocity1])
        cov1 = np.eye(2) * np.abs(np.random.normal(1))
        positionObj1 = self.oneDPositionVelocity(
            'object1',
            {'t': 0,
             'stateVector': np.array([position1, velocity1]),
             'covariance': cov1,
             'stateVectorID': 0
             }
        )
        myFilter.addStates('object1', positionObj1)

        signalSource1 = self.oneDObjectMeasurement('object1')
        myFilter.addSignalSource('object1', signalSource1)

        
        position2 = np.random.normal(1)
        velocity2 = np.random.normal(1)
        cov2 = np.eye(2)*np.abs(np.random.normal(1))
        x2 = np.array([position2, velocity2])
        positionObj2 = self.oneDPositionVelocity(
            'object2',
            {'t': 0,
             'stateVector': np.array([position2, velocity2]),
             'covariance': cov2,
             'stateVectorID': 0
             }
        )
        myFilter.addStates('object2', positionObj2)
        signalSource2 = self.oneDObjectMeasurement('object2')
        myFilter.addSignalSource('object2', signalSource2)

        # Generate postition measurement for state 1
        pos1Meas = np.random.normal(x1[0])
        state1Meas = {
            'position': {
                'value': pos1Meas,
                'var': 1
                }
        }
        myFilter.measurementUpdateEKF(state1Meas, 'object1')

        # Now compute the measurement update independently.
        H = np.array([1,0])
        S = H.dot(cov1).dot(H.transpose()) + np.array([[1]])
        K = cov1.dot(H.transpose())/(S[0,0])
        dY = pos1Meas - H.dot(x1)
        x1Plus = x1 + K.dot(dY)
        IminusKH = (np.eye(2) - np.outer(K,H))
        P1Plus = IminusKH.dot(cov1).dot(IminusKH.transpose()) + np.outer(K,K)

        # Verify that the updated state vector and covariance matches ours
        self.assertTrue(np.allclose(x1Plus, positionObj1.stateVector))
        self.assertTrue(np.allclose(P1Plus, positionObj1.covariance()))
        
        # Verify that state 2 is unchanged
        self.assertTrue(np.allclose(x2,positionObj2.stateVector))
        self.assertTrue(np.allclose(cov2, positionObj2.covariance()))
        
        # Generate velocity measurement for state2
        vVar = 1
        v2Meas = np.random.normal(x2[1], np.sqrt(vVar))
        state2Meas = {
            'velocity': {
                'value': v2Meas,
                'var': vVar
                }
        }
        myFilter.measurementUpdateEKF(state2Meas, 'object2')

        # Now compute the measurement update independently.
        H = np.array([0,1])
        S = H.dot(cov2).dot(H.transpose()) + np.array([[vVar]])
        K = cov2.dot(H.transpose())/(S[0,0])
        dY = v2Meas - H.dot(x2)
        x2Plus = x2 + K.dot(dY)
        IminusKH = (np.eye(2) - np.outer(K,H))
        P2Plus = IminusKH.dot(cov2).dot(IminusKH.transpose()) + np.outer(K,K)*vVar
        
        # Verify that state 2 is updated and matches our state2 computations
        self.assertTrue(np.allclose(x2Plus, positionObj2.stateVector))
        self.assertTrue(np.allclose(P2Plus, positionObj2.covariance()))
        # Verify that state 1 is unchanged
        self.assertTrue(np.allclose(x1Plus, positionObj1.stateVector))
        self.assertTrue(np.allclose(P1Plus, positionObj1.covariance()))

        self.assertTrue(
            np.allclose(block_diag(P1Plus, P2Plus), myFilter.covarianceMatrix)
        )
        
        
    
unittest.main(argv=['ignored','-v'], exit=False)
