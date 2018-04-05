import numpy as np
from scipy.linalg import block_diag
from numpy import sin, cos, arcsin, arccos, arctan2, square, sqrt, abs, power
from numpy.linalg import norm
from pyquaternion import Quaternion

import sys
import os
sys.path.append("/home/joel/Documents/astroSourceTracking/libraries")
from SpaceGeometry import *
from Signals import PointSource
from SmartPanda import SmartPanda

"""
AttitudeState6DOF estimates the attitude of a vehicle.  It is a six-state
estimator: three attitude states and three gyro bias states.

This class can function as a stand-alone class, or it can function as a
"SubState" of the ModularFilter class.  The functions required for use as a
SubState are defined first after __init__, then functions specific to this
class are defined next.

The state uses quaternions to store attitude, which avoids issues of gimbal
lock and increases numerical stability over other approaches, such as Euler
angles.  The quaternion itself is not treated as a part of the state vector.
Rather, the state vector includes three attitude "error states," which are
updated at each measurement, then used to correct the attitude quaternion.
After each correction, the error states are set back to zero.

The algorithms used for the state update mostly come from the book
"Fundamentals of Spacecraft Attitude Determination and Control" (FSADC) by
Markley and Crassidis.  Chapter, section and page numbers will be referenced
where appropriate.
"""
class AttitudeState6DOF():
    def __init__(
            self,
            attitudeQuaternion,
            attitudeErrorCovariance,
            gyroBias,
            gyroBiasCovariance
            ):

        self.qHat = attitudeQuaternion
        self.bHat = gyroBias

        self.PHat = block_diag(
            attitudeErrorCovariance,
            gyroBiasCovariance
        )

        self.lastMeasID = None
        self.lastSourceID = None
        self.lastMeasMat = None

        self.eulerAngleVec = SmartPanda(
            ['t', 'eulerAngles', 'eulerSTD'],
            [1, 3, 3]
        )
        
        return

    ###########################################################################
    # Functions Required To Function as a "Substate"                          #
    #                                                                         #
    # The following functions are required in order for this class to be used #
    # as a substate in ModularFilter.  The inside of the functions may be     #
    # changed or updated, but their "black box" behavior must remain the      #
    # same; i.e. they must still perform the same essential functions and     #
    # return the same things.                                                 #
    ###########################################################################
    """
    getStateVector is responsible for passing whatever version of the state
    vector should be used for time or measurement updates.

    The "state" for this filter does not include the attitude quaternion.  Rather, the attitude quaternion is updated internally.  Instead, the state that represents attitude is the attitude error state, as described in FSADC.
    """
    def getStateVector(
            self
            ):
        return np.append(np.zeros(3), self.bHat)

    def storeStateVector(
            self,
            deltaXPlus,
            newP,
            time,
            aPriori=False
            ):

        # Only update the quaternion if the new state vector is the result of
        # a measurement update.  The attitude class is responsible for
        # time-updating the quaternion.
        if aPriori is False:
            errorQ = Quaternion(
                array=np.array([
                    1.0,
                    deltaXPlus[0]/2.0,
                    deltaXPlus[1]/2.0,
                    deltaXPlus[2]/2.0
                ]))
            qPlus = errorQ * self.qHat
            qPlus = qPlus.normalised

            self.qHat = qPlus
            self.bHat = deltaXPlus[3:6]

        self.eulerAngleVec.append({
            't': time,
            'eulerAngles': self.eulerAngles(),
            'eulerSTD': np.sqrt(self.PHat.diagonal()[0:3])
        })

        self.PHat = newP
        
        return
    
    def dimension(
            self
            ):
        return(6)

    def covariance(
            self
            ):
        return (self.PHat)

    def timeUpdate(
            self,
            dT,
            dynamics=None
    ):

        if (
                (dynamics is not None) and
                ('omega' in dynamics)
        ):
                omegaDict = dynamics['omega']
                myOmega = omegaDict['value']
                omegaVar = omegaDict['var']
        else:
            myOmega = np.zeros([3])
            omegaVar = 0

        if (
                (dynamics is not None) and
                ('bias' in dynamics)
        ):
            biasVar = dynamics['bias']['var']
        else:
            biasVar = 0
        correctedOmega = myOmega - self.bHat

        qUpdateMatrix = self.quaternionTimeUpdateMatrix(correctedOmega, dT)
        errorUpdateMatrix = self.errorStateTimeUpdateMatrix(correctedOmega, dT)
        processNoise = self.processNoiseMatrix(dT, omegaVar, biasVar)
        
        self.qHat = Quaternion(qUpdateMatrix.dot(self.qHat.q))
        self.qHat = self.qHat.normalised

        return {'F': errorUpdateMatrix, 'Q': processNoise}

    def getMeasurementMatrices(
            self,
            measurement,
            source=None
    ):
            
        if (
                isinstance(source, PointSource)
        ):
            measurementMatrices = self.RaDecMeasMatrices(
                source,
                measurement
                )

            HDict = {'unitVector': measurementMatrices['H']}
            RDict = {'unitVector': measurementMatrices['R']}
            dyDict = {'unitVector': measurementMatrices['dY']}
            
        else:
            HDict = {'': None}
            RDict = {'': None}
            dyDict = {'': None}

        measurementMatricesDict = {
            'H': HDict,
            'R': RDict,
            'dY': dyDict
            }
        
        return(measurementMatricesDict)


    ###########################################################################
    # Functions Specific to This Class                                        #
    #                                                                         #
    # The following remaining functions are not required in order for this    #
    # class to be used as a SubState, and may be changed as needed,           #
    # including inputs and outputs.                                           #
    ###########################################################################

    """
    See Fundamentals of Spacecraft Attitude Determination and Control,
    Section 6.2.2, page 251, equation 6.60
    """
    def quaternionTimeUpdateMatrix(
            self,
            myOmega,
            deltaT
    ):
        omegaNorm = norm(myOmega)
        cosineTerm = cos(0.5 * omegaNorm * deltaT)
        if abs(omegaNorm) < 1e-100:
            psiK = np.zeros(3)
        else:
            psiK = (sin(0.5 * omegaNorm * deltaT) / omegaNorm) * myOmega
        theta = np.zeros([4,4])

        theta[1:4, 1:4] = cosineTerm*np.eye(3) - self.skewSymmetric(psiK)

        theta[0, 1:4] = -psiK
        theta[1:4, 0] = psiK
        theta[0, 0] = cosineTerm
        return theta

    """
    See Fundamentals of Spacecraft Attitude Determination and Control,
    Section 6.2.4, page 258, equation 6.83
    """
    def errorStateTimeUpdateMatrix(
            self,
            myOmega,
            deltaT
    ):
        omegaNorm = norm(myOmega)
        omegaNormSquare = square(omegaNorm)
        omegaNormDT = omegaNorm * deltaT
        omegaSkew = self.skewSymmetric(myOmega)
        omegaSkewSquare = power(omegaSkew, 2)
        if omegaNorm>1e-100:
            phi11 = (
                np.eye(3) -
                (omegaSkew * sin(omegaNormDT)/omegaNorm) +
                (omegaSkewSquare * (1 - cos(omegaNormDT))/omegaNormSquare)
                )
            phi12 = (
                (omegaSkew * (1 - cos(omegaNormDT))/omegaNormSquare) -
                (np.eye(3) * deltaT) -
                (omegaSkewSquare * (omegaNormDT - sin(omegaNormDT))/power(omegaNorm,3))
                )
        else:
            phi11 = (
                np.eye(3)
                )
                
            phi12 = (
                (-np.eye(3) * deltaT)
                )

        phi = np.vstack([
            np.concatenate([phi11, phi12], axis=1),
            np.concatenate([np.zeros([3,3]), np.eye(3)], axis=1)
            ])
                    
        return(phi)
    
    """
    See Fundamentals of Spacecraft Attitude Determination and Control,
    Section 6.2.4, page 260, equation 6.93
    """
    def processNoiseMatrix(
            self,
            deltaT,
            omegaVar,
            biasVar
            ):
        deltaTSquared = power(deltaT, 2)
        Q11 = (
            (omegaVar * deltaT) +
            (biasVar * power(deltaT, 3))/3
            ) * np.eye(3)

        Q12 = (
            -(biasVar * deltaTSquared)/2
            ) * np.eye(3)

        Q21 = Q12

        Q22 = (biasVar * deltaT) * np.eye(3)

        Q = np.vstack([
            np.concatenate([Q11, Q12], axis=1),
            np.concatenate([Q21, Q22], axis=1)
            ])
                    
        return(Q)
    
    def RaDecMeasMatrices(
            self,
            source,
            measurement
            ):

        RaDecTrue = source.RaDec()
        
        uTrue = self.sidUnitVec(RaDecTrue)
        uMeas = self.sidUnitVec(
            {
                'RA': measurement['RA']['value'],
                'DEC': measurement['DEC']['value']
            }
        )
        
        estimatedAttitudeMatrix = self.qHat.rotation_matrix.transpose()

        uPred = estimatedAttitudeMatrix.dot(uTrue)

        H = estimatedAttitudeMatrix.dot(self.skewSymmetric(uTrue))
        H = np.append(H, np.zeros([3, 3]), axis=1)

        varR = measurement['RA']['var']
        varD = measurement['DEC']['var']

        measR = measurement['RA']['value']
        measD = measurement['DEC']['value']

        sinD = sin(measD)
        cosD = cos(measD)

        sinR = sin(measR)
        cosR = cos(measR)

        # R = np.zeros([2, 2])
        # R[0, 0] = varR
        # R[1, 1] = varD

        # M = np.zeros([3, 2])

        # M[0, 0] = -(sinR * cosD)
        # M[0, 1] = -(cosR * sinD)

        # M[1, 0] = (cosR * cosD)
        # M[1, 1] = -(sinR * sinD)
        
        # M[2, 0] = 0
        # M[2, 1] = cosD
        # R = M.dot(R).dot(M.transpose())

        # Note:
        # Fundamentals of Spacecraft Attitude Determination and Control
        # asserts that the measurement noise matrix can be approximated by an
        # identity matrix as shown below.  I don't understand exactly why this
        # works, but using the standard EKF derivation leads to a very poorly
        # behaved filter.  So, using this for now.
        #
        # See Section 6.2.3, page 254
        
        R = np.eye(3) * varR
        
        dY = uMeas - uPred

        measMatrices = {
            'H': H,
            'R': R,
            'dY': dY
            }
        
        return(measMatrices)
    
    def eulerAngles(self):
        q = self.qHat
        phi = arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(square(q[1]) + square(q[2])))
        theta = arcsin(2 * ((q[0] * q[2]) - (q[3] * q[1])))
        psi = arctan2(2 * (q[0] * q[3] + q[1]*q[2]), 1 - 2*(square(q[2]) + square(q[3])))

        return [phi, theta, psi]

    def RaDecRoll(self):
        eulerAngles = self.eulerAngles()

        return([eulerAngles[2], -eulerAngles[1], eulerAngles[0]])

    def sidUnitVec(
            self,
            RaDec):
        cosD = np.cos(RaDec['DEC'])
        sinD = np.sin(RaDec['DEC'])
        cosRA = np.cos(RaDec['RA'])
        sinRA = np.sin(RaDec['RA'])

        return np.array([cosD * cosRA, cosD * sinRA, sinD])

    def skewSymmetric(
            self,
            vector
    ):
        matrix = np.zeros([3, 3])

        matrix[0, 1] = -vector[2]
        matrix[0, 2] = vector[1]

        matrix[1, 0] = vector[2]
        matrix[1, 2] = -vector[0]

        matrix[2, 0] = -vector[1]
        matrix[2, 1] = vector[0]

        return(matrix)
