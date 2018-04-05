## @package AttitudeSubstate
# This package contains the AttitudeState6DOF class.

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


## @class AttitudeState6DOF
# @brief Estimates the attitude of a vehicle in three dimensions, along with
# three gyro bias states.
#
# @details
# This class contains a six-state attitude estimator: three attitude states
# and three gyro bias states.
#
# This class can function as a stand-alone class, or it can function as a
# "SubState" of the State.ModularFilter class.  The functions required for use
# as a SubState are defined first after __init__, then functions specific to
# this class are defined next.
#
# The state uses quaternions to store attitude, which avoids issues of gimbal
# lock and increases numerical stability over other approaches, such as Euler
# angles.  The quaternion itself is not treated as a part of the state vector.
# Rather, the state vector includes three attitude "error states," which are
# updated at each measurement, then used to correct the attitude quaternion.
# After each correction, the error states are set back to zero.
#
# The algorithms used for the state update mostly come from the book
# "Fundamentals of Spacecraft Attitude Determination and Control" (FSADC) by
# Markley and Crassidis.  Chapter, section and page numbers will be referenced
# where appropriate.
class AttitudeState6DOF():
    
    ## @fun __init__
    # @brief This function initializes the 6DOF attitude estimator
    #
    # @details This function is responsible for initializing an instance of
    # the AttitudeState6DOF class and storing all the variables as member
    # variables.
    #
    # @param self The object pointer
    # @param attitudeQuaternion A pyquaternion.Quaternion object containing
    # the initial attitude estimate.  This variable gets stored as #qHat.
    # @param attitudeErrorCovariance A 3x3 numpy array containing the
    # covariance of the current attitude estimate.  This matrix is used to
    # form the upper diagonal part of #PHat.
    # @param gyroBias A 3 dimensional numpy array containing the estimate of
    # gyro bias.  This array is stored as #bHat.
    # @param gyroBiasCovariance A 3x3 numpy array containing the estimate of
    # covariance of gyro bias.  This array is used to form the lower diagonal
    # part of #PHat.
    def __init__(
            self,
            attitudeQuaternion=Quaternion([1,0,0,0]),
            attitudeErrorCovariance=np.eye(3),
            gyroBias=np.zeros(3),
            gyroBiasCovariance=np.eye(3)
            ):
        
        ## @brief Current estimate of attitude, stored as a Quaternion object
        # Mathematically generally referred to as \f$\mathbf{\hat{q}}^{-}_{k}\f$
        # for the a priori value, or \f$\mathbf{\hat{q}}^{+}_{k}\f$ for the a
        # posteriori value.
        self.qHat = attitudeQuaternion

        ## @brief Current estimate of gyro bias
        self.bHat = gyroBias

        ## @brief Current joint covariance matrix.
        # @details Upper 3x3 diagonal contains covariance of the attitude
        # estimate (related to #qHat), while lower 3x3 diagonal contains the
        # covariance of the gyro bias #bHat.
        self.PHat = block_diag(
            attitudeErrorCovariance,
            gyroBiasCovariance
        )

        ## @brief Last measurement used to generate measurement matrices
        self.lastMeasID = None
        
        ## @brief Last signal used to generate measurement matrices
        self.lastSourceID = None
        
        ## @brief Last set of measurement matrices
        # @details This allows class to avoid redundant computation of the
        # same set of measurement matrices.
        self.lastMeasMat = None

        ## @brief Array of Euler angle history
        # @details This is a SmartPanda array that stores the history of time,
        # Euler angles, and Euler angle covariances.
        self.eulerAngleVec = SmartPanda(
            ['t', 'eulerAngles', 'eulerSTD'],
            [1, 3, 3]
        )

        return

    """
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

    ## @fun getStateVector is responsible for passing whatever version of the
    # state vector should be used for time or measurement updates.
    #
    # @details The "state" for this filter does not include the attitude
    # quaternion.  Rather, the attitude quaternion is updated internally.
    # Instead, the state that represents attitude is the attitude error state,
    # as described in FSADC.
    #
    # @note This function is one of mandatory functions required for
    # AttitudeState6DOF to function as a sub-state of State.ModularFilter.
    #
    # @param self The object pointer
    def getStateVector(
            self
            ):
        return np.append(np.zeros(3), self.bHat)

    ## @fun storeStateVector is responsible for taking an updated version of
    # the state vector, and storing it in the class variables.
    #
    # @details This function is designed to receive an time or measurement
    # updated state vector and covariance, and store it.  This function is
    # used by State.ModularFilter to store a jointly updated state.
    #
    # Depending on whether the state vector is the result of a time update
    # (aPriori=True) or a measurement update (aPriori=False), the function
    # may disregard the value of the attitude error state.  This is because
    # this class handles the time-update of #qHat internally, so the updated
    # attitude error state is only relevant after a measurement update.
    #
    # @note This function is one of mandatory functions required for
    # AttitudeState6DOF to function as a sub-state of State.ModularFilter.
    #
    # @param self The object pointer
    # @param xPlus A 6 dimensional array (usually numpy) that contains the
    # updated attitude error state and updated gyro bias state.
    # @param PPlus A 6x6 array that contains the updated covariance matrix
    # @param time Contains the time-tag of the new state vector and covariance
    # matrix
    # @param aPriori=False An optional tag to indicate whether the updated
    # state vector is "a priori" or "a posteriori."
    def storeStateVector(
            self,
            xPlus,
            PPlus,
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
                    xPlus[0]/2.0,
                    xPlus[1]/2.0,
                    xPlus[2]/2.0
                ]))
            qPlus = errorQ * self.qHat
            qPlus = qPlus.normalised

            self.qHat = qPlus
            self.bHat = xPlus[3:6]

        self.eulerAngleVec.append({
            't': time,
            'eulerAngles': self.eulerAngles(),
            'eulerSTD': np.sqrt(self.PHat.diagonal()[0:3])
        })

        self.PHat = PPlus
        
        return

    ## @fun dimension returns the dimension of the sub-state vector
    #
    # @note This function is one of mandatory functions required for
    # AttitudeState6DOF to function as a sub-state of State.ModularFilter.
    #
    # @param self The object pointer
    #
    # @return Returns the dimension of state vector, 6
    def dimension(
            self
            ):
        return(6)
    
    ## @fun dimension returns the covariance of the sub-state vector
    #
    # @note This function is one of mandatory functions required for
    # AttitudeState6DOF to function as a sub-state of State.ModularFilter.
    #
    # @param self The object pointer
    #
    # @return Returns covariance matrix, #PHat
    def covariance(
            self
            ):
        return (self.PHat)

    ## @fun timeUpdate returns the time-update matrices, and handles the
    # internal time update of the attitude estimate #qHat.
    #
    # @details This function generates the time-update matrices F and Q, to be
    # used for a time update, either locally or jointly as part of a
    # State.ModularFilter.
    #
    # This function looks for angular velocity (omega) and bias variance in
    # the dynamics dict, and uses these to construct the time update matrices.
    # If these are not included in the dynamics dict, then the function
    # assumes these values to be zero.
    #
    # This function also updates the attitude quaternion internally.  It does
    # not update the covariance matrix however; this must be done externally.
    #
    # @note This function is one of mandatory functions required for
    # AttitudeState6DOF to function as a sub-state of State.ModularFilter.
    #
    # @param self The object pointer
    # @param dynamics: A dict containing information about the dynamics.
    #
    # @return A dict containing the state transition matrix ("F") and the
    # process noise matrix ("Q")
    def timeUpdate(
            self,
            dT,
            dynamics=None
    ):

        # Check the dynamics dict for angular velocity
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

        # Check dynamics dict for bias variance
        if (
                (dynamics is not None) and
                ('bias' in dynamics)
        ):
            biasVar = dynamics['bias']['var']
        else:
            biasVar = 0

        # Estimated angular velocity is equal to the measured velocity minus
        # the estimated gyro bias
        correctedOmega = myOmega - self.bHat

        # Generate the time update matrices
        qUpdateMatrix = self.quaternionTimeUpdateMatrix(correctedOmega, dT)
        errorUpdateMatrix = self.errorStateTimeUpdateMatrix(correctedOmega, dT)
        processNoise = self.processNoiseMatrix(dT, omegaVar, biasVar)

        # Perform time update on attitude quatnerion
        self.qHat = Quaternion(qUpdateMatrix.dot(self.qHat.q))
        self.qHat = self.qHat.normalised

        # Dictionary to containing the time update matrix and process noise
        # matrix
        timeUpdateDict = {
            'F': errorUpdateMatrix,
            'Q': processNoise
        }

        return timeUpdateDict

    ## @fun getMeasurementMatrices computes and returns measurement update
    # matrices
    #
    # @details This function receives a dictionary containing a measurement,
    # along with an object that contains the source model of the measurement.
    # If the source is a Signals.PointSource type signal, then it generates
    # unit-vector attitude measurement type matrices.  Otherwise, the function
    # returns dicts populated with None.
    #
    # @note This function is one of mandatory functions required for
    # AttitudeState6DOF to function as a sub-state of State.ModularFilter.
    #
    # @param self The object pointer
    # @param measurement A dictionary containing measurement information
    # @param source The source object that produced the measurement
    #
    # @return A dictionary containing the measurement matrices H, R, and dY
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

    """
    ###########################################################################
    # Functions Specific to This Class                                        #
    #                                                                         #
    # The following remaining functions are not required in order for this    #
    # class to be used as a SubState, and may be changed as needed,           #
    # including inputs and outputs.                                           #
    ###########################################################################
    """

    ## @fun quaternionTimeUpdateMatrix produces a time-update matrix for the
    # attitude quaternion
    #
    # @details This function produces a 4x4 matrix which, when multiplied by
    # an attitude quaternion, rotates the quaternion by an amount
    # corresponding to the angular velocity and time ellapsed.  The attitude
    # quaternion is updated as follows:
    #
    # \f[
    # \attVec[est=True,aPriori=True, t=k+1] \approx
    # \bar{\Theta}(\omegaVec[est=True,aPriori=True, t=k], \Delta T)
    # \attVec[est=True, aPriori=False, t=k]
    # \f]
    #
    # where
    #
    # \f[
    # \bar{\Theta}( \omegaVec[est=True,aPriori=True, t=k], \Delta T ) =
    # \begin{bmatrix}
    # \textrm{cos} \left(\frac{1}{2}
    # ||\omegaVec[est=True,aPriori=True, t=k]|| \Delta t \right) I_3 -
    # \left[\boldsymbol{\hat{\Psi}}_k^+ \times \right]
    # & \left[\boldsymbol{\hat{\Psi}}_k^+ \times \right] \\
    # - \left[\boldsymbol{\hat{\Psi}}_k^+ \times \right] &
    # \textrm{cos} \left(\frac{1}{2} ||\mathbf{\hat{\omega}}_k^+|| \Delta t \right)
    # \end{bmatrix}
    # \f]
    #
    # and
    #
    # \f[
    # \left[\boldsymbol{\hat{\Psi}}_k^+ \times \right] =
    # \frac{
    # \textrm{sin}\left(\frac{1}{2} || \omegaVec[est=True,aPriori=True, t=k] || \Delta t \right)
    # \omegaVec[est=True,aPriori=True, t=k]
    # }{
    # || \omegaVec[est=True,aPriori=True, t=k] ||
    # }
    # \f]
    #
    # The matrix returned by this function is
    # \f$\bar{\Theta}(\omegaVec[est=True,aPriori=True, t=k], \Delta T)\f$.
    #
    # See Fundamentals of Spacecraft Attitude Determination and Control,
    # Section 6.2.2, page 251, equation 6.60 for more details.
    #
    # @param self The object pointer
    # @param myOmega The angular velocity estimate used to update the attitude
    # quaternion
    # @param deltaT The amount of time elapsed for the time-update, used for
    # numerical integration of kinematics equation.
    #
    # @return The quaternion time-update matrix
    # \f$\bar{\Theta}(\omegaVec[est=True,aPriori=True, t=k], \Delta T)\f$
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

    ## @fun errorStateTimeUpdateMatrix produces a time-update matrix for the
    # attitude error state
    #
    # @details
    # This function the discrete-time error-state transition matrix.  This is
    # the matrix which propagates the attitude error state covariance and gyro
    # bias covariance forward in time based on time ellapsed and angular
    # velocity estimate.
    #
    # The error-state transition matrix is defined as follows:
    #
    # \f[
    # \boldsymbol{\Phi} = \begin{bmatrix}
    # \boldsymbol{\Phi}_{11} & \boldsymbol{\Phi}_{12} \\
    # \boldsymbol{\Phi}_{21} & \boldsymbol{\Phi}_{22} \\
    # \end{bmatrix}
    # \f]
    #
    # where
    #
    # \f[
    # \boldsymbol{\Phi}_{11} = \eye[3] -
    # \left[\omegaVec[est=True,aPriori=True, t=k] \times \right]
    # \frac
    # {\textrm{sin}(||\omegaVec[est=True,aPriori=True, t=k]|| \Delta t)}
    # {||\omegaVec[est=True,aPriori=True, t=k]||} +
    # \left[\omegaVec[est=True,aPriori=True, t=k] \times \right]^2
    # \frac
    # {1 - \textrm{cos}(1 - ||\omegaVec[est=True,aPriori=True, t=k]|| \Delta t)}
    # {||\omegaVec[est=True,aPriori=True, t=k]||^2}
    # \f]
    #
    # \f[
    # \boldsymbol{\Phi}_{12} = 
    # \left[\omegaVec[est=True,aPriori=True, t=k] \times \right]
    # \frac
    # {1 - \textrm{cos}(1 - ||\omegaVec[est=True,aPriori=True, t=k]|| \Delta t)}
    # {||\omegaVec[est=True,aPriori=True, t=k]||^2} -
    # \eye[3]\Delta t -
    # \left[\omegaVec[est=True,aPriori=True, t=k] \times \right]^2
    # \frac
    # {||\omegaVec[est=True,aPriori=True, t=k]|| \Delta t -
    # \textrm{sin}(||\omegaVec[est=True,aPriori=True, t=k]|| \Delta t)}
    # {||\omegaVec[est=True,aPriori=True, t=k]||^3}
    # \f]
    #
    # \f[
    # \boldsymbol{\Phi}_{21} = \mathbf{0}_{3 \times 3}
    # \f]
    #
    # \f[
    # \boldsymbol{\Phi}_{22} = \eye[3]
    # \f]
    #
    # See Fundamentals of Spacecraft Attitude Determination and Control,
    # Section 6.2.4, page 258, equation 6.83 for more details and derivation.
    #
    # @param self The object pointer
    # @param myOmega The angular velocity estimate used to update the attitude
    # quaternion
    # @param deltaT The amount of time elapsed for the time-update, used for
    # numerical integration of kinematics equation.
    #
    # @return Returns the error-state time update matrix, \f$\boldsymbol{\Phi}\f$
    
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
