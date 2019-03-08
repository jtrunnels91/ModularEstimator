## @file Attitude
# This file contains the Attitude class.
import numpy as np
from scipy.linalg import block_diag
from numpy import sin, cos, arcsin, arccos, arctan2, square, sqrt, power
from numpy.linalg import norm
from pyquaternion import Quaternion

#from . SubState import SubState
from .. signals.pointsource import PointSource
from . import substate
from .. utils import covarianceContainer, spacegeometry, QuaternionHelperFunctions


class Attitude(substate.SubState):
    """
    Estimates the attitude of a vehicle in three dimensions, along with
    three gyro bias states.

    This class contains a six-state attitude estimator: three attitude states
    and three gyro bias states.

    This class can function as a stand-alone class, or it can function as a
    :class:`~modest.substates.substate.SubState` of the 
    :class:`modest.modularfilter.ModularFilter` class.  The functions required
    for use as a SubState are defined first after __init__, then functions
    specific to this class are defined next.

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

    Args:
     attitudeQuaternion (pyquaternion.Quaternion): object containing the initial attitude estimate.  This variable gets stored as :attr:`qHat`.
     attitudeErrorCovariance (numpy.ndarray): A 3x3 numpy array containing the covariance of the current attitude estimate.  This matrix is used to form the upper diagonal part of :attr:`PHat`.
     gyroBias (numpy.ndarray): A 3 dimensional numpy array containing the estimate of gyro bias.  This array is stored as :attr:`bHat`.
     gyroBiasCovariance (numpy.ndarray): A 3x3 numpy array containing the estimate of covariance of gyro bias.  This array is used to form the lower diagonal part of :attr:`PHat`.

    """
    
    def __init__(
            self,
            attitudeQuaternion=Quaternion([1,0,0,0]),
            attitudeErrorCovariance=np.eye(3),
            gyroBias=np.zeros(3),
            gyroBiasCovariance=np.eye(3),
            t=0,
            covarianceStorage='covariance',
            useUnitVector=True,
            storeLastStateVectors=0,
            ):

        self.useUnitVector = useUnitVector
        """
        (bool) Determines whether the unit vector is used for the measurement matrix, or the right ascension declination measurement model
        """        
        
        self.qHat = attitudeQuaternion
        """
        (pyquaternion.Quaternion): Current estimate of attitude, stored as a Quaternion object. 

        Mathematically generally referred to as :math:`\mathbf{\hat{q}}^{-}_{k}` for the *a priori* value, or :math:`\mathbf{\hat{q}}^{+}_{k}` for the *a posteriori*
        """        

        self.bHat = gyroBias
        """
        (numpy.ndarray) Current estimate of gyro bias
        """

        self.PHat = covarianceContainer(
            block_diag(
                attitudeErrorCovariance,
                gyroBiasCovariance
            ),
            covarianceStorage
        )
        """
        (covarianceContainer)  Current joint covariance matrix. 

        Upper 3x3 diagonal contains covariance of the attitude estimate (related to :attr:`qHat`), while lower 3x3 diagonal contains the covariance of the gyro bias :attr:`bHat`.
        """
        
        self.lastMeasID = None
        """ Last measurement used to generate measurement matrices
        """
        
        self.lastSourceID = None
        """ Last signal used to generate measurement matrices
        """
        
        self.lastMeasMat = None
        """ Last set of measurement matrices
        This allows class to avoid redundant computation of the same set of measurement matrices.
        """


        super().__init__(
            stateDimension=6,
            stateVectorHistory={
                't': t,
                'stateVector': np.append(np.zeros(3), self.bHat),
                'covariance': self.PHat,
                'aPriori': True,
                'q': self.qHat.q,
                'eulerAngles': self.eulerAngles(),
                'eulerSTD': self.eulerSTD(),
                'stateVectorID': -1
            },
            storeLastStateVectors=storeLastStateVectors
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


    def storeStateVector(
            self,
            svDict
            ):
        """
        storeStateVector is responsible for taking an updated version of
        the state vector, and storing it in the class variables.

        This function is designed to receive a time or measurement
        updated state vector and covariance, and store it.  This function is
        used by :class:`~modest.modularfilter.ModularFilter~ to store a
        jointly updated state.

        Depending on whether the state vector is the result of a time update
        (aPriori=True) or a measurement update (aPriori=False), the function
        may disregard the value of the attitude error state.  This is because
        this class handles the time-update of #qHat internally, so the updated
        attitude error state is only relevant after a measurement update.

        Args:
         svDict (dict): A dictionary containing the current state vector information, including the state vector, the covariance, the time, and whether the state vector is "a priori" or "a posteriori."
        See Also:
         :meth:`SubState.storeStateVector`

        Note:
         This function is one of mandatory functions required for :class:`Attitude` to function as a sub-state of :class:`~modest.modularfilter.ModularFilter`.

        """
        xPlus = svDict['stateVector']
        aPriori = svDict['aPriori']
        time = svDict['t']
        PPlus = svDict['covariance']

        # Only update the quaternion if the new state vector is the result of
        # a measurement update.  The attitude class is responsible for
        # time-updating the quaternion.
        
        if not aPriori:
            errorQ = Quaternion(
                array=np.array([
                    1.0,
                    xPlus[0]/2.0,
                    xPlus[1]/2.0,
                    xPlus[2]/2.0
                ]))

            # NOTE: The pyquaternion library uses backward notation from what is given in Markley and Crassidia.  So this is equivalent to q^+ = dq * qHat
            qPlus = self.qHat * errorQ 
            qPlus = qPlus.normalised

            self.qHat = qPlus
            self.bHat = xPlus[3:6]

        # self.eulerAngleVec.append({
        #     't': time,
        # })

        self.PHat = PPlus

        super().storeStateVector(
            {
                't': time,
                'stateVector': np.append(np.zeros(3), self.bHat),
                'covariance': self.PHat,
                'aPriori': aPriori,
                'q': self.qHat.q,
                'eulerAngles': self.eulerAngles(),
                'eulerSTD': self.eulerSTD(),
                'stateVectorID': svDict['stateVectorID']
            }
        )

        return
    

    def timeUpdate(
            self,
            dT,
            dynamics=None
    ):
        """
        timeUpdate returns the time-update matrices, and handles the
        internal time update of the attitude estimate #qHat.
    
        This function generates the time-update matrices F and Q, to be
        used for a time update, either locally or jointly as part of a
        State.ModularFilter.
    
        This function looks for angular velocity (omega) and bias variance in
        the dynamics dict, and uses these to construct the time update matrices.
        If these are not included in the dynamics dict, then the function
        assumes these values to be zero.
    
        This function also updates the attitude quaternion internally.  It does
        not update the covariance matrix however; this must be done externally.
        
        Args:
         dynamics (dict): A dict containing information about the dynamics.
    
        Returns:
         (dict) A dict containing the state transition matrix ("F") and the
         process noise matrix ("Q")
    
        See Also: 
         :meth:`SubState.timeUpdate`

        Note:
         This function is one of mandatory functions required for :class:`Attitude` to function as a sub-state of :class:`~modest.modularfilter.ModularFilter`.

        """
        
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
            omegaVar = 1e-100

        # Check dynamics dict for bias variance
        if (
                (dynamics is not None) and
                ('bias' in dynamics)
        ):
            biasVar = dynamics['bias']['var']
        else:
            biasVar = 1e-100

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

    def getMeasurementMatrices(
            self,
            measurement,
            source=None,
            useUnitVector=None
    ):
        """
        getMeasurementMatrices computes and returns measurement update
        matrices

        This function receives a dictionary containing a measurement,
        along with an object that contains the source model of the measurement.
        If the source is a Signals.PointSource type signal, then it generates
        unit-vector attitude measurement type matrices.  Otherwise, the function
        returns dicts populated with None.

        Args:
         measurement (dict): A dictionary containing measurement information
         source (`modest.modularfilter.signalsources.signalsource.SignalSource): The source object that produced the measurement

        Returns: 
         (dict) A dictionary containing the measurement matrices H, R, and dY

        See Also:
         :meth:`SubState.getMeasurementMatrices`

        Note:
         This function is one of mandatory functions required for :class:`Attitude` to function as a sub-state of :class:`~modest.modularfilter.ModularFilter`.

        """
            
        if (
                isinstance(source, PointSource)
        ):
            if useUnitVector is None:
                useUnitVector = self.useUnitVector
            if useUnitVector:
                measurementMatrices = self.unitVectorMeasurmentMatrices(
                    source,
                    measurement
                    )
                HDict = {'unitVector': measurementMatrices['H']}
                RDict = {'unitVector': measurementMatrices['R']}
                dyDict = {'unitVector': measurementMatrices['dY']}
            else:
                # print('using ra dec meas mat')
                measurementMatrices = self.RaDecMeasurementMatrices(
                    source,
                    measurement
                    )

                HDict = {'RaDec': measurementMatrices['H']}
                RDict = {'RaDec': measurementMatrices['R']}
                dyDict = {'RaDec': measurementMatrices['dY']}
            
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

    def quaternionTimeUpdateMatrix(
            self,
            myOmega,
            deltaT
    ):
        r"""
        quaternionTimeUpdateMatrix produces a time-update matrix for the attitude quaternion
    
        This function produces a 4x4 matrix which, when multiplied by
        an attitude quaternion, rotates the quaternion by an amount
        corresponding to the angular velocity and time ellapsed.  The attitude
        quaternion is updated as follows:
    
        .. math::
            \hat{\mathbf{q}}_k^- \approx{} \bar{\Theta}(\hat{\boldsymbol{\omega}}_j^+, \Delta T) \hat{\boldsymbol{q}}_j^+`

        where
        
        .. math::
           \bar{\Theta}( \hat{\boldsymbol{\omega}}_j^+, \Delta T ) =
           \begin{bmatrix}
             \textrm{cos} \left(\frac{1}{2}
             ||\hat{\boldsymbol{\omega}}_j^+|| \Delta t \right) I_3 -
             \left[\boldsymbol{\hat{\Psi}}_k^+ \times \right]
             & \left[\boldsymbol{\hat{\Psi}}_k^+ \times \right] \\
             - \left[\boldsymbol{\hat{\Psi}}_k^+ \times \right] &
             \textrm{cos} \left(\frac{1}{2} ||\mathbf{\hat{\omega}}_j^+|| \Delta t \right)
           \end{bmatrix}

        and

        .. math::
         \left[\boldsymbol{\hat{\Psi}}_k^+ \times \right] =
         \frac{
         \textrm{sin}\left(\frac{1}{2} || \mathbf{\hat{\omega}}_j^+ || \Delta t \right)
         \mathbf{\hat{\omega}}_j^+
         }{
         || \mathbf{\hat{\omega}}_j^+ ||
         }

        The matrix returned by this function is
        :math:`\bar{\Theta}(\mathbf{\hat{\omega}}_j^+, \Delta T)`.

        See Fundamentals of Spacecraft Attitude Determination and Control,
        Section 6.2.2, page 251, equation 6.60 for more details.

        Args:
         myOmega (numpy.ndarray): The angular velocity estimate used to update the attitude quaternion (:math:`\mathbf{\hat{\omega}}_k^-`)
         deltaT (float): The amount of time elapsed for the time-update, used for numerical integration of kinematics equation (:math:`\Delta T`)

        Returns:
         (numpy.ndarray): The quaternion time-update matrix :math:`\bar{\Theta}(\mathbf{\hat{\omega}}_j^+, \Delta T)`
        """
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

    def errorStateTimeUpdateMatrix(
            self,
            myOmega,
            deltaT
    ):
        r"""
        errorStateTimeUpdateMatrix produces a time-update matrix for the
        attitude error state
    
        This function the discrete-time error-state transition matrix.  This is
        the matrix which propagates the attitude error state covariance and gyro
        bias covariance forward in time based on time ellapsed and angular
        velocity estimate.
    
        The error-state transition matrix is defined as follows:
    
        .. math::
           \boldsymbol{\Phi} = \begin{bmatrix}
            \boldsymbol{\Phi}_{11} & \boldsymbol{\Phi}_{12} \\
            \boldsymbol{\Phi}_{21} & \boldsymbol{\Phi}_{22} \\
           \end{bmatrix}
    
        where
    
        .. math::
          \boldsymbol{\Phi}_{11} = \mathbf{I}_{3 \times 3} -
          \left[\mathbf{\hat{\omega}}_k^- \times \right]
          \frac
          {\textrm{sin}(||\mathbf{\hat{\omega}}_k^-]|| \Delta t)}
          {||\mathbf{\hat{\omega}}_k^-||} +
          \left[\mathbf{\hat{\omega}}_k^+ \times \right]^2
          \frac
          {1 - \textrm{cos}(1 - ||\mathbf{\hat{\omega}}_k^-|| \Delta t)}
          {||\mathbf{\hat{\omega}}_k^-||^2}
    
        .. math::
          \boldsymbol{\Phi}_{12} =
          \left[\mathbf{\hat{\omega}}_k^- \times \right]
          \frac
          {1 - \textrm{cos}(1 - ||\mathbf{\hat{\omega}}_k^-|| \Delta t)}
          {||\mathbf{\hat{\omega}}_k^-||^2} -
          \mathbf{I}_{3\times 3}\Delta t -
          \left[\mathbf{\hat{\omega}}_k^- \times \right]^2
          \frac
          {||\mathbf{\hat{\omega}}_k^-|| \Delta t -
          \textrm{sin}(||\mathbf{\hat{\omega}}_k^-|| \Delta t)}
          {||\mathbf{\hat{\omega}}_k^-||^3}
        
        .. math::
          \boldsymbol{\Phi}_{21} = \mathbf{0}_{3 \times 3}
        
        .. math::
          \boldsymbol{\Phi}_{22} = \mathbf{I}_{3\times3}
        
        See Fundamentals of Spacecraft Attitude Determination and Control,
        Section 6.2.4, page 258, equation 6.83 for more details and derivation.
    
        Args:
         myOmega (numpy.ndarray): The angular velocity estimate used to update the attitude quaternion (:math:`\mathbf{\hat{\omega}}_k^-`)
         deltaT (float):  The amount of time elapsed for the time-update, used for numerical integration of kinematics equation (:math:`\Delta T`)

        Returns:
         (numpy.ndarray) The error-state time update matrix, :math:`\boldsymbol{\Phi}`
        """        
        omegaNorm = norm(myOmega)
        omegaNormSquare = square(omegaNorm)
        omegaNormDT = omegaNorm * deltaT
        omegaSkew = self.skewSymmetric(myOmega)
        omegaSkewSquare = power(omegaSkew, 2)

        # Check to see if omega is very close to zero.  If it is, avoid error
        # dividing by zero.
        if omegaNorm > 1e-100:
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

    def processNoiseMatrix(
            self,
            deltaT,
            omegaVar,
            biasVar
            ):
        r"""
        processNoiseMatrix generates a the process noise matrix
    
        This function generates the process noise matrix for time update of
        attitude error covariance and gyro bias covariance.  The process noise
        matrix is a function propagation time, angular velocity noise, and gyro
        bias noise.  It is defined as follows:

        .. math::
         \mathbf{Q} = \begin{bmatrix}
         \left(\sigma_v^2 \Delta T + \frac{1}{3}\sigma_u^2 \Delta T^3\right) \mathbf{I}_{3\times 3} &
         -\left( \frac{1}{2} \sigma_u^2 \Delta T^2 \right) \mathbf{I}_{3\times 3} \\
         -\left( \frac{1}{2} \sigma_u^2 \Delta T^2 \right) \mathbf{I}_{3\times 3}&
         \left( \sigma_u^2 \Delta T \right) \mathbf{I}_{3\times 3}
         \end{bmatrix}
    
        where :math:`\sigma_v^2` is the angular velocity noise (i.e. gyro
        measurement noise) and :math:`\sigma_u^2` is the gyro bias process noise.
        
        See Fundamentals of Spacecraft Attitude Determination and Control,
        Section 6.2.4, page 260, equation 6.93 for derivation and more details.
        
        Args:
         deltaT (float): The amount of time corresponding to the time update (:math:`\Delta T`)
         omegaVar (float): The variance of the angular velocity (gyro) measurement (:math:`\sigma_v^2`)
         biasVar (float): The variance of the gias bias process noise, indicates how much the gyro bias changes over time (:math:`\sigma_u^2`)

        Returns:
         (np.ndarray) The comibined 6x6 process noise matrix (:math:`\mathbf{Q}`)
        """
        
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

    ## @fun RaDecMeasurementMatrices
    def RaDecMeasurementMatrices(
            self,
            source,
            measurement
    ):
        
        raDecRoll = self.RaDecRoll()
        raEst = raDecRoll[0]
        decEst = raDecRoll[1]
        rollEst = raDecRoll[2]
        
        sourceRaDec = source.RaDec()
        # # print("True Ra Dec:")
        # print(source.name)
        
        
        sourceUnitVec = self.sidUnitVec(sourceRaDec)
        
        modifiedEulerAngles = QuaternionHelperFunctions.quaternion2euler(self.qHat)
        modifiedEulerAngles[0] = 0 # Set roll to zero
        modifiedRotationMatrix = QuaternionHelperFunctions.euler2quaternion(
            modifiedEulerAngles).rotation_matrix.transpose()

        sourceUnitVecLocal = modifiedRotationMatrix.dot(sourceUnitVec)

        sourceRaLocal, sourceDecLocal = spacegeometry.unitVector2RaDec(
            sourceUnitVecLocal
        )
        # print("Local (unrotated) RA and Dec")
        # print(sourceRaLocal)
        # print(sourceDecLocal)

        predictedUnitVec = self.qHat.rotation_matrix.transpose().dot(sourceUnitVec)
        predictedRa, predictedDec = spacegeometry.unitVector2RaDec(predictedUnitVec)
        # print("Predicted (estimated) RA and Dec")
        # print(predictedRa)
        # print(predictedDec)
        # print('Estimated Ra, Dec, Roll')
        # print(raEst)
        # print(decEst)
        # print(rollEst)
        # raDiff = sourceRaDec['RA'] - raEst
        # while np.abs(raDiff) > np.pi:
        #     raDiff = raDiff - np.pi*np.sign(raDiff)
        # decDiff = sourceRaDec['DEC'] - decEst
        # while np.abs(decDiff) > np.pi/2:
        #     decDiff = decDiff - np.pi*np.sign(decDiff)/2

        
        raDiff = sourceRaLocal
        decDiff = sourceDecLocal
        
        # print('Source Ra, Dec')
        # print(sourceRaDec['RA'])
        # print(sourceRaDec['DEC'])
        
        # print('Ra, Dec Diff')
        # print(raDiff)
        # print(decDiff)
        
        
        sinTheta = np.sin(rollEst)
        cosTheta = np.cos(rollEst)
        # H = np.array(
        #     [
        #         [0, sinTheta, -cosTheta],
        #         [0, cosTheta, sinTheta]
        #     ]
        # )
        # H = np.array(
        #     [
        #         [-raDiff*sinTheta + decDiff*cosTheta, sinTheta, -cosTheta],
        #         [-raDiff*cosTheta - decDiff*sinTheta, cosTheta, sinTheta]
        #     ]
        # )
        # H = np.array(
        #     [
        #         [raDiff*sinTheta + decDiff*cosTheta, 0, -1],
        #         [-raDiff*cosTheta - decDiff*sinTheta, 1, 0]
        #     ]
        # )
        H = np.array(
            [
                [decDiff, 0, -1],
                [-raDiff, 1, 0]
            ]
        )
        # print(H)
        

        H = np.hstack([H, np.zeros([2,3])])
        # print(H)
        # predictedRa = (raDiff * cosTheta) - (decDiff * sinTheta)
        # predictedDec = (raDiff * sinTheta) + (decDiff * cosTheta)
        # predictedRa = (raDiff * cosTheta) + (decDiff * sinTheta)
        # predictedDec = -(raDiff * sinTheta) + (decDiff * cosTheta)
        # predictedRa = sourceRaLocal
        # predictedDec = sourceDecLocal
        dY = np.array([
            measurement['RA']['value'] - predictedRa,
            measurement['DEC']['value'] - predictedDec
        ])

        R = np.array([
            [measurement['RA']['var'], 0],
            [0, measurement['DEC']['var']]
        ])
        if np.isscalar(source.extent):
            R = R + np.eye(2) * np.square(source.extent)
        else:
            RTransform = np.array([
                [ np.cos(rollEst), -np.sin(rollEst)],
                [ np.sin(rollEst), np.cos(rollEst)]
            ])
            R = R + RTransform.dot(source.extent).dot(RTransform.transpose())
        
        measMatrices = {
            'H': H,
            'R': R,
            'dY': dY
            }
        
        return(measMatrices)
    
    
    ## @fun unitVectorMeasurmentMatrices generates measurement matrices for a angle
    # measurement of a point source.
    #
    # @details
    # This function generates the set of measurement matrices
    # \f$ \mathbf{H} \f$, \f$ \mathbf{dY} \f$, and \f$ \mathbf{R} \f$
    # corresponding to an inferred unit vector measurement from a set of two
    # angle measurments (local right ascension and declination of a point
    # source).
    #
    # The measurement matrices are a function of the measurement itself, the
    # source from which the measurement originated, and the current estimate
    # of attitude, #qHat. They are defined as follows:
    #
    # \f[
    # \mathbf{H}_k[\sv[aPriori=True,timeIndex=k]] =
    # A(\attVec[est=True,aPriori=True,t=k])
    # \unitVec[signalSource=S, frame=nav] \times
    # \f]
    #
    # \f[
    # \mathbf{dY} =
    # \measurementVec[S](\RADEC) -
    # A(\attVec[est=True,aPriori=True,t=k])
    # \unitVec[signalSource=S, frame=nav]
    # \f]
    #
    # \f[
    # \mathbf{R} =  \eye[3] \sigma^2_{\RADEC}
    # \f]
    #
    # The measured unit vector is a unit vector computed from the measured
    # angles, using #sidUnitVec.
    #
    # See Fundamentals of Spacecraft Attitude Determination and Control,
    # Section 6.2.4, page 257, Table 6.3 for more details and derivation.
    #
    # @note Currently, the measurement noise matrix is an identity matrix
    # mutiplied by the angle measurement error.  This is my interpretation of
    # Section 6.2.3 in FSADC.  However, if we derive the measurement noise
    # matrix using the usual EKF method, we get a different result which
    # results in an unstable estimator.  I'm not convinced I understand why
    # this method is right, or why the EKF method is unstable, but it works, so
    # we're using it for now.
    #
    # @param self The object pointer
    # @param source A Signals.PointSource object from which the measurement
    # was generated.
    # @param measurement A dictionary containing the right ascension and
    # declination measurements as sub-dictionaries, each with their own value
    # and variance.
    #
    # @returns A dictionary containing the measurement matrices H, R, and dY
    def unitVectorMeasurmentMatrices(
            self,
            source,
            measurement
            ):

        # RaDecTrue = source.RaDec()
        uTrue = source.unitVec()
        # uTrue = self.sidUnitVec(RaDecTrue)
        uMeas = self.sidUnitVec(
            {
                'RA': measurement['RA']['value'],
                'DEC': measurement['DEC']['value']
            }
        )
        
        estimatedAttitudeMatrix = self.qHat.rotation_matrix.transpose()
        # estimatedAttitudeMatrix = self.qHat.rotation_matrix

        uPred = estimatedAttitudeMatrix.dot(uTrue)

        # H = estimatedAttitudeMatrix.dot(self.skewSymmetric(uTrue))
        H = self.skewSymmetric(uPred)
        H = np.append(H, np.zeros([3, 3]), axis=1)

        varR = measurement['RA']['var'] + np.square(source.extent)
        # varD = measurement['DEC']['var']

        # measR = measurement['RA']['value']
        # measD = measurement['DEC']['value']

        # sinD = sin(measD)
        # cosD = cos(measD)

        # sinR = sin(measR)
        # cosR = cos(measR)

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
        # print(measMatrices)
        return(measMatrices)

    ## @fun eulerAngles computes the Euler angles (roll, pitch and yaw) based
    # on the current attitude.
    #
    # @details This function computes the Euler angles (or, technically the
    # "Tait-Bryan angles"), i.e. roll, pitch and yaw from the current attitude
    # quaternion #qHat.
    #
    # @note See Wikipedia's article on the
    # <a href="https://en.wikipedia.org/wiki/Euler_angles">
    # equatorial coordinate system</a> for more details.
    #
    # @param self The object pointer
    #
    # @returns A list containing the three angles
    def eulerAngles(self, t=None):
        if t is None:
            q = self.qHat
            phi = arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(square(q[1]) + square(q[2])))
            theta = arcsin(2 * ((q[0] * q[2]) - (q[3] * q[1])))
            psi = arctan2(2 * (q[0] * q[3] + q[1]*q[2]), 1 - 2*(square(q[2]) + square(q[3])))

        return [phi, theta, psi]

    ## @fun RaDecRoll returns the current attitude in terms of right ascension,
    # declination, and roll
    #
    # @details This function essentially computes the euler angles (1-2-3)
    # from the current attitude quaternion #qHat and returns them in reverse
    # order (3-2-1).  This is helpful because the attitude of spacecraft is
    # commonly expressed in terms of RA-Dec-Roll, rather than roll-pitch-yaw
    # which is more standard in other aerospace applications.
    #
    # @note See Wikipedia's article on the
    # <a href="https://en.wikipedia.org/wiki/Equatorial_coordinate_system">
    # equatorial coordinate system</a> for more details.
    #
    # @param self The object pointer
    #
    # @returns A list containing the three angles
    def RaDecRoll(self):
        eulerAngles = self.eulerAngles()

        return([eulerAngles[2], -eulerAngles[1], eulerAngles[0]])

    ## @fun sidUnitVec generates a unit vector from two angles
    #
    # @details
    # This function computes the unit vector in siderial coordinates from a
    # measurement of right ascension and declination. This is a "helper"
    # function; it doesn't really need to be included in the class and could
    # be moved to a seperate library probably.
    #
    # @param self Object pointer
    # @param RaDec A dictionary containing the two angles, in radians
    #
    # @returns A unit vector generated from the angles given
    def sidUnitVec(
            self,
            RaDec):
        if isinstance(RaDec['DEC'], dict) and isinstance(RaDec['RA'], dict):
            cosD = np.cos(RaDec['DEC']['value'])
            sinD = np.sin(RaDec['DEC']['value'])
            cosRA = np.cos(RaDec['RA']['value'])
            sinRA = np.sin(RaDec['RA']['value'])
        else:
            cosD = np.cos(RaDec['DEC'])
            sinD = np.sin(RaDec['DEC'])
            cosRA = np.cos(RaDec['RA'])
            sinRA = np.sin(RaDec['RA'])

        myUV = np.array([cosD * cosRA, cosD * sinRA, sinD])
        myUV = myUV / np.linalg.norm(myUV)
        return myUV
    
    ## @fun skewSymmetric generates a skew-symmetric matrix from a 3x1 vector
    #
    # @details
    # This function generates a skew symmetric matrix from a 3x1 vector.  It
    # is a "helper" function and doesn't actually need to be a member function.
    # It could (should?) be moved to its own library.
    #
    # @param self Object pointer
    # @param vector The vector
    #
    # @returns The skew symmetric matrix
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

    def eulerSTD(self):
        
        newCov = self.PHat.convertCovariance('covariance').value[0:3,0:3]
        # newCov = (
        #     self.qHat.rotation_matrix.transpose().dot(
        #         newCov
        #     ).dot(self.qHat.rotation_matrix)
        # )

        eulerSTD = np.sqrt(newCov.diagonal()[0:3])
        return eulerSTD
