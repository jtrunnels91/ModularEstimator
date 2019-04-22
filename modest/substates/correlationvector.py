## @file CorrelationVector
# This package contains the #CorrelationVector class

import numpy as np
#from numpy import sin, cos, arcsin, arccos, arctan2, square, sqrt, abs, power
import matplotlib.pyplot as plt
from . import substate
from .. modularfilter import ModularFilter
from . oneDimensionalPositionVelocity import oneDPositionVelocity
from .. signals.oneDimensionalObject import oneDObjectMeasurement
from .. utils import covarianceContainer
from scipy.linalg import block_diag
from scipy.special import factorial
from math import isnan

## @class CorrelationVector
# @brief CorrelationVector estimates the correlation vector and delay between
# a signal and a time-delayed measurement of that signal
#
# @details
# This class contains an estimator which estimates the correlation vector
# between a signal (the #__trueSignal__) and measurements of that signal.  This
# correlation vector is then used to estimate the delay between the
# #__trueSignal__ and the measurements of that signal.
#
# The estimator in this class currently assumes that the signal source is
# "distant," or infinitely far away.  This implies that the unit vector to the
# signal source is perfectly known, and not changing.  A later implementation
# could include the option of having a non-distant signal source, in which the
# unit vector is changing as a function of position and therefore uncertain.
#
# @note This class is essentially an implementation of the estimator presented in
# <a href="https://doi.org/10.2514/1.G002650">
# Recursive Range Estimation Using Astrophysical Signals of Opportunity</a>,
# J. Runnels, D. Gebre, Journal of Guidance, Control and Dynamics, 2017.  Some
# equations from the paper are included in the class documentation for
# reference.  A more detailed discussion and derivation of the estimator can
# be found in the journal article..

class CorrelationVector(substate.SubState):
    
    ## @fun #__init__ is responsible for initializing a correlation vector
    # estimator
    #
    # @details The primary function of the #__init__ method is to initialize
    # the correlation vector estimator, and store the relevant user inputs.  A
    # few key user inputs are required in order to initialize the filter.
    # Additionally, because the algorithm is relatively complicated, there are
    # a number of optional tuning parameters which may be inputed at
    # initialization.
    #
    # In general, the parameters which are required inputs are the ones that
    # are critical for initialization of the filter, and should not be changed
    # during the course of the filter's lifetime.  These inputs are stored as
    # "private" variables; indicating that the should not be changed during
    # the object's lifetime.
    #
    # The optional inputs, on the other hand, are inputs which are used in the
    # estimation functions (#timeUpdate, #getMeasurementMatrices, etc.).
    # These parameters could conceivably be changed during the lifetime of the
    # filter without causing problems, and the user may want to change them
    # depending on external factors.  These parameters are initalized with
    # default values, and stored as public variables that the user can in
    # theory change.
    #
    # There are also a set of class variables which are publicly accessable
    # and which hold the most recent state estimate.  These exist primarily
    # for convenience, and are never actually used within the class.
    # Modifying them will have no affect on the state estimates.  The only way
    # to modify a state estimate is through the #storeStateVector method.
    #
    # The #__init__ method also checks the user inputs to ensure that they are
    # consistent with how they will be used in the class (where applicable).
    #
    # The trueSignal input is checked to see whether it has the following
    # methods:
    # - flux()
    # - signalID()
    # - unitVec()
    #
    # @param trueSignal An object that describes the signal for which
    # correlation should be estimated.
    # @param filterOrder The number of bins or "taps" in the correlation vector
    # @param dT The sampling period, or time-step between bins in the
    # correlation vector
    #
    # @param #t (optional) The initial starting time.  If no value is passed,
    # initialized to zero by default.
    # @param #correlationVector (optional) The initial value of the
    # correlation vector.  If not supplied, the correlation vector will be
    # initialized based on the filter #__dT__ and maximum flux of
    # the #__trueSignal__.
    # @param #correlationVectorCovariance (optional) The initial value of the
    # correlation vector estimate covariance.  If not supplied, the covariance
    # matrix will be initialized based on the filter #__dT__ and maximum flux
    # of #__trueSignal__.
    # @param #signalDelay (optional) The initial estimate of delay between
    # the #__trueSignal__ and the signal measurements.  If not supplied,
    # #signalDelay is initialized to zero.
    # @param #delayVar (optional) The variance of the estimate of delay
    # @param #aPriori (optional) Indicates whether initial estimates are a
    # priori or a posteriori.  Default=True
    #
    # @param #centerPeak (optional) Boolean indicating whether the correlation
    # vector should be "shifted" after each update to keep the peak centered
    # at the zero index.  Default is True.
    # @param #peakFitPoints (optional) Number of points used on either side of
    # max for quadratic fit in #computeSignalDelay.  Minimum is 1, default is 1.
    # @param #processNoise (optional) Scalar term of additional process noise
    # added to covariance matrix in time update.  Default is 1e-12
    # @param #measurementNoiseScaleFactor (optional) Scale factor to inflate
    # the measurement noise. Default is 1.
    def __init__(
            self,
            trueSignal,
            filterOrder,
            dT,
            t=0,
            correlationVector=None,
            correlationVectorCovariance=None,
            signalTDOA=0, 
            TDOAVar=0,
            aPriori=True,
            centerPeak=True,
            peakFitPoints=1,
            processNoise=1e-12,
            measurementNoiseScaleFactor=1,
            peakLockThreshold=1,
            covarianceStorage='covariance',
            internalNavFilter=None,
            navProcessNoise=1,
            tdoaStdDevThreshold=None,
            velStdDevThreshold=None,
            tdoaNoiseScaleFactor=None,
            velocityNoiseScaleFactor=None,
            storeLastStateVectors=0,
            vInitial=None,
            aInitial=None,
            gradInitial=None,
            peakEstimator='EK'
            ):
        print('updated correlation filter')
        self.peakLockThreshold = peakLockThreshold
        self.peakCenteringDT = 0
        
        self.peakOffsetFromCenter = 0

        self.navProcessNoise = navProcessNoise
        """
        This is the default process noise that is injected into the navigation states as noise in the derivative of the highest order state.
        """
        
        ## @brief #__unitVecToSignal__ is a unit vector which points to the signal
        # source
        self.__unitVecToSignal__ = trueSignal.unitVec()
        
        ## @brief #__trueSignal__ is a signal object that contains the "true"
        # signal for which the correlation vector is being estimated
        self.__trueSignal__ = trueSignal

        ## @brief #__filterOrder__ is the number of "taps" in the estimated
        # correlation vector, #correlationVector.
        self.__filterOrder__ = filterOrder

        ## @brief #__dT__ is the "sample period" or "bin size" of the estimated
        # correlation vector
        self.__dT__ = dT

        ## @brief #t The current time
        self.t = t

        ## @brief #aPriori Indicates whether the current state vector is the
        # result of a time update (#aPriori = True) or a measurement update
        # (#aPriori = False)
        self.aPriori = aPriori

        if correlationVector is None:
            correlationVector = (
                np.ones(self.__filterOrder__) *
                self.__trueSignal__.peakAmplitude * self.__dT__
            )
        ## @brief #correlationVector is the current estimate of the
        # correlation vector between the incoming signal measurements and the
        # #__trueSignal__
        self.correlationVector = correlationVector

        if correlationVectorCovariance is None:
            if covarianceStorage == 'covariance':
                correlationVectorCovariance = (
                    np.eye(self.__filterOrder__) *
                    np.square(self.__trueSignal__.peakAmplitude * self.__dT__)
                )
            elif covarianceStorage == 'cholesky':
                correlationVectorCovariance = (
                    np.eye(self.__filterOrder__) *
                    self.__trueSignal__.peakAmplitude * self.__dT__
                )
        # Store the correlation vector covariance in a container
        ## @brief #correlationVectorCovariance is the covariance matrix of the
        # correlation vector estimate, #correlationVector
        self.correlationVectorCovariance = correlationVectorCovariance

        ## @brief #signalDelay is the current estimate of the delay between
        # the incoming signal measurements and the #__trueSignal__
        self.signalTDOA = signalTDOA
        
        ## @brief #delayVar is the variance of the signal delay estimate
        # #signalDelay
        self.TDOAVar = TDOAVar
        
        ## @brief #centerPeak indicates whether the correlation vector is
        # shifted to maintain the peak at the middle tap
        self.centerPeak = centerPeak

        ## @brief #peakLock indicates whether the current estimate of
        # correlation vector and peak location is accurate enough to "know"
        # that we've locked on to the correct peak.
        self.peakLock = False

        ## @brief #peakFitPoints is a variable which controls the number of
        # points used for quadratically estimating the location of the
        # correlation vector peak
        self.peakFitPoints = peakFitPoints

        ## @brief #processNoise is the scalar value used to generate an
        # additional process noise term in #timeUpdate.
        self.processNoise = processNoise

        ## @brief #measurementNoiseScaleFactor is a factor used to scale the
        # measurement noise matrix.  The default value is 1 (no scaling).
        self.measurementNoiseScaleFactor = measurementNoiseScaleFactor

        
        self.peakEstimator = peakEstimator
        """
        String that determines which algorithm is used to estimate peak.  Use either EK (extended Kalman Filter) or UK (Unscented)
        """
        
        self.__halfLength__ = int(np.ceil(self.__filterOrder__ / 2))
        self.__halfLengthSeconds__ = self.__halfLength__ * self.__dT__

        xAxis = np.linspace(0, self.__filterOrder__-1, self.__filterOrder__)
        self.xAxis = xAxis * self.__dT__
        
        self.__xVec__ = np.linspace(
                0, 
                self.peakFitPoints * 2, 
                (self.peakFitPoints * 2) + 1
                )

        self.internalNavFilter = internalNavFilter
        print(internalNavFilter)
        if self.internalNavFilter == 'none':
            self.internalNavFilter = None
            self.INF_type = 'none'
        elif self.internalNavFilter == 'deep':
            self.INF_type = 'deep'
        elif self.internalNavFilter:
            self.INF_type = 'external'
        else:
            self.internalNavFilter = None
            self.INF_type = 'none'

        stateVector = correlationVector
        svCovariance = correlationVectorCovariance
        
        if self.INF_type == 'deep':
            if not vInitial:
                raise ValueError('In order to use the deep internal navigation filter, you must initialize.  Filter expects to receive at least vInitial, but received None')

            self.velocity = vInitial['value']
            self.velocityStdDev = np.sqrt(vInitial['var'])
            if not aInitial:
                self.navVectorLength = 1
                navVector = np.zeros(1)
                navVector[0] = vInitial['value']

                navVar = np.zeros([1,1])
                navVar[0,0] = vInitial['var']
                
            elif not gradInitial:
                self.acceleration = aInitial['value']
                self.accelerationStdDev = np.sqrt(aInitial['var'])
                
                self.navVectorLength = 2
                navVector = np.zeros(2)
                navVector[0] = vInitial['value']
                navVector[1] = aInitial['value']

                navVar = np.zeros([2,2])
                navVar[0,0] = vInitial['var']
                navVar[1,1] = aInitial['var']
            else:
                self.acceleration = aInitial['value']
                self.accelerationStdDev = np.sqrt(aInitial['var'])
                
                self.navVectorLength = 3
                navVector = np.zeros(3)
                navVector[0] = vInitial['value']
                navVector[1] = aInitial['value']
                navVector[2] = gradInitial['value']

                navVar = np.zeros([3,3])
                navVar[0,0] = vInitial['var']
                navVar[1,1] = aInitial['var']
                navVar[2,2] = aInitial['var']
                
            stateVector = np.append(stateVector,navVector)
            svCovariance = block_diag(svCovariance, navVar)

        svCovariance = covarianceContainer(
            svCovariance,
            covarianceStorage
        )

        self.mostRecentF = np.eye(self.__filterOrder__)
        self.stateVector = stateVector
        super().__init__(
            stateDimension=len(stateVector),
            stateVectorHistory={
                't': t,
                'stateVector': stateVector,
                'covariance': svCovariance,
                'aPriori': aPriori,
                'signalTDOA': signalTDOA,
                'TDOAVar': TDOAVar,
                'xAxis': self.xAxis,
                'stateVectorID': -1
            },
            storeLastStateVectors=storeLastStateVectors
        )

        self.tdoaStdDevThreshold = tdoaStdDevThreshold
        self.velStdDevThreshold = velStdDevThreshold

        self.tdoaNoiseScaleFactor = tdoaNoiseScaleFactor
        self.velocityNoiseScaleFactor = velocityNoiseScaleFactor

        if self.INF_type == 'external':
            self.navState = self.internalNavFilter.subStates['oneDPositionVelocity']['stateObject']
        return


    ##
    # @name Mandatory SubState Functions
    # The following functions are required in order for this class to be used
    # as a substate in ModularFilter.  The inside of the functions may be
    # changed or updated, but their "black box" behavior must remain the
    # same; i.e. they must still perform the same essential functions and
    # return the same things.
    # @{

    ## @fun #storeStateVector stores an updated estimate of the state vector
    def storeStateVector(
            self,
            svDict
            ):
        # Unpack updated state vector values
        self.t = svDict['t']
        self.aPriori = svDict['aPriori']

        # Compute new estimate of delay based on new state vector, store in
        # svDict and local attributes
        if not svDict['aPriori']:
            self.correlationVector = svDict['stateVector'][0:self.__filterOrder__]
            self.correlationVectorCovariance = svDict['covariance']
            self.stateVector = svDict['stateVector']

            if self.peakEstimator == 'UK':
                tdoaDict = self.estimateSignalTDOA_UT(
                    self.correlationVector,
                    self.correlationVectorCovariance
                )

            elif self.peakEstimator == 'EK':
                tdoaDict = self.estimateSignalTDOA_EK(
                    self.correlationVector,
                    self.correlationVectorCovariance
                )
            else:
                raise ValueError('Unrecougnized peak finding algorithm %s' %self.peakEstimator)
            
            newTDOA = (
                (
                    tdoaDict['meanTDOA'] 
                ) *
                self.__dT__
            ) + self.peakCenteringDT
            
            newTDOAVar = tdoaDict['varTDOA'] * np.square(self.__dT__)
            if not isnan(newTDOA) and not isnan(newTDOAVar):
                self.signalTDOA = newTDOA
                self.TDOAVar = newTDOAVar

            svDict['signalTDOA'] = self.signalTDOA
            svDict['TDOAVar'] = self.TDOAVar

            if self.INF_type == 'external':
                if (
                        (np.sqrt(self.TDOAVar) < (self.tdoaStdDevThreshold))
                        or (self.tdoaStdDevThreshold == 0)
                ):
                    self.internalNavFilter.measurementUpdateEKF(
                        {'position': {
                            'value': self.signalTDOA,
                            'var': self.TDOAVar * self.tdoaNoiseScaleFactor
                        }},
                        'oneDPositionVelocity'
                    )
                else:
                    self.internalNavFilter.measurementUpdateEKF(
                        {}, ''
                    )
                
            if self.peakLock is True and self.centerPeak is True:
                self.peakOffsetFromCenter = tdoaDict['meanTDOA'] - self.__halfLength__ + 1
                # self.peakOffsetFromCenter = np.mod(tdoaDict['meanTDOA'], self.__dT__)
                # print(self.peakOffsetFromCenter)
            else:
                self.peakOffsetFromCenter = 0

        else:
            
#            if self.peakLock is True and self.centerPeak is True:
#                svDict['stateVector'] = self.correlationVector
#            else:
                
            # self.correlationVector = svDict['stateVector']
            # if self.peakOffsetFromCenter != 0:
            #     FLDict = self.buildFLMatrices(
            #         -self.peakOffsetFromCenter*self.__dT__,
            #         self.correlationVector
            #     )
            #     self.correlationVector = FLDict['F'].dot(self.correlationVector)
            #     self.peakOffsetFromCenter = 0
            
            # self.correlationVector = svDict['stateVector'][0:self.__filterOrder__]
            self.correlationVector = self.mostRecentF.dot(self.correlationVector)
            svDict['stateVector'][0:self.__filterOrder__] = self.correlationVector
            self.stateVector = svDict['stateVector']
            self.correlationVectorCovariance = svDict['covariance']

            if self.peakEstimator == 'UK':
                tdoaDict = self.estimateSignalTDOA_UT(
                    self.correlationVector,
                    self.correlationVectorCovariance
                )

            elif self.peakEstimator == 'EK':
                tdoaDict = self.estimateSignalTDOA_EK(
                    self.correlationVector,
                    self.correlationVectorCovariance
                )
            else:
                raise ValueError('Unrecougnized peak finding algorithm %s' %self.peakEstimator)
            # newTDOA = (
            #     (
            #         tdoaDict['meanTDOA'] 
            #     ) *
            #     self.__dT__
            # ) + self.peakCenteringDT
            
            newTDOAVar = tdoaDict['varTDOA'] * np.square(self.__dT__)

            # self.signalTDOA = newTDOA
            self.TDOAVar = newTDOAVar
            
            svDict['signalTDOA'] = self.signalTDOA
            svDict['TDOAVar'] = self.TDOAVar
            self.peakOffsetFromCenter = 0

            
        svDict['xAxis'] = self.xAxis + self.peakCenteringDT
        # svDict['xAxis'] = self.xAxis - self.signalTDOA
        
        tdoaSTD = np.sqrt(self.TDOAVar)
        if tdoaSTD < (self.peakLockThreshold * self.__dT__):
            if not self.peakLock:
                print(
                    'Substate %s reached peak lock at time %s'
                    %(self.__trueSignal__.name, self.t)
                )
            self.peakLock = True
        else:
            if self.peakLock and tdoaSTD > (self.peakLockThreshold * self.__dT__ * 1.1):
                print(
                    'Substate %s lost peak lock at time %s'
                    %(self.__trueSignal__.name, self.t)
                )
                self.peakLock = False
                self.peakOffsetFromCenter = 0

        if self.INF_type == 'deep':
            fO = self.__filterOrder__
            currentV = self.stateVector[fO]
            currentVStdDev = np.sqrt(self.correlationVectorCovariance[fO,fO].value)
            self.velocity = currentV
            self.velocityStdDev = currentVStdDev
            svDict['velocity'] = {'value':currentV, 'stddev': currentVStdDev}
            if self.navVectorLength == 2:
                currentA = self.stateVector[fO+1]
                currentAStdDev = np.sqrt(self.correlationVectorCovariance[fO+1,fO+1].value)
                svDict['acceleration'] = {'value':currentA, 'stddev': currentAStdDev}

                self.acceleration = currentA
                self.accelerationStdDev = currentAStdDev
            elif self.navVectorLength == 3:
                currentA = self.stateVector[fO+1]
                currentAStdDev = np.sqrt(self.correlationVectorCovariance[fO+1,fO+1].value)
                svDict['acceleration'] = {'value':currentA, 'stddev': currentAStdDev}
                self.acceleration = currentA
                self.accelerationStdDev = currentAStdDev
                
                currentGrad = self.stateVector[fO+2]
                currentGradStdDev = np.sqrt(self.correlationVectorCovariance[fO+2,fO+2].value)
                svDict['aGradient'] = {'value':currentGrad, 'stddev': currentGradStdDev}
                self.aGrad = currentGrad
                self.aGradStdDev = currentGradStdDev
                
        elif self.INF_type == 'external':
            self.velocity = self.navState.currentVelocity
            self.velocityStdDev = np.sqrt(self.navState.velocityVar)
                
        super().storeStateVector(svDict)
        return

    ## @fun #timeUpdate returns the matrices for performing the correlation
    # vector time update.
    #
    # @details This function calls the #buildTimeUpdateMatrices method to
    # generate the time-update matrices.
    #
    # @param self The object pointer
    # @param dT The amount of time ellapsed over which the time update is to
    # be performed
    # @param dynamics A dictionary containing the dynamics for the time update
    # (e.g. velocity)
    #
    # @sa SubStates.SubState.timeUpdate
    def timeUpdate(
            self,
            dT,
            dynamics=None
            ):

        if self.INF_type != 'deep':

            timeUpdateMatrices = self.buildTimeUpdateMatrices(
                dT, dynamics, self.correlationVector
            )

            L = timeUpdateMatrices['L']
            Q = timeUpdateMatrices['Q']

            Qmat = (
                np.outer(L, L) * Q +
                (
                        np.eye(self.__filterOrder__) * 
                              self.processNoise * dT * 
                              np.square(self.__trueSignal__.peakAmplitude * self.__dT__)
                )
            )

            if dynamics is not None and 'acceleration' in dynamics:
                oneDAcceleration = (
                    dynamics['acceleration']['value'].dot(self.__unitVecToSignal__) /
                    self.speedOfLight()
                )

                oneDAccelerationVar = (
                    self.__unitVecToSignal__.dot(
                        dynamics['acceleration']['value'].dot(
                            self.__unitVecToSignal__.transpose()
                        )
                    ) / np.square(self.speedOfLight())
                )
            else:
                oneDAcceleration = 0
                oneDAccelerationVar = self.navProcessNoise

            if self.INF_type == 'external':
                self.internalNavFilter.timeUpdateEKF(
                    dT,
                    dynamics = {
                        'oneDPositionVelocityacceleration': {
                            'value': oneDAcceleration,
                            'var': oneDAccelerationVar
                        }
                    }
                )

        else:
            timeUpdateMatrices = self.buildDeepTimeUpdateMatrices(dT, dynamics, self.correlationVector)
            # if dynamics is not None and 'accelerationGrad' in dynamics:
            #     navProcessNoise = (
            #         dynamics['accelerationGrad']['value'].dot(self.__unitVecToSignal__) /
            #         self.speedOfLight()
            #     )

            #     oneDAccelerationGradVar = (
            #         self.__unitVecToSignal__.dot(
            #             dynamics['accelerationGrad']['value'].dot(
            #                 self.__unitVecToSignal__.transpose()
            #             )
            #         ) / np.square(self.speedOfLight())
            #     )
            # else:
            
            L = timeUpdateMatrices['L']
            Qmat = (
                np.outer(L, L) * self.navProcessNoise  + (
                    (
                        block_diag(np.eye(self.__filterOrder__),np.zeros([self.navVectorLength,self.navVectorLength])) * 
                        self.processNoise * dT * 
                        np.square(self.__trueSignal__.flux * self.__dT__)
                    )
                )
            )
        self.mostRecentF = timeUpdateMatrices['F'][0:self.__filterOrder__, 0:self.__filterOrder__]
        return {'F': timeUpdateMatrices['F'], 'Q': Qmat}

    def buildDeepTimeUpdateMatrices(self,dT, dynamics, h):
        
        FMatrixShift = -self.peakOffsetFromCenter

        filterOrder = self.__filterOrder__
        
        # Initialize empty matricies
        F = np.zeros([filterOrder + self.navVectorLength, filterOrder+self.navVectorLength])
        
        halfLength = self.__halfLength__
        indexDiff = dT/self.__dT__
        
        peakShift = self.stateVector[self.__filterOrder__] * indexDiff

        # Velocity term
        self.peakCenteringDT = (
            self.peakCenteringDT + self.stateVector[self.__filterOrder__] * dT 
        )
        
        if self.navVectorLength > 1:
            # Acceleration term (if acceleration is being estimated)
            self.peakCenteringDT = (
                self.peakCenteringDT +
                self.stateVector[self.__filterOrder__ + 1] * np.power(dT,2)/2
            )
            peakShift = (
                peakShift + self.stateVector[self.__filterOrder__ + 1]*np.power(indexDiff,2)/2
            )
            
        if self.navVectorLength > 2:
            # Acceleration gradient term
            self.peakCenteringDT = (
                self.peakCenteringDT +
                self.stateVector[self.__filterOrder__] *
                self.stateVector[self.__filterOrder__ + 2] *
                np.power(dT,3)/6
            )
            peakShift = (
                peakShift +
                self.stateVector[self.__filterOrder__] *
                self.stateVector[self.__filterOrder__ + 2] *
                np.power(indexDiff,3)/6
            )
            
            
        self.peakCenteringDT = self.peakCenteringDT + (self.peakOffsetFromCenter*self.__dT__)
        
        # Build arrays of indicies from which to form the sinc function

        if np.mod(filterOrder, 2) == 0:
            baseVec = (
                np.linspace(
                1 - halfLength,
                    halfLength,
                    filterOrder
                )
            )

        else:
            baseVec = (
                np.linspace(
                    1 - halfLength,
                    halfLength - 1,
                    filterOrder
                )
            )

        # Compute the sinc function of the base vector
        sincBase = np.sinc(baseVec + FMatrixShift)
        diffBase = np.zeros_like(sincBase)

        for i in range(len(baseVec)):
            diffBase[i] = self.sincDiff(baseVec[i] + peakShift)

        sincBase = np.roll(sincBase, 1 - int(halfLength))
        diffBase = np.roll(diffBase, 1 - int(halfLength))

        for i in range(len(sincBase)):
            currentDiff = np.roll(diffBase, i).dot(h)
            F[i, 0:filterOrder] = np.roll(sincBase, i)
            F[i, filterOrder] = currentDiff * indexDiff
            
            if self.navVectorLength > 1:
                F[i, filterOrder+1] = currentDiff * np.power(indexDiff, 2)/2
                if self.navVectorLength > 2:
                    F[i, filterOrder+2] = (
                        currentDiff *
                        self.stateVector[filterOrder] *
                        np.power(indexDiff, 3)/6
                    )

        L = np.zeros(filterOrder+self.navVectorLength)
        
        if self.navVectorLength == 1:
            F[filterOrder, filterOrder] = 1
            L[filterOrder] = dT
            
        elif self.navVectorLength == 2:
            F[filterOrder, filterOrder] = 1.0
            F[filterOrder, filterOrder+1] = dT
            F[filterOrder+1, filterOrder+1] = 1.0
            
            L[filterOrder] = np.power(dT, 2)/2
            L[filterOrder + 1] = dT
            
        elif self.navVectorLength == 3:
            vCurrent = self.stateVector[filterOrder]
            aCurrent = self.stateVector[filterOrder + 1]
            gradCurrent = self.stateVector[filterOrder + 2]
            
            F[filterOrder,filterOrder] = 1 + (gradCurrent * np.power(dT,2)/2)
            F[filterOrder,filterOrder+1] = dT
            F[filterOrder,filterOrder+2] = vCurrent * np.power(dT,2)/2
            
            F[filterOrder+1,filterOrder+1] = 1
            F[filterOrder+1,filterOrder+2] = vCurrent * dT
            
            F[filterOrder+2,filterOrder+2] = 1

            L[filterOrder] = vCurrent * np.power(dT,3)/6
            L[filterOrder + 1] = vCurrent * np.power(dT,2)/2
            L[filterOrder + 2] = dT
            
        
        
        diffBase = np.zeros_like(sincBase)

        for i in range(len(baseVec)):
            diffBase[i] = self.sincDiff(baseVec[i])

        diffBase = np.roll(diffBase, 1 - int(halfLength))
        
        for i in range(len(baseVec)):
            L[i] = (
                np.roll(diffBase, i).dot(h) *
                np.power(indexDiff,self.navVectorLength+1)/factorial(self.navVectorLength+1)
            )
        
        # # Setting L to zero for test purposes only
        # L = np.zeros(filterOrder+self.navVectorLength)

        return({'F':F, 'L':L})
        
    def getMeasurementMatrices(
            self,
            measurement,
            source=None
    ):
        if (
                (source.signalID() == self.__trueSignal__.signalID()) and
                ('t' in measurement)
        ):

            measurementMatrices = self.getTOAMeasurementMatrices(
                measurement,
                self.correlationVector
            )

            HDict = {'correlationVector': measurementMatrices['H']}
            RDict = {'correlationVector': measurementMatrices['R']}
            dyDict = {'correlationVector': measurementMatrices['dY']}
        else:
            HDict = {'': None}
            RDict = {'': None}
            dyDict = {'': None}

        measurementMatricesDict = {
            'H': HDict,
            'R': RDict,
            'dY': dyDict
            }

        return measurementMatricesDict

    ## @}

    ## @fun #buildTimeUpdateMatrices constructs the correlation vector time
    # update matrices
    #
    # @details The #buildTimeUpdateMatrices method constructs the matrices required to perform the time update of the correlation vector sub-state.
    #
    # The time update matrices are a function of the estimated spacecraft velocity (\f$\mathbf{v}\f$), velocity variance (\f$\mathbf{P}_{\mathbf{v}}\f$), and the elapsed time over which the time update occurs (\f$\Delta T\f$).  The matrices are constructed as follows:
    #
    # \f[
    # \mathbf{F}_{j \to k} = \begin{bmatrix}
    # \textrm{sinc}(\hat{\delta}) & \hdots & \textrm{sinc}(\hat{\delta} + N - 1) \\
    # \vdots & \ddots & \vdots \\
    # \textrm{sinc}(\hat{\delta} - N + 1) & \hdots & \textrm{sinc}(\hat{\delta})
    # \end{bmatrix}
    # \f]
    # 
    # \f[
    # \mathbf{L}_{j} = \begin{bmatrix}
    # \frac{\textrm{cos}}{(\hat{\delta})} - \frac{\textrm{sin}}{(\hat{\delta}^2)}   & \hdots \\
    # \vdots & \ddots  \\
    # \end{bmatrix} \sv[timeIndex = k]
    # \f]
    #
    # \f[
    # Q_{\delta} = \left(\frac{(k-j)}{c}\right)^2
    # {\LOSVec[S]}^T \mathbf{P}_{\mathbf{v}} \LOSVec[S]
    # \f]
    #
    # where
    #
    # \f[
    # \hat{\delta}_{j \to k} = \frac{\mathbf{v} \LOSVec[S] \Delta T}{c T}
    # \f]
    #
    # @param self The object pointer
    # @param deltaT The amount of time over which the time update is occuring
    # @param dynamics A dictionary containing the relevant dynamics for the
    # time update
    # @param h The current correlation vector
    #
    # @returns A dictionary containing the matrices \f$\mathbf{F}\f$,
    # \f$\mathbf{L}\f$, and the scalar \f$Q\f
    def buildTimeUpdateMatrices(
            self,
            deltaT,
            dynamics,
            h
    ):
        
        indexDiff = deltaT/self.__dT__
            
        if (
                (dynamics is not None and 'velocity' in dynamics) or
                (
                    self.INF_type == 'external' and
                    (
                        (np.sqrt(self.navState.velocityVar) < self.velStdDevThreshold) or
                        self.velStdDevThreshold == 0
                    )
                )
        ):
            if 'velocity' in dynamics:

                velocity = dynamics['velocity']['value']
                vVar = dynamics['velocity']['var'] * self.velocityNoiseScaleFactor


                peakShift = (
                    (velocity.dot(self.__unitVecToSignal__) * indexDiff) /
                    self.speedOfLight()
                )

                # velocityTDOA = peakShift * self.__dT__
                velocityTDOA = (
                    velocity.dot(self.__unitVecToSignal__) * deltaT /
                    self.speedOfLight()
                )
                Q = (
                    self.__unitVecToSignal__.dot(
                        vVar
                    ).dot(self.__unitVecToSignal__) *
                    np.square(indexDiff / self.speedOfLight())
                )
                tdoaQ = (
                    self.__unitVecToSignal__.dot(vVar
                    ).dot(self.__unitVecToSignal__) *
                    np.square(deltaT/self.speedOfLight()))
            elif self.INF_type=='external':

                peakShift = self.navState.currentVelocity * indexDiff
                velocityTDOA = self.navState.currentVelocity * deltaT
                Q = self.navState.velocityVar * np.square(indexDiff) * self.velocityNoiseScaleFactor
                tdoaQ = self.navState.velocityVar * np.square(deltaT) * self.velocityNoiseScaleFactor

        else:
            velocityTDOA = 0
            peakShift = 0
            Q = self.navProcessNoise * np.power(indexDiff,4)/4
            
            tdoaQ = self.navProcessNoise * np.power(deltaT,4)/4

        FMatrixShift = -self.peakOffsetFromCenter # - peakShift
        self.signalTDOA = (
            self.signalTDOA +
            velocityTDOA
        )
        self.TDOAVar = self.TDOAVar + tdoaQ

        self.peakCenteringDT = (
            self.peakCenteringDT + velocityTDOA  +
            (self.peakOffsetFromCenter*self.__dT__)
        )

        # Initialize empty matricies
        F = np.zeros([self.__filterOrder__, self.__filterOrder__])
        L = np.zeros([self.__filterOrder__, self.__filterOrder__])

        # Build arrays of indicies from which to form the sinc function

        if np.mod(self.__filterOrder__, 2) == 0:
            baseVec = (
                np.linspace(
                    1 - self.__halfLength__,
                    self.__halfLength__,
                    self.__filterOrder__
                )
            )

        else:
            baseVec = (
                np.linspace(
                    1 - self.__halfLength__,
                    self.__halfLength__ - 1,
                    self.__filterOrder__
                )
            )

        # Compute the sinc function of the base vector
        sincBase = np.sinc(baseVec + FMatrixShift)
        diffBase = np.zeros_like(sincBase)
        
        for i in range(len(baseVec)):
            diffBase[i] = self.sincDiff(baseVec[i] + peakShift)
            
        sincBase = np.roll(sincBase, 1 - int(self.__halfLength__))
        diffBase = np.roll(diffBase, 1 - int(self.__halfLength__))

        for i in range(len(F)):
            F[i] = np.roll(sincBase, i)
            L[i] = np.roll(diffBase, i)
        L = L.dot(h)

        # else:
        #     # If no velocity was included in dynamics, then do nothing during
        #     # time update
        #     F = np.eye(self.__filterOrder__)
        #     L = np.zeros(self.__filterOrder__)
        #     Q = 0
        
        timeUpdateDict = {
            'F': F,
            'L': L,
            'Q': Q
        }
        
        return(timeUpdateDict)

    def buildFLMatrices(self, peakShift, h):
        # Initialize empty matricies
        F = np.zeros([self.__filterOrder__, self.__filterOrder__])
        L = np.zeros([self.__filterOrder__, self.__filterOrder__])

        # Build arrays of indicies from which to form the sinc function

        if np.mod(self.__filterOrder__, 2) == 0:
            baseVec = (
                np.linspace(
                    1 - self.__halfLength__,
                    self.__halfLength__,
                    self.__filterOrder__
                )
            )

        else:
            baseVec = (
                np.linspace(
                    1 - self.__halfLength__,
                    self.__halfLength__ - 1,
                    self.__filterOrder__
                )
            )

        # Compute the sinc function of the base vector
        sincBase = np.sinc(baseVec + peakShift)
        diffBase = np.zeros_like(sincBase)

        for i in range(len(baseVec)):
            diffBase[i] = self.sincDiff(baseVec[i] + peakShift)
            
        sincBase = np.roll(sincBase, 1 - int(self.__halfLength__))
        diffBase = np.roll(diffBase, 1 - int(self.__halfLength__))

        for i in range(len(F)):
            F[i] = np.roll(sincBase, i)
            L[i] = np.roll(diffBase, i)

        L = L.dot(h)

        return {'F':F, 'L':L}

    ## @}
    
    ## @{
    # @name Functions Specific to #CorrelationVector
    #
    # The following remaining functions are not required in order for this
    # class to be used as a SubState, and may be changed as needed,
    # including inputs and outputs.
    def getTOAMeasurementMatrices(
            self,
            measurement,
            corrVec
    ):
        photonTOA = measurement['t']['value']
        
        adjustedTOA = photonTOA + self.peakCenteringDT
        
        H = np.eye(self.__filterOrder__)

        if self.INF_type == 'deep':
            H = np.append(H, np.zeros([self.__filterOrder__, self.navVectorLength]), axis=1)
        timeVector = np.linspace(
            0,
            (self.__filterOrder__ - 1),
            self.__filterOrder__
        )
        timeVector = timeVector * self.__dT__

        timeVector = (
            timeVector + adjustedTOA
            )

        # if self.peakLock is True:
        #     timeVector = timeVector - self.signalDelay

        signalTimeHistory = np.zeros(self.__filterOrder__)
        halfDT = self.__dT__/2.0
#        for timeIndex in range(len(timeVector)):
#            signalTimeHistory[timeIndex] = (
#                    self.__trueSignal__.getSignal(timeVector[timeIndex]) *
#                                                 self.__dT__
#             )
        for timeIndex in range(len(timeVector)):
            signalTimeHistory[timeIndex] = (
                self.__trueSignal__.signalIntegral(
                    timeVector[timeIndex]-halfDT,
                    timeVector[timeIndex] + halfDT
                )
            )
        # plt.plot(signalTimeHistory)
        # plt.show(block=False)
        # 1/0
        # print(corrVec)
        # print(signalTimeHistory)
        dY = signalTimeHistory - corrVec

        R = (
            np.eye(self.__filterOrder__) *
            #self.__trueSignal__.flux *
            self.__trueSignal__.peakAmplitude *
            self.__dT__ *
            np.dot(corrVec, corrVec) *
            self.measurementNoiseScaleFactor
        )

        measMatDict = {
            'H': H,
            'dY': dY,
            'R': R
            }
        
        return measMatDict

    ## @fun #computeSignalTDOA computes the delay between the #__trueSignal__ and
    # measurements based on a correlation vector
    #
    # @details The #computeSignalDelay function is a rudimentary function
    # which takes an estimate of the correlation vector and uses it to
    # estimate the location of the peak.  It functions by finding the tap with
    # the maximum value, and then fitting a quadratic to the points
    # surrounding the maximum value tap.  The number of points to which the
    # quadratic is fitted is determined by the value of #peakFitPoints; the
    # number of points is equal to \f$2 * n + 1\f$ where \f$n = \f$
    # #peakFitPoints.
    #
    # The delay estimate that is returned is in units of #__dT__.  So, a returned
    # value of 2 would imply that the peak is located at 2, and therefore the
    # delay corresponding to the correlation vector is 2 #__dT__.
    #
    # The returned delay may not include previously accumulated #signalDelay
    # between the signal and the measurements.  See the #storeStateVector
    # function for more information on how the #signalDelay is stored and
    # accumulated delay is accounted for.
    #
    # @param self The object pointer
    # @param c The correlation vector
    # @param P the correlation vector covariance matrix
    #
    # @return The estimate of the delay
    def computeSignalTDOA(
            self,
            c,
            P
    ):

        # First estimate of peak location is the location of the max value
        peakLocation = np.argmax(c)

        # Next, we "roll" the correlation vector so that the values being
        # fitted quadratically are the first 2 * peakFitPoints + 1 values

        lowerBound = peakLocation - self.peakFitPoints
        upperBound = lowerBound + (self.peakFitPoints * 2) + 1
        if (lowerBound < 0) or (upperBound > self.__filterOrder__):
            mySlice = range(lowerBound, upperBound)
            slicedC = c.take(mySlice, mode='wrap')
            slicedP = P.take(mySlice, axis=0, mode='wrap').take(mySlice, axis=1, mode='wrap')
        else:
            mySlice = slice(lowerBound, upperBound)
            slicedC = c[mySlice]
            slicedP = P[mySlice, mySlice]

        # xVec is the vector of "x" values corresponding the "y" values to
        # which the quadratic is being fit.
        xVec = self.__xVec__
#        xVec = xVec - rollFactor
        xVec = xVec + lowerBound

        # Get the quadratic function that fits the peak and surrounding values,
        # and use it to estimate the location of the max
        # print(slicedC)
        if len(xVec) == 3:
            TDOA = self.peakFinder(xVec, slicedC)
        else:
            quadraticVec = self.quadraticFit(xVec, slicedC)
            try:
                TDOA = (-quadraticVec[1] / (2 * quadraticVec[0]))
            except:
                TDOA = xVec[peakLocation]

        return TDOA

    ## @fun #estimateSignalTDOA_UT uses a unscented tranform to estimate the
    # delay corresponding to a correlation vector
    #
    # @details The #estimateSignalDelayUT method is responsible for computing
    # the estimated value of delay corresponding to a correlation vector, as
    # well as the variance of that estimate.  These values are computed using
    # a unscented transform (i.e. sigma-point) approach.
    #
    # The method receives the an estimate of the correlation vector, as well
    # as the covariance matrix corresponding to that vector.  From there it
    # computes a set of n sigma points (where n is the length of the
    # correlation vector), and for each of the generated sigma point vectors,
    # it computes the peak location using the #computeSignalDelay method.
    #
    # @param self The object pointer
    # @param h The correlation vector
    # @param P The correlation vector covariance matrix
    #
    # @returns A dict containing the estimate of the peak location
    # ("meanDelay") and the estimate variance ("varDelay")
    def estimateSignalTDOA_UT(
            self,
            h,
            P,
            useMean=True
    ):
        # Compute sigma points
        hDimension = len(h)

        maxHIndex = np.argmax(h)
        rollAmount = -maxHIndex + self.__halfLength__
        # rollAmount = 1
        # hRolled = np.roll(h, rollAmount)
        
        # PRolled = np.roll(np.roll(P.value, rollAmount, axis=0), rollAmount, axis=1)
        # Compute the square root of P.
        if P.form == 'covariance':
            sqrtP = np.linalg.cholesky(
                hDimension * P.value[0:self.__filterOrder__, 0:self.__filterOrder__]
            )
        elif P.form == 'cholesky':
            # PVal = P.convertCovariance('covariance').value
            # sqrtP = np.linalg.cholesky(hDimension * PVal)
            
            sqrtP = P.value[0:self.__filterOrder__, 0:self.__filterOrder__] * np.sqrt(hDimension)
                
        sigmaPoints = h + np.append(sqrtP, -sqrtP, axis=0)

        # Append one more row of sigma points containing the unmodified estimate
        sigmaPoints = np.append(np.array([h]), sigmaPoints, axis=0)

        # Initiate vector to store the resulting peaks from each sigma point
        sigmaPointResults = np.zeros(len(sigmaPoints))

        # Compute the peak corresponding to each sigma point vector
        for i in range(len(sigmaPoints)):
            sigmaPointResults[i] = (
                self.computeSignalTDOA(sigmaPoints[i], P.convertCovariance('covariance').value)
            )
            

        
        #meanTDOA = np.mean(sigmaPointResults)
        meanTDOA = sigmaPointResults[0]
        for i in range(len(sigmaPoints)):
            if (meanTDOA - sigmaPointResults[i]) > self.__halfLength__:
                sigmaPointResults[i] += self.__dimension__
            elif (sigmaPointResults[i] - meanTDOA) > self.__halfLength__:
                sigmaPointResults[i] -= self.__dimension__
        
        # meanTDOA = self.computeSignalTDOA(h, P)
        varTDOA = np.var(sigmaPointResults)

        return {'meanTDOA': meanTDOA, 'varTDOA': varTDOA, 'sigmaPoints': sigmaPointResults}

    def estimateSignalTDOA_EK(self, h, P):

        if P.form == 'covariance':
            P = P.value[0:self.__filterOrder__, 0:self.__filterOrder__]

        elif P.form == 'cholesky':
            P = P.convertCovariance('covariance').value[0:self.__filterOrder__, 0:self.__filterOrder__]
        
        # First estimate of peak location is the location of the max value
        peakLocation = np.argmax(h)

        # Next, we "roll" the correlation vector so that the values being
        # fitted quadratically are the first 3 values

        lowerBound = peakLocation - 1
        upperBound = lowerBound + (1 * 2) + 1
        if (lowerBound < 0) or (upperBound > self.__filterOrder__):
            mySlice = range(lowerBound, upperBound)
            slicedC = h.take(mySlice, mode='wrap')
            slicedP = P.take(mySlice, axis=0, mode='wrap').take(mySlice, axis=1, mode='wrap')
        else:
            mySlice = slice(lowerBound, upperBound)
            slicedC = h[mySlice]
            slicedP = P[mySlice, mySlice]

        # xVec is the vector of "x" values corresponding the "y" values to
        # which the quadratic is being fit.
        xVec = self.__xVec__
        xVec = xVec + lowerBound

        # Get the quadratic function that fits the peak and surrounding values,
        # and use it to estimate the location of the max
        TDOA = self.peakFinder(xVec, slicedC)
        jacobian = self.peakFinderJacobian(xVec, slicedC)

        variance = jacobian.dot(slicedP).dot(jacobian.transpose())
        
        return {'meanTDOA': TDOA, 'varTDOA': variance}

    
    def speedOfLight(
            self
    ):
        return (299792)

    @staticmethod
    def sincDiff(x):
        if np.abs(x) < 1e-100:
            myDiff = 0.0

        else:
            piX = np.pi*x
            # myDiff = np.pi * (
            #     (((np.pi * x) * np.cos(x * np.pi)) - np.sin(x * np.pi))
            #     /
            #     np.square(x * np.pi)
            # )
            myDiff = (piX*np.cos(piX) - np.sin(piX))/(np.pi * np.power(x,2))
            # myDiff
        return myDiff


    @staticmethod
    def quadraticFit(x, y):
        X_T = np.array([np.power(x, 2), x, np.ones(len(x))])
        X = X_T.transpose()
        if len(x) < 3:
            raise ValueError(
                "Cannot fit a quadratic to less than three data points."
                )
        elif len(x) == 3:
            # Note: Suprisingly, it is faster to directly invert the X matrix
            # than it is to do a linear solve.  Strange.
            
            #coef = np.linalg.solve(X, y)
            coef = np.linalg.inv(X).dot(y)
        else:
            #coef = np.linalg.solve(X_T.dot(X).dot(X_T), y)
            coef = np.linalg.inv(X_T.dot(X)).dot(X_T).dot(y)
            
        return coef

    def initializeRealTimePlot(
            self,
            plotHandle=None,
            axisHandle=None            
    ):
        super().initializeRealTimePlot(plotHandle, axisHandle)
        self.RTPlotTDOA = self.RTPaxisHandle.scatter(
            self.signalTDOA,
            1
        )
        
        self.RTPlotTDOA_error, = self.RTPaxisHandle.plot(
            [
                self.signalTDOA - np.sqrt(self.TDOAVar),
                self.signalTDOA + np.sqrt(self.TDOAVar)
            ],
            [1,1]
        )
        
        return
    
    def realTimePlot(
            self,
            normalized=True
    ):
        if self.RTPlotHandle is None:
            self.initializeRealTimePlot()
        
        self.RTPlotTDOA.set_offsets([self.signalTDOA, 1])
        self.RTPlotTDOA_error.set_data(
            [
                self.signalTDOA - np.sqrt(self.TDOAVar),
                self.signalTDOA + np.sqrt(self.TDOAVar)
            ],
            [1,1]
        )
        super().realTimePlot(normalized, substateRange = slice(0,self.__filterOrder__))
        return

    @staticmethod
    def peakFinder(x,y):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        
        y1 = y[0]
        y2 = y[1]
        y3 = y[2]
        
        x0 = (
            -(y1*(np.square(x3) - np.square(x2)) + y2*(np.square(x1) - np.square(x3)) + y3*(np.square(x2) - np.square(x1)))
            /
            (2*(y1*(x2-x3) + y2*(x3-x1) + y3*(x1-x2)))
        )
        x0 = (
            (
                y1*(np.square(x2)-np.square(x3)) +
                y2*(np.square(x3)-np.square(x1)) +
                y3*(np.square(x1)-np.square(x2))
            )
            /
            (2*(y1*(x2-x3) + y2*(x3-x1) + y3*(x1-x2)))
        )
        return(x0)

    @staticmethod
    def peakFinderJacobian(x,y):
        x1 = x[0]
        x2 = x[1]
        x3 = x[2]
        
        y1 = y[0]
        y2 = y[1]
        y3 = y[2]

        A = np.square(x2) - np.square(x3)
        B = np.square(x3) - np.square(x1)
        C = np.square(x1) - np.square(x2)

        D = x2-x3
        E = x3-x1
        # E = x1-x2
        F = x1-x2

        AE = A*E
        AF = A*F
        
        BD = B*D
        BF = B*F

        CD = C*D
        CE = C*E
        denom = 2*np.power(((D*y1) + (E*y2) + (F*y3)),2)

        dT_dy1 = (
            ((AE - BD)*y2 + (AF - CD)*y3)
            /
            denom
        )
        
        dT_dy2 = (
            ((BD - AE)*y1 + (BF - CE)*y3)
            /
            denom
        )
        
        dT_dy3 = (
            ((CD - AF)*y1 + (CE - BF)*y2)
            /
            denom
        )

        return np.array([dT_dy1, dT_dy2, dT_dy3])
