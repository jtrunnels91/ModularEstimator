## @file CorrelationVector
# This package contains the #CorrelationVector class

import numpy as np
from scipy.linalg import block_diag
from numpy import sin, cos, arcsin, arccos, arctan2, square, sqrt, abs, power
from numpy.linalg import norm
from pyquaternion import Quaternion

import sys
import os
sys.path.append("/home/joel/Documents/astroSourceTracking/libraries")
from Signals import PointSource
from SubState import SubState

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

class CorrelationVector(SubState):

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
    # - peakAmplitude()
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
            signalDelay=0,
            delayVar=0,
            aPriori=True,
            centerPeak=True,
            peakFitPoints=1,
            processNoise=1e-12,
            measurementNoiseScaleFactor=1
            ):

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
                np.square(self.__trueSignal__.peakAmplitude() * self.__dT__)
            )
        ## @brief #correlationVector is the current estimate of the
        # correlation vector between the incoming signal measurements and the
        # #__trueSignal__
        self.correlationVector = correlationVector

        if correlationVectorCovariance is None:
            correlationVectorCovariance = (
                np.eye(self.__filterOrder__) *
                np.square(self.__trueSignal__.peakAmplitude() * self.__dT__)
            )
        ## @brief #correlationVectorCovariance is the covariance matrix of the
        # correlation vector estimate, #correlationVector
        self.correlationVectorCovariance = correlationVectorCovariance

        ## @brief #signalDelay is the current estimate of the delay between
        # the incoming signal measurements and the #__trueSignal__
        self.signalDelay = signalDelay
        
        ## @brief #delayVar is the variance of the signal delay estimate
        # #signalDelay
        self.delayVar = delayVar

        ## @brief #centerPeak indicates whether the correlation vector is
        # shifted to maintain the peak at the 0th tap
        self.centerPeak = centerPeak

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
        
        super().__init__(
            stateDimension=filterOrder,
            stateVectorHistory={
                't': t,
                'stateVector': correlationVector,
                'covariance': correlationVectorCovariance,
                'aPriori': aPriori,
                'signalDelay': signalDelay,
                'delayVar': delayVar
            }
        )

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
        self.correlationVector = svDict['stateVector']
        self.correlationVectorCovariance = svDict['covariance']
        self.aPriori = svDict['aPriori']

        # Compute new estimate of delay based on new state vector, store in
        # svDict and local attributes
        delayDict = self.estimateSignalDelayUT(
            self.correlationVector,
            self.correlationVectorCovariance
        )

        self.signalDelay = delayDict['meanDelay']
        self.delayVar = delayDict['varDelay']

        svDict['signalDelay'] = delayDict['meanDelay']
        svDict['delayVar'] = delayDict['varDelay']
        
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
        
        timeUpdateMatrices = self.buildTimeUpdateMatrices(
            dT, dynamics, self.correlationVector
        )
        
        L = timeUpdateMatrices['L']
        Q = timeUpdateMatrices['Q']

        Qmat = np.outer(L) * Q + (np.eye(self.__filterOrder__) * self.processNoise)
        
        return {'F': timeUpdateMatrices['F'], 'Q': Qmat}

    def getMeasurementMatrices(
            self,
            measurement,
            source=None
    ):
        if (
                (source.signalID() == self.__trueSignal__.signalID()) and
                ('t' in measurement)
        ):

            measurementMatrices = self.getTOAMeasurementMatrices(measurement)

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
        if 'velocity' in dynamics:

            velocity = dynamics['velocity']['value']
            vVar = dynamics['velocity']['var']

            indexDiff = deltaT/self.__dT__
            
            fractionalDelay = (
                (velocity.dot(self.__unitVecToSignal__) * indexDiff) /
                self.speedOfLight()
            )

            Q = (
                self.__unitVecToSignal__.dot(
                    vVar
                ).dot(self.__unitVecToSignal__) *
                np.square(indexDiff / self.speedOfLight())
                )
            
            # Initialize empty matricies
            F = np.zeros([self.__filterOrder__, self.__filterOrder__])
            L = np.zeros([self.__filterOrder__, self.__filterOrder__])

            # Build arrays of indicies from which to form the sinc function
            halfLength = np.ceil(self.__filterOrder__ / 2)

            if np.mod(self.__filterOrder__, 2) == 0:
                baseVec = (
                    np.linspace(
                        1 - halfLength,
                        halfLength,
                        self.__filterOrder__
                    ) + fractionalDelay
                )

            else:
                baseVec = (
                    np.linspace(
                        1 - halfLength,
                        halfLength - 1,
                        self.__filterOrder__
                    ) + fractionalDelay
                )

            # Compute the sinc function of the base vector
            sincBase = np.sinc(baseVec)
            diffBase = np.zeros_like(sincBase)

            for i in range(len(baseVec)):
                diffBase[i] = self.sincDiff(baseVec[i])
            
            sincBase = np.roll(sincBase, 1 - int(halfLength))
            diffBase = np.roll(diffBase, 1 - int(halfLength))

            for i in range(len(F)):
                F[i] = np.roll(sincBase, i)
                L[i] = np.roll(diffBase, i)

            L = L.dot(h)

        else:
            # If no velocity was included in dynamics, then do nothing during
            # time update
            F = np.eye(self.__filterOrder__)
            L = np.zeros(self.__filterOrder__)
            Q = 0
        
        timeUpdateDict = {
            'F': F,
            'L': L,
            'Q': Q
        }
        
        return(timeUpdateDict)

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
        photonTOA = measurement['t']
            
        H = np.eye(self.__filterOrder__)

        timeVector = np.linspace(
            0,
            (1-self.__filterOrder__) * self.__dT__,
            self.__filterOrder__
        )

        timeVector = timeVector + photonTOA - self.signalDelay

        signalTimeHistory = np.zeros(self.__filterOrder__)

        for timeIndex in range(len(timeVector)):
            signalTimeHistory[timeIndex] = (
                self.__trueSignal__.flux(timeVector[timeIndex]) * self.__dT__
            )

        dY = signalTimeHistory - corrVec

        R = (
            np.eye(self.__filterOrder__) *
            self.__trueSignal__.peakAmplitude() *
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

    ## @fun #computeSignalDelay computes the delay between the #__trueSignal__ and
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
    def computeSignalDelay(
            self,
            c,
            P
    ):

        # First estimate of peak location is the location of the max value
        peakLocation = np.argmax(c)

        # Next, we "roll" the correlation vector so that the values being
        # fitted quadratically are the first 2 * peakFitPoints + 1 values
        rollFactor = self.peakFitPoints - peakLocation

        if rollFactor != 0:
            slicedC = np.roll(
                c,
                rollFactor
            )
            slicedP = (
                np.roll(
                    np.roll(P,
                            rollFactor,
                            axis=0),
                    rollFactor,
                    axis=1
                )
            )
        else:
            slicedC = c
            slicedP = P

        # Extract the portion of the correlation vector used for fitting
        # quadratic
        mySlice = slice(0, self.peakFitPoints * 2 + 1)

        slicedC = slicedC[mySlice]
        slicedP = slicedP[mySlice, mySlice]

        # np.polyfit assumes that the weights will be the inverse standard
        # deviation
        weightVector = np.diag(slicedP)
        weightVector = 1 / np.sqrt(weightVector)

        # xVec is the vector of "x" values corresponding the "y" values to
        # which the quadratic is being fit.
        xVec = np.linspace(0, self.peakFitPoints * 2, (self.peakFitPoints * 2) + 1)
        xVec = xVec - rollFactor

        # Get the quadratic function that fits the peak and surrounding values,
        # and use it to estimate the location of the max
        quadraticVec = np.polyfit(xVec, slicedC, 2, w=weightVector)
        delay = (-quadraticVec[1] / (2 * quadraticVec[0]))

        return delay

    ## @fun #estimateSignalDelayUT uses a unscented tranform to estimate the
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
    def estimateSignalDelayUT(
            self,
            h,
            P
    ):
        # Compute sigma points
        hDimension = len(h)
        sqrtP = np.linalg.cholesky(hDimension * P)
        
        sigmaPoints = h + np.append(sqrtP, -sqrtP, axis=0)

        sigmaPoints = np.append(np.array([h]), sigmaPoints, axis=0)

        # Initiate vector to store the resulting peaks from each sigma point
        sigmaPointResults = np.zeros(len(sigmaPoints))

        # Compute the peak corresponding to each sigma point vector
        for i in range(len(sigmaPoints)):
            sigmaPointResults[i] = (
                self.computeSignalOffset(sigmaPoints[i], P)
            )

        meanDelay = np.mean(sigmaPointResults)
        meanDelay = sigmaPointResults[0]
        varDelay = np.var(sigmaPointResults)

        return {'meanDelay': meanDelay, 'varDelay': varDelay}

    
    def speedOfLight(
            self
    ):
        return (299792)

    @staticmethod
    def sincDiff(x):
        if np.abs(x) < 1e-100:
            myDiff = 0.0

        else:
            myDiff = np.pi * (
                ((np.pi * x) * np.cos(x * np.pi) - np.sin(x * np.pi))
                /
                np.square(x * np.pi)
            )

        return myDiff
