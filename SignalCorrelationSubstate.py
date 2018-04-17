## @package SignalCorrelationSubstate
# This package contains the #CorrelationFilter class

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
from SubStates import SubState
from SmartPanda import SmartPanda

## @class CorrelationFilter
# @brief CorrelationFilter estimates the correlation vector and delay between
# a signal and a time-delayed measurement of that signal
#
# @details
# This class contains an estimator which estimates the correlation vector
# between a signal (the #trueSignal) and measurements of that signal.  This
# correlation vector is then used to estimate the delay between the
# #trueSignal and the measurements of that signal.
#
# @note This class is essentially an implementation of the estimator presented in
# <a href="https://doi.org/10.2514/1.G002650">
# Recursive Range Estimation Using Astrophysical Signals of Opportunity</a>,
# J. Runnels, D. Gebre, Journal of Guidance, Control and Dynamics, 2017.  Some
# equations from the paper are included in the class documentation for
# reference.  A more detailed discussion and derivation of the estimator can
# be found in the journal article..

class CorrelationFilter(SubState):
    def __init__(
            self,
            signal,
            filterOrder,
            dT,
            correlationVector,
            correlationVectorCovariance,
            signalDelay=0,
            delayVar=0,
            centerPeak=True,
            t=0,
            peakFitPoints=1,
            processNoise=1e-12
            ):

        if isinstance(signal, PointSource):
            self.unitVecToSignal = signal.unitVec()
        
        ## @brief #trueSignal is a string which identifies which
        # #Signals.SignalSource object is the "true" signal for which the
        # correlation vector is being estimated
        self.trueSignal = signal

        ## @brief #filterOrder is the number of "taps" in the estimated
        # correlation vector, #cHat.
        self.filterOrder = filterOrder

        ## @brief #dT is the "sample period" or "bin size" of the estimated
        # correlation vector
        self.dT = dT

        ## @brief #cHat is the current estimate of the correlation vector
        # between the incoming signal measurements and the #trueSignal
        self.cHat = correlationVector
        
        ## @brief #PHat is the covariance matrix of the correlation vector
        # estimate, #cHat
        self.PHat = correlationVectorCovariance

        ## @brief #signalDelay is the current estimate of the delay between
        # the incoming signal measurements and the #trueSignal
        self.signalDelay = signalDelay
        
        ## @brief #delayVar is the variance of the signal delay estimate
        # #signalDelay
        self.delayVar = delayVar

        ## @brief #centerPeak indicates whether the correlation vector is
        # shifted to maintain the peak at the 0th tap
        self.centerPeak = True

        ## @brief #peakFitPoints is a variable which controls the number of
        # points used for quadratically estimating the location of the
        # correlation vector peak
        self.peakFitPoints = peakFitPoints

        ## @brief #processNoise is the scalar value used to generate an
        # additional process noise term in #timeUpdate.
        self.processNoise = processNoise
        
        super().__init__(
            stateDimension=filterOrder,
            stateVectorHistory={
                't': t,
                'stateVector': correlationVector,
                'covariance': correlationVectorCovariance,
                'aPriori': True,
                'signalDelay': signalDelay,
                'delayVar': delayVar
            }
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
        
        return

    ## @fun #timeUpdate returns the matrices for performing the correlation
    # vector time update.
    def timeUpdate(
            self,
            dT,
            dynamics=None
            ):
        
        timeUpdateMatrices = self.buildTimeUpdateMatrices(
            dT, dynamics, self.cHat
        )
        
        L = timeUpdateMatrices['L']
        Q = timeUpdateMatrices['Q']

        Qmat = np.outer(L) * Q + (np.eye(self.filterOrder) * self.processNoise)
        
        return {'F': timeUpdateMatrices['F'], 'Q': Qmat}

    def getMeasurementMatrices(
            self,
            measurement,
            source=None
    ):
        if (
                (source.signalID() == self.trueSignal.signalID()) and
                ('t' in measurement)
        ):

            measMat = self.getTOAMeasurementMatrices(measurement)
        
        return measMat
    
    """
    ###########################################################################
    # Functions Specific to This Class                                        #
    #                                                                         #
    # The following remaining functions are not required in order for this    #
    # class to be used as a SubState, and may be changed as needed,           #
    # including inputs and outputs.                                           #
    ###########################################################################
    """

    def getTOAMeasurementMatrices(
            self,
            measurement,
            corrVec
    ):
        photonTOA = measurement['t']
            
        H = np.eye(self.filterOrder)

        timeVector = np.linspace(
            0,
            (1-self.filterOrder) * self.dT,
            self.filterOrder
        )

        timeVector = timeVector + photonTOA - self.signalDelay

        signalTimeHistory = np.zeros(self.filterOrder)

        for timeIndex in range(len(timeVector)):
            signalTimeHistory[timeIndex] = (
                self.trueSignal.flux(timeVector[timeIndex]) * self.dT
            )

        dY = signalTimeHistory - corrVec
        
        measMatDict = {
            'H': H,
            'dY': dY
            }
        
        return measMatDict
        
    ## @fun #computeSignalDelay computes the delay between the #trueSignal and
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
    # The delay estimate that is returned is in units of #dT.  So, a returned
    # value of 2 would imply that the peak is located at 2, and therefore the
    # delay corresponding to the correlation vector is 2 #dT.
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

            indexDiff = deltaT/self.dT
            
            fractionalDelay = (
                (velocity.dot(self.unitVecToSignal) * indexDiff) /
                self.speedOfLight()
            )

            Q = (
                self.unitVecToSignal.dot(vVar).dot(self.unitVecToSignal) *
                np.square(indexDiff / self.speedOfLight())
                )
            
            # Initialize empty matricies
            F = np.zeros([self.filterOrder, self.filterOrder])
            L = np.zeros([self.filterOrder, self.filterOrder])

            # Build arrays of indicies from which to form the sinc function
            halfLength = np.ceil(self.filterOrder / 2)

            if np.mod(self.filterOrder, 2) == 0:
                baseVec = (
                    np.linspace(
                        1 - halfLength,
                        halfLength,
                        self.filterOrder
                    ) + fractionalDelay
                )

            else:
                baseVec = (
                    np.linspace(
                        1 - halfLength,
                        halfLength - 1,
                        self.filterOrder
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
            F = np.eye(self.filterOrder)
            L = np.zeros(self.filterOrder)
            Q = 0
        
        timeUpdateDict = {
            'F': F,
            'L': L,
            'Q': Q
        }
        
        return(timeUpdateDict)

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
