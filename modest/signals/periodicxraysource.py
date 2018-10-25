## @file periodicxraysource.py
# @brief This file contains the PeriodicXRaySource class

import numpy as np
from math import factorial, isnan
import matplotlib.pyplot as plt

from . import pointsource
from . import poissonsource
from .. utils import spacegeometry as sg


## @class PeriodicPoissonSource is a class which models the signal from a
# periodic poisson source, e.g. a pulsar.
class PeriodicXRaySource(
        poissonsource.DynamicPoissonSource,
        pointsource.PointSource
):
    def __init__(
            self,
            profile,
            avgPhotonFlux=None,
            pulsedFraction=None,
            PARFile=None,
            movePeakToZero=True,
            normalizeProfile=True,
            phaseDerivatives=None,
            pulsarPeriod=None,
            RA=None,
            DEC=None,
            TZRMJD=None,
            name=None,
            attitudeStateName='attitude',
            correlationStateName=None,
            useProfileColumn=None,
            startTime=0,
            extent=0
    ):
        
        # Store the user-passed arguments first.  These take priority of
        # parameters received in the PAR file.

        ## @brief Stores the derivatives of phase with respect to time
        # @detail #phaseDerivatives is a dict which contains the derivatives
        # of phase with respect to time.  The 0th derivative of phase is
        # simply a constant phase offset (zero by default).  The 1st
        # derivative of phase is frequency.  Higher order derivatives reflect
        # the change in frequency with respect to time.
        #
        # The dict keys should be integers which indicate the order of the
        # derivative. e.g.
        # \code
        # phaseDerivatives = {0: 0, 1: 29.7417, 2: -3.7184e-19, 3: 1.1949e-20}
        # \endcode
        # (Phase derivatives shown for Crab Nebula pulsar)
        self.phaseDerivatives = phaseDerivatives


        ## @brief #TZRMJD stores the Modified Julian Date (MJD) corresponding
        # to t=0 seconds.
        self.TZRMJD = TZRMJD

        ## @brief #name is an arbitrary string that is used to identify the
        # signal.  Can be set to whatever the user wants, since it is only
        # used for display purposes.  Usually set to the astrophysical
        # signal's name (i.e. "Crab Pulsar" or "PSR B0531+21")
        self.name = name

        ## @brief #averagePhotonFlux is the mean value of photons expected to
        # arrive from the signal per second (units of photons/second)
        self.avgPhotonFlux = avgPhotonFlux

        ## @brief #pulsedFraction is a value from 0 to 1 which indicates what
        # percentage of the #avgPhotonFlux is "pulsed" and what percentage is
        # constant/background
        self.pulsedFraction = pulsedFraction

        
        # Process the PAR file, if received.  Give priority to parameters
        # passed directly as init arguments.
        if PARFile is not None:
            PAR_RA, PAR_Dec = self.processPARFile(PARFile)
            if (RA is None):
                RA = PAR_RA
            if (DEC is None):
                DEC = PAR_Dec


        # Initialize PointSource with Right ascension and declination values
        pointsource.PointSource.__init__(
            self, RA, DEC, attitudeStateName=attitudeStateName, extent=extent
        )

        # Check to make sure that we received either a phaseDerivatives dict
        # or a value for pulsar period
        if (
                (self.phaseDerivatives is None) and
                (pulsarPeriod is None)
        ):
            raise ValueError(
                "Not enough timing information was received to initialize a " +
                "pulsar signal.  One of the following inputs is required: \n" +
                "- pulsarPeriod \n" +
                "- phaseDerivatives" +
                "- PARFile containing frequency information"
            )

        ## @brief #normalizeProfile is a boolean flag used to indicate whether
        # the profile is normalized from zero to one.  If so, then the signal
        # should be multiplied by #peakAmplitude.
        self.normalizeProfile = normalizeProfile
        
        # Process whatever was passed as the profile
        self.processProfile(profile, normalizeProfile, movePeakToZero, useProfileColumn)

        if correlationStateName is None:
            correlationStateName = self.name
        
        
        # Update the pulsar period and time array.

        ## @brief #pulsarPeriod is the amouont of time (in seconds) for one
        # complete pulsar pulse.
        self.pulsarPeriod = 1/self.phaseDerivatives[1]

        self.computeSinglePeriodIntegral()
        
        poissonsource.DynamicPoissonSource.__init__(
            self,
            self.peakAmplitude,
            correlationStateName=correlationStateName,
            startTime=startTime
        )
        
        return

    def processProfile(
            self,
            profile,
            normalizeProfile=True,
            movePeakToZero=True,
            useProfileColumn=None
    ):
        # If a string is received, assume this points to a file and try to
        # open it and import the data.
        if type(profile) is str:
            profileArray = np.loadtxt(profile)
            if useProfileColumn is None:
                profileArray = profileArray[:, len(profileArray[0]) - 1]
            else:
                profileArray = profileArray[:, int(useProfileColumn)]
            profile = profileArray

        if normalizeProfile is True:
            if (self.avgPhotonFlux is None) or (self.pulsedFraction is None):
                raise Warning(
                    "You are normalizing the profile from zero to one, but " +
                    "you haven't given values for average photon flux or " +
                    "pulsed fraction.  This will result in a signal which " +
                    "is most likely not scaled properly."
                )
            profile = profile - np.min(profile)
            profile = profile/np.max(profile)

        if movePeakToZero is True:
            if np.argmax(profile) != 0:
                profile = np.roll(profile, -np.argmax(profile))

        profile = np.append(profile, profile[0])

        ## @brief #profile is a numpy array containing the numerical value of
        # flux over a single period of the signal.
        #
        # @detail The #profile array contains the signal profile of the pulsar
        # (or periodic source) being modeled.  If the user selected to
        # normalize the profle, then the profile will be normalized from zero
        # to one, and then scaled based on the average flux value.  If the
        # profile is not normalized, then the raw values will be used for
        # computing the signal.  If the profile is normalized but no average
        # flux value is received, a warning will be issued.
        self.profile = profile
        self.profileIndex = np.linspace(0, 1, len(self.profile))

    def processPARFile(
            self,
            PARFile,
            replaceCurrentValues=False
    ):
        # Read PAR file, and split into lines
        parTextFile = open(PARFile, "r")
        lines = parTextFile.read().split('\n')

        PARPhaseDerivatives = {0: 0}

        for line in lines:
            # Split the line into a list of strings, and strip the
            # whitespace
            splitLine = line.strip().split()

            # If the line contains relevant information, store it.
            if len(splitLine) > 0:

                # FREQUENCY
                # If line contains frequency information, the first string
                # should be of the format "F<i>" where i is an integer.
                if (
                        (splitLine[0][0] == 'F') and
                        (len(splitLine[0]) == 2)
                ):
                    # Extract the order of derivative of the phase.
                    # e.g. F0 is the 1st derivative of phase, F1 is the
                    # 2nd derivative of phase, etc.
                    freqOrder = int(splitLine[0][1]) + 1
                    PARPhaseDerivatives[freqOrder] = float(splitLine[1])

                # RIGHT ASCENSION
                elif ((splitLine[0] == 'RAJ') or
                      (splitLine[0] == 'RA')):
                    # PAR files store right ascension as HH:MM:SS, so split
                    # on the ":" character
                    hmsArray = splitLine[1].split(':')

                    PAR_RA = (
                        sg.hms2rad(
                            float(hmsArray[0]),
                            float(hmsArray[1]),
                            float(hmsArray[2])
                        )
                    )

                # DECLINATION
                elif ((splitLine[0] == 'DECJ') or
                      (splitLine[0] == 'DEC')):
                    # Split on ":" and convert to radians
                    dmsArray = splitLine[1].split(':')

                    PAR_Dec = (
                        sg.dms2rad(
                            float(dmsArray[0]),
                            float(dmsArray[1]),
                            float(dmsArray[2])
                        )
                    )

                # T-zero Mod Julian Date
                elif (splitLine[0] == 'TZRMJD'):
                    PAR_TZRMJD = float(splitLine[1])
                    if (self.TZRMJD is None) or (replaceCurrentValues is True):
                        self.TZRMJD = PAR_TZRMJD

                # Pulsar Name
                elif ((splitLine[0] == 'PSRJ')
                      or
                      (splitLine[0] == 'PSR')):
                    if (self.name is None) or (replaceCurrentValues is True):
                        self.name = splitLine[1]
                        
            if (
                    (self.phaseDerivatives is None) or
                    (replaceCurrentValues is True)
            ):
                self.phaseDerivatives = PARPhaseDerivatives
                
        return(PAR_RA, PAR_Dec)

    def computeSinglePeriodIntegral(
            self
            ):
        # Compute the integral as a function of time of the pulsar flux.  This
        # will be used later to compute expected value of flux in the case
        # where time is uncertain.
        self.singlePeriodIntegral = np.zeros(len(self.profile))

        singlePeriodTimeArray = np.linspace(
            0, self.pulsarPeriod, len(self.profile)
        )

        self.profileLen = len(self.profile)
        
        for i in range(len(self.profile)):

            self.singlePeriodIntegral[i] = np.trapz(
                self.profile[0:i + 1],
                singlePeriodTimeArray[0:i + 1],
                axis=0
            )

        # Store the total flux integral over one period
        fluxIntegral = self.singlePeriodIntegral[-1]

        # Compute the scaling factor based on the average photon flux (if the
        # average flux was given)
        if self.avgPhotonFlux is not None:

            # Compute number of photons for a given pulsar period
            self.photonsPerPeriod = (
                self.avgPhotonFlux * self.pulsarPeriod
            )

            # Scale the peak amplitude so that a single integrated pulse will
            # result in the expected number of photons
            self.peakAmplitude = self.photonsPerPeriod / fluxIntegral

            # If some of the flux is unpulsed, scale the peak amplitude
            # accordingly and compute a background rate to account for the
            # unpulsed portion
            if self.pulsedFraction is not None:
                self.scaleFactor = self.peakAmplitude * self.pulsedFraction
                self.backgroundCountRate = (
                    self.avgPhotonFlux *
                    (1 - self.pulsedFraction)
                )
            else:
                self.backgroundCountRate = 0
                self.scaleFactor = self.peakAmplitude

        else:
            self.peakAmplitude = np.max(self.profile)
            self.scaleFactor = 1.0
            self.backgroundCountRate = 0

        return

    def getPhase(
            self,
            observatoryTime
    ):
        phase = 0
        for order in self.phaseDerivatives:
            phase = (
                phase +
                (
                    self.phaseDerivatives[order] *
                    np.power(observatoryTime, order) /
                    factorial(order)
                )
            )
        return(phase)

    def getFrequency(
            self,
            observatoryTime
    ):
        frequency = 0
        for order in self.phaseDerivatives:
            if order > 0:
                frequency  = (
                    frequency +
                    (
                        self.phaseDerivatives[order] *
                        np.power(observatoryTime, order-1)/
                        factorial(order - 1)
                    )
                )
        return frequency
    def getPeriod(
            self,
            observatoryTime
    ):
        return 1/self.getFrequency(observatoryTime)
    
    ## @fun #getSignalMJD is a wrapper function that returns the photon flux
    # at a given Modified Julian Date
    #
    # @param self The object pointer
    # @param MJD The Modified Jullian Date for which flux is to be returned
    #
    # @returns The signal at the requested date
    def getSignalMJD(
            self,
            MJD
    ):
        observatoryTime = self.MJD2seconds(MJD)
        
        return (self.getSignal(observatoryTime))

    ## @fun #getSignal is responsible for returning the photon flux from the
    # pulsar at a given time
    #
    # @details The #getSignal method is the method which is responsible for
    # returning the current value of flux from the signal source at a given
    # time.  If there is uncertainty in the time, then the expected value of
    # the signal is returned.  Uncertainty in time is indicated by the
    # optional tVar argument.
    #
    # In every case, the method calls the #getPhase method to determine the
    # current signal source phase.  If no uncertainty in time is passed, then
    # the #getPulseFromPhase method is called to lookup/interpolate the
    # current flux based on the phase and the signal #profile.
    #
    # If a value is passed for tVar, then the process is more complicated.
    # Rather than simply look up the signal value from the phase, the function
    # returns an approximation of the expected value of the signal, given the
    # mean and variance of time.
    #
    # Normally, the expected value would be computed by integrating the
    # product of the true distribution of time (probably a normal distribution)
    # with the flux as a function of time.  However, due to the non-analytical
    # nature of the flux function, the direct computation of this integral is
    # intractable.  To overcome this limitation, the time distributuion is
    # approximated as a moment-matched uniform distribution.  Using this
    # approximation, the approximate integral may be directly computed simply
    # by looking up the start and end values of #singlePeriodIntegral, and
    # dividing by the appropriate scaling factor.
    #
    # @param self The object pointer
    # @param observatoryTime The time for which to compute the signal
    # @param tVar (optional) The variance of the time estimate
    #
    # @return The signal at the requested time
    def getSignal(
            self,
            observatoryTime,
            tVar=None,
            #state=None
    ):
        # if state is not None:
        #     if 'signalDelay' in state:
        #         delay = state['signalDelay']
        #         if 'delayVar' in state:
        #             delayVar = state['delayVar']
        #         else:
        #             delayVar = 0
        #         observatoryTime = observatoryTime + delay
        #         tVar = tVar + delayVar
        # Get the phase corresponding to the current time
        phase = self.getPhase(observatoryTime)

        # If a value was received for the tVar, then we compute the expected
        # value of flux
        if tVar is not None:

            # Get standard deviation of t
            tSigma = np.sqrt(tVar)

            # Convert the time standard deviation to phase standard deviation
            phaseSigma = tSigma/self.pulsarPeriod

            # Check to see if the phase std is bigger than the std
            # corresponding to a uniform distribution with support = 1.  If so,
            # this indicates that we effectively have no meaningful knowledge
            # of phase, and can just return the average flux.
            if phaseSigma > np.sqrt(1/12):
                signal = self.avgPhotonFlux * self.pulsedFraction
            elif phaseSigma < 1/(100 * self.profileLen):
                signal = self.getPulseFromPhase(phase)

            else:
                phaseFraction = np.mod(phase, 1.0)
                upperSigma = phaseFraction + (np.sqrt(12) * phaseSigma / 2)
                lowerSigma = phaseFraction - (np.sqrt(12) * phaseSigma / 2)
                upperSigmaOffset = 0
                lowerSigmaOffset = 0
                if upperSigma > 1:
                    upperSigma = upperSigma - 1
                    upperSigmaOffset = self.singlePeriodIntegral[-1]
                if lowerSigma < 0:
                    lowerSigma = lowerSigma + 1
                    lowerSigmaOffset = self.singlePeriodIntegral[-1]
                signal = (
                    upperSigmaOffset +
                    np.interp(
                        upperSigma,
                        self.profileIndex,
                        self.singlePeriodIntegral
                    ) -
                    np.interp(
                        lowerSigma,
                        self.profileIndex,
                        self.singlePeriodIntegral)
                    + lowerSigmaOffset
                )
                signal = (
                    signal /
                    (np.sqrt(12) * tSigma)
                )
                signal = signal * self.scaleFactor
                
        else:
            signal = self.getPulseFromPhase(phase)
                    
        if self.backgroundCountRate is not None:
            signal = signal + self.backgroundCountRate

        return(signal)

    def signalIntegral(
            self,
            tStart,
            tStop,
            state=None
            ): 
        if state is not None:
            if 'signalDelay' in state:
                delay = state['signalDelay']
                
                tStart = tStart + delay
                tStop = tStop + delay
        # Get the phase corresponding to the current time
        phaseStart = self.getPhase(tStart)
        phaseStop = self.getPhase(tStop)

        completeCycles = np.floor(phaseStop-phaseStart)

        phaseStartFraction = np.mod(phaseStart, 1.0)
        phaseStopFraction = np.mod(phaseStop, 1.0)

        integralTStart = np.interp(
            phaseStartFraction,
            self.profileIndex,
            self.singlePeriodIntegral
        )
        integralTStop = np.interp(
            phaseStopFraction,
            self.profileIndex,
            self.singlePeriodIntegral
        )
        phaseFractionIntegral = integralTStop - integralTStart
        if phaseFractionIntegral < 0:
            phaseFractionIntegral = (
                phaseFractionIntegral + self.singlePeriodIntegral[-1]
                )
        signalIntegral = (
            phaseFractionIntegral + self.singlePeriodIntegral[-1] * completeCycles
            )
        signalIntegral = signalIntegral * self.scaleFactor

        if self.backgroundCountRate is not None:
            signalIntegral = (
                signalIntegral +
                self.backgroundCountRate * (tStop - tStart)
                )
       
        return signalIntegral
    
    def getPulseFromPhase(self,
                          phase):
        pFrac = np.mod(phase, 1.0)
        signal = np.interp(pFrac, self.profileIndex, self.profile)
        signal = signal * self.scaleFactor
        return signal

    def MJD2seconds(
            self,
            MJD
    ):
        return (MJD - self.TZRMJD) * (24.0 * 60.0 * 60.0)

    def seconds2MJD(
            self,
            seconds
    ):
        return self.TZRMJD + (seconds/(24.0 * 60.0 * 60.0))
    
    def computeAssociationProbability(
            self,
            measurement,
            stateDict,
            validationThreshold=0
    ):
        anglePR = pointsource.PointSource.computeAssociationProbability(
            self,
            measurement,
            stateDict,
            validationThreshold
        )

        poisPR = poissonsource.DynamicPoissonSource.computeAssociationProbability(
            self,
            measurement,
            stateDict
            )

        poisPR = 1.0
        myPr = (anglePR * poisPR * self.flux)
        if isnan(myPr):
            raise ValueError(
                'Computed NaN probability.  Components: AOA %s, TOA %s, Flux %s'
                %(anglePR, poisPR, self.peakAmplitude)
            )
        return myPr
    
    def plot(self,
             nPeriods=1,
             tVar=None,
             figureHandle=None,
             nPoints=1000
    ):
        if figureHandle is None:
            plt.figure()
            
        tArray = np.linspace(
            0,
            self.pulsarPeriod * nPeriods,
            nPoints * nPeriods)
        
        signalArray = np.zeros(nPoints * nPeriods)

        for index in range(len(signalArray)):
            signalArray[index] = self.getSignal(tArray[index], tVar)
        plt.plot(tArray, signalArray)
        plt.show(block=False)

    def generatePhotonArrivals(
            self,
            tMax,
            t0=0,
            position=None,
            attitude=None,
            FOV=None,
            AOA_StdDev=None,
            TOA_StdDev=None
            ):

        nCandidates = np.int((tMax - t0) * self.peakAmplitude * 1.1)

        # Generate a batch of candidate arrival times (more efficient than generating on the fly)
        candidateTimeArray = np.random.exponential(1.0/self.peakAmplitude, nCandidates)
        selectionVariableArray = np.random.uniform(0, 1, nCandidates)

        photonMeasurements = []
        photonArrivalTimes = []
        tLastCandidate = t0

        # If we have a position, then we want to get the signal at T0 at that
        # position, not the SSB.  So, shift T0 accordingly.
        if position is not None:
            rangeDeltaT = self.unitVec().dot(position(t0)) / self.speedOfLight()
            tLastCandidate = tLastCandidate + rangeDeltaT
        
        candidateIndex = 0
        while tLastCandidate < tMax:
            # Draw the next arrival time and selection variable from our
            # pre-generated arrays
            if candidateIndex < len(candidateTimeArray):
                nextCandidate = candidateTimeArray[candidateIndex]
                selectionVariable = selectionVariableArray[candidateIndex]
            # If we run out, generate more on the fly
            else:
                nextCandidate = np.random.exponential(1.0/self.peakAmplitude)
                selectionVariable = np.random.uniform(0, 1)
                print('Generating on the fly!')
            tNextCandidate = (
                tLastCandidate +
                nextCandidate
                )
            currentFluxNormalized = (
                self.getSignal(tNextCandidate)/self.peakAmplitude
                )
            # This if statement uses a uniform variable to determine whether
            # the next generated photon arrival time is a real photon arrival
            # time.
            if selectionVariable <= currentFluxNormalized:
                if position is not None:
                    rangeDeltaT = (
                        self.unitVec().dot(position(tNextCandidate)) /
                        self.speedOfLight()
                        )
                    newPhotonArrivalTime = tNextCandidate - rangeDeltaT
                    photonArrivalTimes.append(tNextCandidate - rangeDeltaT)
                else:
                    newPhotonArrivalTime = tNextCandidate
                    photonArrivalTimes.append(tNextCandidate)

                if TOA_StdDev:
                    newPhotonArrivalTime = (
                        newPhotonArrivalTime + np.random.normal(scale=TOA_StdDev)
                    )
                    measurementDict = {
                        't': {
                            'value': newPhotonArrivalTime,
                            'var': np.square(TOA_StdDev)
                        },
                        'name': self.name
                    }
                else:
                    measurementDict = {
                        't': {'value': newPhotonArrivalTime},
                        'name': self.name
                    }

                if attitude is not None:
                    measurementDict.update(
                        self.generateArrivalVector(attitude(newPhotonArrivalTime), AOA_StdDev)
                    )
                    # measurementDict = {
                    #     't': {'value': newPhotonArrivalTime},
                    #     'unitVec': {'value': unitVecMeas},
                    #     'RA': {'value': RaMeas},
                    #     'DEC': {'value': DecMeas},
                    #     'name': self.name
                    #     }
                    
                photonMeasurements.append(measurementDict)
                
            tLastCandidate = tNextCandidate
            candidateIndex = candidateIndex + 1

        return photonMeasurements
    
    # @staticmethod
    # def hms2rad(h, m, s):
    #     hours = h + m / 60.0 + s / 3600.0

    #     return 2.0 * np.pi * hours / 24.0

    # @method
    # def dms2rad(d, m, s):
    #     degrees = d + m / 60.0 + s / 3600.0
    #     return 2.0 * np.pi * degrees / 360.0
    
    def speedOfLight(
            self
    ):
        return (299792)
