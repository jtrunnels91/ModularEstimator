import numpy as np
from scipy.stats import multivariate_normal
from . import pointsource
from . import poissonsource 

class StaticXRayPointSource(pointsource.PointSource, poissonsource.StaticPoissonSource):

    def __init__(
            self,
            RA,
            DEC,
            peakPhotonFlux,
            attitudeStateName='attitude'
    ):

        pointsource.PointSource.__init__(self, RA, DEC, attitudeStateName)
        poissonsource.StaticPoissonSource.__init__(self, peakPhotonFlux)
        self.peakPhotonFlux = peakPhotonFlux

        return

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

        poisPR = poissonsource.StaticPoissonSource.computeAssociationProbability(
            self,
            measurement
            )

        return (anglePR * poisPR * self.peakPhotonFlux)


class UniformNoiseXRaySource(poissonsource.StaticPoissonSource):
    def __init__(
            self,
            photonFlux
    ):
        super().__init__(photonFlux)
        
        self.photonFlux = photonFlux

        return

    def computeAssociationProbability(
            self,
            measurement,
            stateDict,
            validationThreshold=0):

        anglePR = 1/(4 * np.pi)

        poisPR = super().computeAssociationProbability(measurement)
        
        totalPR = anglePR * poisPR * self.photonFlux

        return totalPR

    
## @class PeriodicPoissonSource is a class which models the signal from a
# periodic poisson source, e.g. a pulsar.  
class PeriodicXRaySource(poissonsource.PoissonSource, pointsource.PointSource):
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
            attitudeStateName='attitude'
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
        pointsource.PointSource.__init__(self, RA, DEC, attitudeStateName)


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

        # Process whatever was passed as the profile
        self.processProfile(profile, normalizeProfile, movePeakToZero)

        # Update the pulsar period and time array.
        self.pulsarPeriod = 1/self.phaseDerivatives[1]
        self.singlePeriodTimeArray = np.linspace(
            0, self.pulsarPeriod, len(self.profile)
        )

        self.computeSinglePeriodIntegral()
        
        return

    def processProfile(
            self,
            profile,
            normalizeProfile=True,
            movePeakToZero=True
    ):
        # If a string is received, assume this points to a file and try to
        # open it and import the data.
        if type(profile) is str:
            profileArray = np.loadtxt(profile)
            profileArray = profileArray[:, len(profileArray[0]) - 1]
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
                        self.hms2rad(
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
                        self.dms2rad(
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
        for i in range(len(self.profile)):

            self.singlePeriodIntegral[i] = np.trapz(
                self.profile[0:i + 1],
                self.singlePeriodTimeArray[0:i + 1],
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
                self.peakAmplitude = self.peakAmplitude * self.pulsedFraction
                self.backgroundCountRate = (
                    self.photonsPerPeriod *
                    self.phaseDerivatives[1] *
                    (1 - self.pulsedFraction)
                )
            else:
                self.backgroundCountRate = 0

        else:
            self.peakAmplitude = 1
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
    def getSignal(
            self,
            observatoryTime,
            tVar=None
    ):
        phase = self.getPhase(observatoryTime)

        if tVar is not None:
            tSigma = np.sqrt(tVar)
            phaseSigma = tSigma/self.pulsarPeriod
            if phaseSigma > np.sqrt(1/12):
                signal = self.avgPhotonFlux
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
                    signal *
                    self.peakAmplitude /
                    self.days2seconds(np.sqrt(12) * tSigma)
                )
        else:
            signal = self.getPulseFromPhase(phase)
                    
        if self.backgroundCountRate is not None:
            signal = signal + self.backgroundCountRate

        return(signal)

    
    
    def getPulseFromPhase(self,
                          phase):
        pFrac = np.mod(phase, 1.0)
        signal = np.interp(pFrac, self.profileIndex, self.profile)
        return signal * self.peakAmplitude

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

    def plot(self, nPeriods=1):
        tArray = np.linspace(
            0,
            self.pulsarPeriod * nPeriods,
            100 * nPeriods)
        plt.figure()
        plt.plot(tArray, self.getSignal(tArray))
        plt.show(block=False)

    @staticmethod
    def hms2rad(h, m, s):
        hours = h + m / 60.0 + s / 3600.0

        return 2.0 * np.pi * hours / 24.0

    @staticmethod
    def dms2rad(d, m, s):
        degrees = d + m / 60.0 + s / 3600.0
        return 2.0 * np.pi * degrees / 360.0
        
