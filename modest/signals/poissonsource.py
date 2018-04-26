import numpy as np
# from scipy.stats import multivariate_normal
from . import signalsource
from math import factorial


class PoissonSource(signalsource.SignalSource):
    def __init__(
            self,
            flux
            ):
        super().__init__()
        self.lastTime = 0
        self.flux = flux
        return

    def computeAssociationProbability(
            self,
            currentFlux,
            measurement
            ):
        time = measurement['time']
        dT = time - self.lastTime
        self.lastTime = time
        return np.exp(-self.flux * dT) * currentFlux


class StaticPoissonSource(PoissonSource):
    def __init__(
            self,
            flux
            ):
        super().__init__(flux)

    def computeAssociationProbability(
            self,
            measurement
            ):
        poissonProb = super().computeAssociationProbability(
            self.flux,
            measurement
        )
        return(poissonProb)

    
class PeriodicPoissonSource(PoissonSource):
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
            name=None
    ):
        # Store the user-passed arguments first.  These take priority of
        # parameters received in the PAR file.
        self.phaseDerivatives = phaseDerivatives
        self.RA = RA
        self.DEC = DEC
        self.TZRMJD = TZRMJD
        self.name = name

        self.avgPhotonFlux = avgPhotonFlux
        self.pulsedFraction = pulsedFraction
        
        # Process the PAR file, if received.  Give priority to parameters
        # passed directly as init arguments.
        if PARFile is not None:
            self.processPARFile(PARFile)

        # Update the pulsar period and time array.
        self.pulsarPeriod = 1/self.phaseDerivatives[1]
        self.singlePeriodTimeArray = np.linspace(
            0, self.pulsarPeriod, len(self.profile)
        )

        # Process whatever was passed as the profile
        self.processProfile(profile, normalizeProfile, movePeakToZero)

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
                    if (self.RA is None) or (replaceCurrentValues is True):
                        self.RA = PAR_RA

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
                    if (self.DEC is None) or (replaceCurrentValues is True):
                        self.DEC = PAR_Dec

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
                
        return

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
                self.singlePeriodTimeArraySeconds[0:i + 1],
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

    def getPhaseMJD(self,
                    MJD
    ):
        timeDiff = (MJD - self.TZRMJD) * (24.0 * 60.0 * 60.0)
        phase = 0
        for order in self.phaseDerivatives:
            phase = (
                phase +
                (
                    self.phaseDerivatives[order] * np.power(timeDiff, order) /
                    factorial(order)
                )
            )
        return(phase)
