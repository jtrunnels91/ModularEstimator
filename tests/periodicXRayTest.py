from context import modest as md
import matplotlib.pyplot as plt
import numpy as np

myProfile='./pulsarData/J0534+2200_profile.txt'
myPARFile = './pulsarData/ephem_J0534+2200_nancay_jodrell.par'

myFlux = 9.93e-9 # erg/cm^2/s

detectorArea = 100 # cm^2
electronVoltPerPhoton = 6e3  # Electron-Volt x 10^3
electronVoltPerErg = 6.242e11
ergsPerElectronVolt = 1 / electronVoltPerErg

myFlux = myFlux * electronVoltPerErg / electronVoltPerPhoton  # photons / s * cm^2
myFlux = myFlux * detectorArea  # photons/s

myPulseFraction = 0.70

myPulsar = md.signals.PeriodicXRaySource(
    myProfile,
    PARFile=myPARFile,
    avgPhotonFlux=myFlux,
    pulsedFraction=myPulseFraction
)
angleVar = 1e-3
angleSTD = np.sqrt(angleVar)
myMeas = {
    't': {'value': 0, 'var': 1e-100},
    'RA': {'value': myPulsar.RaDec()['RA'] + np.random.normal(0, angleSTD), 'var': angleVar},
    'DEC': {'value': myPulsar.RaDec()['DEC'] + np.random.normal(0, angleSTD), 'var': angleVar}
}

state = 'a'

myCorr = md.substates.CorrelationVector(
    myPulsar,
    10,
    myPulsar.pulsarPeriod/10,
    signalDelay=0,
    delayVar=1e-100
)
attitudeSubstate = md.substates.Attitude(
    attitudeErrorCovariance=np.eye(3)*1e-1,
    gyroBiasCovariance=np.eye(3)*1e-10
)
stateDict = {
    myPulsar.name: {
        'stateObject': myCorr,
        'length': myCorr.dimension()
    },
    'attitude': {
        'stateObject': attitudeSubstate,
        'length': attitudeSubstate.dimension()
    }
}

myPr = myPulsar.computeAssociationProbability(myMeas, stateDict)
print(myPr)
