from context import modest as md
import matplotlib.pyplot as plt

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

myMeas = {
    't': {'value': 0, 'var': 1}
}

state = 'a'

myPr=myPulsar.computeAssociationProbability(myMeas, state)
