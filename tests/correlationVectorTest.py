from context import modest as md
import matplotlib.pyplot as plt
import numpy as np

nTaps = 20

myProfile='./pulsarData/J0534+2200_profile.txt'
myPARFile = './pulsarData/ephem_J0534+2200_nancay_jodrell.par'

detectorArea = 100  # cm^2
electronVoltPerPhoton = 6e3  # Electron-Volt x 10^3
electronVoltPerErg = 6.242e11
ergsPerElectronVolt = 1 / electronVoltPerErg

myFlux = 9.93e-9 # erg/cm^2/s
myFlux = myFlux * electronVoltPerErg / electronVoltPerPhoton  # photons / s * cm^2
myFlux = myFlux * detectorArea  # photons/s

myPulseFraction = 0.70

myPulsar = md.signals.PeriodicXRaySource(
    myProfile,
    PARFile=myPARFile,
    avgPhotonFlux=myFlux,
    pulsedFraction=myPulseFraction
)

photonArrivalTimes = myPulsar.generatePhotonArrivals(50)
photonArrivalTimes = photonArrivalTimes + myPulsar.pulsarPeriod/2

print(len(photonArrivalTimes))

myCorrelation = md.substates.CorrelationVector(
    myPulsar,
    nTaps,
    myPulsar.pulsarPeriod/nTaps,
    signalDelay=0,
    delayVar=0,
    measurementNoiseScaleFactor=1
    )

myFilter = md.ModularFilter()
myFilter.addSignalSource(myPulsar.name, myPulsar)
myFilter.addStates(myPulsar.name, myCorrelation)

myMeas = {
    't': {'value': 0}
}
myFilter.measurementUpdateEKF(myMeas, myPulsar.name)

lastUpdateTime = 0
for photon in photonArrivalTimes:
    myMeas = {
        't': {'value': photon}
        }
    myFilter.measurementUpdateEKF(myMeas, myPulsar.name)

    if (photon-lastUpdateTime) > 10:
        myCorrelation.realTimePlot()
        lastUpdateTime = int(photon)
        print('time: %f' % photon)
