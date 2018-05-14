from context import modest as md
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

orbitPeriod = 100/(2*np.pi)
orbitAmplitude = 1e4



vVar = np.square(100)
nTaps = 9

myProfile = './pulsarData/J0534+2200_profile.txt'
myPARFile = './pulsarData/ephem_J0534+2200_nancay_jodrell.par'

detectorArea = 200  # cm^2
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

myUnitVec = myPulsar.unitVec()
constantOffset = -myUnitVec * myPulsar.speedOfLight() * 0.0033622786515540015 * 0

def position(t):
    return(
        np.array([
            (orbitAmplitude * np.sin(t/orbitPeriod) * myUnitVec[0]) + constantOffset[0],
            (orbitAmplitude * np.sin(t/orbitPeriod) * myUnitVec[1]) + constantOffset[1],
            (orbitAmplitude * np.sin(t/orbitPeriod) * myUnitVec[2]) + constantOffset[2]
            ]
        )
    )


def velocity(t):
    return(
        (orbitAmplitude/orbitPeriod) *
        np.array([
            np.cos(t/orbitPeriod) * myUnitVec[0],
            np.cos(t/orbitPeriod) * myUnitVec[1],
            np.cos(t/orbitPeriod) * myUnitVec[2]
            ]
        )
    )

photonArrivalTimes = myPulsar.generatePhotonArrivals(300, position=position)
photonArrivalTimes = photonArrivalTimes

myCorrelation = md.substates.CorrelationVector(
    myPulsar,
    nTaps,
    myPulsar.pulsarPeriod/nTaps,
    signalTDOA=0,
    TDOAVar=0,
    measurementNoiseScaleFactor=1,
    processNoise=1e-9,
    centerPeak=True
    )

myFilter = md.ModularFilter()
myFilter.addSignalSource(myPulsar.name, myPulsar)
myFilter.addStates(myPulsar.name, myCorrelation)

myMeas = {
    't': {'value': 0}
}
myFilter.measurementUpdateEKF(myMeas, myPulsar.name)

lastUpdateTime = 0
lastT = 0

timeUpdateOnlyTDOA = []
lastTUOTDOA = 0
timeUpdateOnlyT = []


for photon in photonArrivalTimes:
    vMeas = velocity(photon) + np.random.normal(0,scale=np.sqrt(vVar),size=3)
    TUOTDOA = vMeas.dot(myUnitVec) * (photon - lastT)/myPulsar.speedOfLight() + lastTUOTDOA
    timeUpdateOnlyTDOA.append(TUOTDOA)
    lastTUOTDOA = TUOTDOA
    timeUpdateOnlyT.append(photon)
    
    dynamics = {
            'velocity': {'value': vMeas, 'var': np.eye(3)*vVar}
            }
    
    myFilter.timeUpdateEKF(photon-lastT, dynamics=dynamics)
#    if myCorrelation.peakLock is True:
#        myCorrelation.realTimePlot()
    #myFilter.timeUpdateEKF(photon-lastT)
    
    myMeas = {
        't': {'value': photon}
        }
    myFilter.measurementUpdateEKF(myMeas, myPulsar.name)

    if (photon-lastUpdateTime) > 5:
        myCorrelation.realTimePlot()
        lastUpdateTime = int(photon)
        print('time: %f' % photon)
    lastT = photon

trueDelay = (
    position(myCorrelation.stateVectorHistory['t']).transpose().dot(myPulsar.unitVec()) /
    myPulsar.speedOfLight()
    )
plt.figure()
plt.plot(
    myCorrelation.stateVectorHistory['t'],
    trueDelay
)
plt.plot(
    myCorrelation.stateVectorHistory['t'],
    myCorrelation.stateVectorHistory['signalTDOA']
)
plt.plot(
    timeUpdateOnlyT,
    timeUpdateOnlyTDOA
)
plt.plot(
    myCorrelation.stateVectorHistory['t'],
    np.sqrt(myCorrelation.stateVectorHistory['TDOAVar'])
)
plt.plot(
    myCorrelation.stateVectorHistory['t'],
    -np.sqrt(myCorrelation.stateVectorHistory['TDOAVar'])
)
plt.show(block=False)
