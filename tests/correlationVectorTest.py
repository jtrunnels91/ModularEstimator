from context import modest as md
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
plt.close('all')

orbitPeriod = 100/(2*np.pi)
orbitAmplitude = 1000

tFinal = 150

vVar = np.square(1000)
nTaps = 9

myProfile = './pulsarData/profiles/J0534+2200_profile.txt'
myPARFile = './pulsarData/PAR_files/ephem_J0534+2200_nancay_jodrell.par'

detectorArea = 100  # cm^2
electronVoltPerPhoton = 6e3  # Electron-Volt x 10^3
electronVoltPerErg = 6.242e11
ergsPerElectronVolt = 1 / electronVoltPerErg

myFlux = 9.93e-9 # erg/cm^2/s
myFlux = myFlux * electronVoltPerErg / electronVoltPerPhoton  # photons / s * cm^2
myFlux = myFlux * detectorArea  # photons/s

myPulseFraction = 0.7

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
            (orbitAmplitude * np.cos(t/orbitPeriod) * myUnitVec[0]) + constantOffset[0],
            (orbitAmplitude * np.cos(t/orbitPeriod) * myUnitVec[1]) + constantOffset[1],
            (orbitAmplitude * np.cos(t/orbitPeriod) * myUnitVec[2]) + constantOffset[2]
            ]
        )
    )


def velocity(t):
    return(
        (orbitAmplitude/orbitPeriod) *
        np.array([
            -np.sin(t/orbitPeriod) * myUnitVec[0],
            -np.sin(t/orbitPeriod) * myUnitVec[1],
            -np.sin(t/orbitPeriod) * myUnitVec[2]
            ]
        )
    )

photonMeasurements = myPulsar.generatePhotonArrivals(tFinal, position=position)
# photonArrivalTimes = photonArrivalTimes

myCorrelation = md.substates.CorrelationVector(
    myPulsar,
    nTaps,
    myPulsar.pulsarPeriod/nTaps,
    signalTDOA=0,
    TDOAVar=0,
    measurementNoiseScaleFactor=2,
    processNoise=1e-100,
    centerPeak=True,
    peakLockThreshold=1
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
lastTUOTDOA = position(0).dot(myUnitVec)/ myPulsar.speedOfLight()
timeUpdateOnlyT = []


for photonMeas in photonMeasurements:
    arrivalT = photonMeas['t']['value']
    vMeas = velocity(arrivalT) + np.random.normal(0,scale=np.sqrt(vVar),size=3)
    TUOTDOA = vMeas.dot(myUnitVec) * (arrivalT - lastT)/myPulsar.speedOfLight() + lastTUOTDOA
    timeUpdateOnlyTDOA.append(TUOTDOA)
    timeUpdateOnlyTDOA.append(TUOTDOA)
    lastTUOTDOA = TUOTDOA
    timeUpdateOnlyT.append(arrivalT)
    timeUpdateOnlyT.append(arrivalT)
    
    dynamics = {
            'velocity': {'value': vMeas, 'var': np.eye(3)*vVar}
            }
    
    # myFilter.timeUpdateEKF(arrivalT-lastT, dynamics=dynamics)
    myFilter.timeUpdateEKF(arrivalT-lastT, dynamics=None)
#    if myCorrelation.peakLock is True:
#        myCorrelation.realTimePlot()
    #myFilter.timeUpdateEKF(photon-lastT)
    
    myFilter.measurementUpdateEKF(photonMeas, myPulsar.name)

    if (arrivalT-lastUpdateTime) > 5:
        myCorrelation.realTimePlot()
        lastUpdateTime = int(arrivalT)
        print('time: %f' % arrivalT)
        print('True TDOA: %f' %(position(arrivalT).dot(myUnitVec)/myPulsar.speedOfLight()))
        print('Peak offset: %f' %(myCorrelation.peakCenteringDT))
    lastT = arrivalT
timeUpdateOnlyTDOA.append(TUOTDOA)
timeUpdateOnlyT.append(arrivalT)
timeUpdateOnlyTDOA.append(TUOTDOA)
timeUpdateOnlyT.append(arrivalT)
tVec = [sv['t'] for sv in myCorrelation.stateVectorHistory]
trueTDOA = np.array([
    (position(t).transpose().dot(myPulsar.unitVec())/myPulsar.speedOfLight())
    for t in tVec
])

estTDOA = np.array([sv['signalTDOA'] for sv in myCorrelation.stateVectorHistory])
plt.figure()
plt.plot(
    tVec,
    trueTDOA-estTDOA,
    label='estimate error'
)
plt.plot(
    timeUpdateOnlyT,
    trueTDOA - timeUpdateOnlyTDOA,
    label='Time update only'
)
tdoaSTD = np.array([np.sqrt(sv['TDOAVar']) for sv in myCorrelation.stateVectorHistory])
plt.plot(
    tVec,
    tdoaSTD
)
plt.plot(
    tVec,
    -tdoaSTD
)
plt.figure()
navPos = np.array([sv['stateVector'][0] for sv in myCorrelation.internalNavFilter.subStates['oneDPositionVelocity']['stateObject'].stateVectorHistory])
navPosStd = np.sqrt(np.array([sv['covariance'].value[0,0] for sv in myCorrelation.internalNavFilter.subStates['oneDPositionVelocity']['stateObject'].stateVectorHistory]))

navVel = np.array([sv['stateVector'][1] for sv in myCorrelation.internalNavFilter.subStates['oneDPositionVelocity']['stateObject'].stateVectorHistory])
navVelStd = np.sqrt(np.array([sv['covariance'].value[1,1] for sv in myCorrelation.internalNavFilter.subStates['oneDPositionVelocity']['stateObject'].stateVectorHistory]))
plt.plot(
    tVec,
    trueTDOA - navPos,
    label='Nav filter pos'
)
plt.plot(
    tVec,
    navPosStd,
)
plt.plot(
    tVec,
    -navPosStd,
)

plt.legend()
plt.show(block=False)

plt.figure()
plt.plot(
    tVec,
    navVel - np.array([velocity(t).dot(myPulsar.unitVec())/myPulsar.speedOfLight() for t in tVec]),
    label='Nav filter vel'
)
plt.plot(
    tVec,
    navVelStd,
)
plt.plot(
    tVec,
    -navVelStd,
)

plt.legend()
plt.show(block=False)
