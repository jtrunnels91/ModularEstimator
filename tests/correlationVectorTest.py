from context import modest as md
import matplotlib.pyplot as plt
import numpy as np
#np.random.seed(1)
plt.close('all')



orbitPeriod = 10000/(2*np.pi)
orbitAmplitude = 0

constantVelocity = 1
constantAcceleration = 0.001 * 0

tFinal = 5000
speedOfLight=299792
velocityStdDev = 1/speedOfLight
vVar = np.square(velocityStdDev)
nTaps = 5
aStdDev = 0.001/speedOfLight
aStdDev = 0
myProfile = './pulsarData/profiles/J0534+2200_profile.txt'
myPARFile = '/home/joel/Documents/pythonDev/research/pulsarJPDAF/Data/2019_03_15_22h02m05s_chandraPhaseFrequency/ephem_B1509-58_chandra5515.par'

myPARFile = '/home/joel/Documents/pythonDev/research/pulsarJPDAF/Data/2019_03_25_10h08m33s_chandraPhaseFrequency/ephem_B0540-6919_chandra1735.par'
myProfile = './pulsarData/profiles/sinProfile.txt'



detectorArea = 250  # cm^2
electronVoltPerPhoton = 6e3  # Electron-Volt x 10^3
electronVoltPerErg = 6.242e11
ergsPerElectronVolt = 1 / electronVoltPerErg

myFlux = 9.93e-9 # erg/cm^2/s
myFlux = myFlux * electronVoltPerErg / electronVoltPerPhoton  # photons / s * cm^2
myFlux = myFlux * detectorArea  # photons/s

myPulseFraction = 0.7

myPulsar = md.signals.PeriodicXRaySource(
    #myProfile,
    PARFile=myPARFile,
    detectorArea=detectorArea
    # avgPhotonFlux=myFlux,
    # pulsedFraction=myPulseFraction
)

speedOfLight = myPulsar.speedOfLight()
myUnitVec = myPulsar.unitVec()
constantOffset = myUnitVec * myPulsar.speedOfLight() * myPulsar.pulsarPeriod * 0.5

def position(t):
    return(
        np.array([
            (orbitAmplitude * np.cos(t/orbitPeriod) * myUnitVec[0]) + constantOffset[0],
            (orbitAmplitude * np.cos(t/orbitPeriod) * myUnitVec[1]) + constantOffset[1],
            (orbitAmplitude * np.cos(t/orbitPeriod) * myUnitVec[2]) + constantOffset[2]
            ]
        ) + (constantVelocity*t + constantAcceleration*np.power(t,2)/2)*myUnitVec
    )


def velocity(t):
    return(
        (orbitAmplitude/orbitPeriod) *
        np.array([
            -np.sin(t/orbitPeriod) * myUnitVec[0],
            -np.sin(t/orbitPeriod) * myUnitVec[1],
            -np.sin(t/orbitPeriod) * myUnitVec[2]
            ]
        )  + (constantVelocity + constantAcceleration*t)*myUnitVec
    )

photonMeasurements = myPulsar.generatePhotonArrivals(tFinal, position=position,TOA_StdDev=1e-9)
# photonArrivalTimes = photonArrivalTimes

myCorrelation = md.substates.CorrelationVector(
    myPulsar,
    nTaps,
    myPulsar.pulsarPeriod/(nTaps+1),
    signalTDOA=0,
    TDOAVar=0,
    measurementNoiseScaleFactor=1,
    processNoise=0,
    centerPeak=True,
    peakLockThreshold=0.5,
    velocityNoiseScaleFactor=1,
    navProcessNoise=np.square(1e-12),
    internalNavFilter='deep',
    vInitial={'value':np.random.normal(constantVelocity/speedOfLight,velocityStdDev), 'var':np.square(velocityStdDev)},
    aInitial={'value':np.random.normal(constantAcceleration/speedOfLight,aStdDev), 'var':np.square(aStdDev)},
    gradInitial={'value':np.random.normal(0,0.001/speedOfLight), 'var':np.square(0.001/speedOfLight)},
    peakEstimator='EK'
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

vVec = [0]
vStdVec = [0]
aVec = [0]
aStdVec = [0]
for photonMeas in photonMeasurements:
    arrivalT = photonMeas['t']['value']
    vMeas = velocity(arrivalT) + np.random.normal(0,scale=np.sqrt(vVar),size=3)
    TUOTDOA = vMeas.dot(myUnitVec) * (arrivalT - lastT)/myPulsar.speedOfLight() + lastTUOTDOA
    timeUpdateOnlyTDOA.append(TUOTDOA)
    timeUpdateOnlyTDOA.append(TUOTDOA)
    lastTUOTDOA = TUOTDOA
    timeUpdateOnlyT.append(arrivalT)
    timeUpdateOnlyT.append(arrivalT)

    vVec.append(myCorrelation.stateVector[nTaps]*speedOfLight)
    vStdVec.append(np.sqrt(myCorrelation.correlationVectorCovariance[nTaps,nTaps].value)*speedOfLight)
    if hasattr(myCorrelation, 'acceleration'):
        aVec.append(myCorrelation.stateVector[nTaps+1]*speedOfLight)
        aStdVec.append(np.sqrt(myCorrelation.correlationVectorCovariance[nTaps+1,nTaps+1].value)*speedOfLight)

    # dynamics = {
    #         'velocity': {'value': vMeas, 'var': np.eye(3)*vVar}
    #         }
    dynamics=None
    
    myFilter.timeUpdateEKF(arrivalT-lastT, dynamics=dynamics)
    # myFilter.timeUpdateEKF(arrivalT-lastT, dynamics=None)
#    if myCorrelation.peakLock is True:
#        myCorrelation.realTimePlot()
    #myFilter.timeUpdateEKF(photon-lastT)
    
    myFilter.measurementUpdateEKF(photonMeas, myPulsar.name)
    vVec.append(myCorrelation.stateVector[nTaps]*speedOfLight)
    vStdVec.append(np.sqrt(myCorrelation.correlationVectorCovariance[nTaps,nTaps].value)*speedOfLight)
    if hasattr(myCorrelation, 'acceleration'):

        aVec.append(myCorrelation.stateVector[nTaps+1]*speedOfLight)
        aStdVec.append(np.sqrt(myCorrelation.correlationVectorCovariance[nTaps+1,nTaps+1].value)*speedOfLight)

    if (arrivalT-lastUpdateTime) > 50:
        myCorrelation.realTimePlot()
        lastUpdateTime = int(arrivalT)
        print('time: %f' % arrivalT)
        print('True TDOA: %f' %(position(arrivalT).dot(myUnitVec)/myPulsar.speedOfLight()))
        print('Peak offset: %f' %(myCorrelation.peakCenteringDT))
        print((myCorrelation.stateVector[myCorrelation.__filterOrder__]*speedOfLight - velocity(arrivalT).dot(myUnitVec))/ velocity(arrivalT).dot(myUnitVec))
        # print(np.sqrt(myCorrelation.correlationVectorCovariance[nTaps,nTaps].value)*speedOfLight)
    lastT = arrivalT
timeUpdateOnlyTDOA.append(TUOTDOA)
timeUpdateOnlyT.append(arrivalT)
timeUpdateOnlyTDOA.append(TUOTDOA)
timeUpdateOnlyT.append(arrivalT)
tVec = [sv['t'] for sv in myCorrelation.stateVectorHistory]
vVec.append(myCorrelation.stateVector[nTaps]*speedOfLight)
vStdVec.append(np.sqrt(myCorrelation.correlationVectorCovariance[nTaps,nTaps].value)*speedOfLight)

if hasattr(myCorrelation, 'acceleration'):

    aVec.append(myCorrelation.stateVector[nTaps+1]*speedOfLight)
    aStdVec.append(np.sqrt(myCorrelation.correlationVectorCovariance[nTaps+1,nTaps+1].value)*speedOfLight)

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
# plt.plot(
#     timeUpdateOnlyT,
#     trueTDOA - timeUpdateOnlyTDOA,
#     label='Time update only'
# )
tdoaSTD = np.array([np.sqrt(sv['TDOAVar']) for sv in myCorrelation.stateVectorHistory])
plt.plot(
    tVec,
    tdoaSTD
)
plt.plot(
    tVec,
    -tdoaSTD
)
plt.legend()
plt.show(block=False)
plt.figure()
plt.plot(tVec,np.array(vVec) - np.array([velocity(t).dot(myUnitVec) for t in tVec]),label='velocity')
plt.plot(tVec,vStdVec,label='velocity std dev')
plt.plot(tVec,-np.array(vStdVec))
plt.legend()
plt.show(block=False)

plt.figure()
plt.plot(tVec,np.array(vVec),label='velocity estimated')
plt.plot(tVec,np.array([velocity(t).dot(myUnitVec) for t in tVec]),label='velocity true')
plt.legend()
plt.show(block=False)
# navPos = np.array([sv['stateVector'][0] for sv in myCorrelation.internalNavFilter.subStates['oneDPositionVelocity']['stateObject'].stateVectorHistory])
# navPosStd = np.sqrt(np.array([sv['covariance'].value[0,0] for sv in myCorrelation.internalNavFilter.subStates['oneDPositionVelocity']['stateObject'].stateVectorHistory]))

# navVel = np.array([sv['stateVector'][1] for sv in myCorrelation.internalNavFilter.subStates['oneDPositionVelocity']['stateObject'].stateVectorHistory])
# navVelStd = np.sqrt(np.array([sv['covariance'].value[1,1] for sv in myCorrelation.internalNavFilter.subStates['oneDPositionVelocity']['stateObject'].stateVectorHistory]))
# plt.plot(
#     tVec,
#     trueTDOA - navPos,
#     label='Nav filter pos'
# )
# plt.plot(
#     tVec,
#     navPosStd,
# )
# plt.plot(
#     tVec,
#     -navPosStd,
# )


if hasattr(myCorrelation, 'acceleration'):

    plt.figure()
    plt.plot(tVec,np.array(aVec) - constantAcceleration,label='acceleration')
    plt.plot(tVec,aStdVec,label='acceleration std dev')
    plt.plot(tVec,-np.array(aStdVec))
    plt.legend()
    plt.show(block=False)

# plt.figure()
# plt.plot(
#     tVec,
#     navVel - np.array([velocity(t).dot(myPulsar.unitVec())/myPulsar.speedOfLight() for t in tVec]),
#     label='Nav filter vel'
# )
# plt.plot(
#     tVec,
#     navVelStd,
# )
# plt.plot(
#     tVec,
#     -navVelStd,
# )

# plt.legend()
# plt.show(block=False)
# myX=np.linspace(-20,20,200000)
# mydiff=[myCorrelation.sincDiff(num) for num in myX]
# myInt = [0]
# myDX = np.diff(myX)[0]
# for diff in mydiff:
#     myInt.append(myInt[-1] + diff*myDX)
# plt.plot(myX, myInt[0:-1])
# plt.plot(myX,np.sinc(myX))
# plt.show(block=False)
