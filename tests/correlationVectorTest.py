from context import modest as md
import matplotlib.pyplot as plt
import numpy as np
#np.random.seed(1)
plt.close('all')



orbitPeriod = 10000/(2*np.pi)
orbitAmplitude = 0

constantVelocity = -20
constantAcceleration = -0.001

tFinal = 3600*24*1
speedOfLight=299792
velocityStdDev = 30/speedOfLight
vVar = np.square(velocityStdDev)
nTaps = 9
aStdDev = 1e-2/speedOfLight
myProfile = './pulsarData/profiles/J0534+2200_profile.txt'
myPARFile = '/home/joel/Documents/pythonDev/research/pulsarJPDAF/Data/2019_03_15_22h02m05s_chandraPhaseFrequency/ephem_B1509-58_chandra5515.par'

# myPARFile = '/home/joel/Documents/pythonDev/research/pulsarJPDAF/Data/2019_03_25_10h08m33s_chandraPhaseFrequency/ephem_B0540-6919_chandra1735.par'
# myProfile = './pulsarData/profiles/sinProfile.txt'
myPARFile = '/home/joel/Documents/pythonDev/research/pulsarJPDAF/pulsarData/PAR_files/ephem_B0540-6919_chandra1735.par'



detectorArea = 400  # cm^2
electronVoltPerPhoton = 6e3  # Electron-Volt x 10^3
electronVoltPerErg = 6.242e11
ergsPerElectronVolt = 1 / electronVoltPerErg

myFlux = 9.93e-9 # erg/cm^2/s
myFlux = myFlux * electronVoltPerErg / electronVoltPerPhoton  # photons / s * cm^2
myFlux = myFlux * detectorArea  # photons/s

myPulseFraction = 0.7
navProcessNoise=1e-30

myPulsar = md.signals.PeriodicXRaySource(
    #myProfile,
    PARFile=myPARFile,
    detectorArea=detectorArea
    # avgPhotonFlux=myFlux,
    # pulsedFraction=myPulseFraction
)

speedOfLight = myPulsar.speedOfLight()
myUnitVec = myPulsar.unitVec()
constantOffset = myUnitVec * myPulsar.speedOfLight() * myPulsar.pulsarPeriod * 0

class myDynamics():
    def __init__(self):
        self.attitude = None
        return
    def position(self,t):
        return(
            np.array([
                (orbitAmplitude * np.cos(t/orbitPeriod) * myUnitVec[0]) + constantOffset[0],
                (orbitAmplitude * np.cos(t/orbitPeriod) * myUnitVec[1]) + constantOffset[1],
                (orbitAmplitude * np.cos(t/orbitPeriod) * myUnitVec[2]) + constantOffset[2]
            ]
            ) + (constantVelocity*t + constantAcceleration*np.power(t,2)/2)*myUnitVec
        )


    def velocity(self,t):
        return(
            (orbitAmplitude/orbitPeriod) *
            np.array([
                -np.sin(t/orbitPeriod) * myUnitVec[0],
                -np.sin(t/orbitPeriod) * myUnitVec[1],
                -np.sin(t/orbitPeriod) * myUnitVec[2]
            ]
            )  + (constantVelocity + constantAcceleration*t)*myUnitVec
        )
    def getRangeFunction(self, unitVec, tMax):
        def myRangeFunction(t):
            return(
                (
                    np.array([
                        (orbitAmplitude * np.cos(t/orbitPeriod) * myUnitVec[0]) + constantOffset[0],
                        (orbitAmplitude * np.cos(t/orbitPeriod) * myUnitVec[1]) + constantOffset[1],
                        (orbitAmplitude * np.cos(t/orbitPeriod) * myUnitVec[2]) + constantOffset[2]
                    ]
                    ) + (constantVelocity*t + constantAcceleration*np.power(t,2)/2)*unitVec
                    
                ).dot(unitVec)
            )
        return myRangeFunction
        
        
    

class mySpacecraftClass():
    def __init__(self):
        self.dynamics = myDynamics()

mySpacecraft = mySpacecraftClass()
from pint import UnitRegistry
ureg = UnitRegistry()
plt.close('all')
userData=md.setupfunctions.buildUserData('/home/joel/Documents/pythonDev/research/pulsarJPDAF/orbitalDynamicsSim.yaml')

mySpacecraft = md.spacecraft.simulation.SimulatedSpacecraft(userData.parameters, ureg)
    
photonMeasurements = myPulsar.generatePhotonArrivals(tFinal, spacecraft=mySpacecraft)
# photonArrivalTimes = photonArrivalTimes
vInitial={
    'value': np.random.normal(mySpacecraft.dynamics.velocity(0).dot(myUnitVec)/speedOfLight,velocityStdDev),
    'var':np.square(velocityStdDev)
}

aInitial = {
    'value': np.random.normal(mySpacecraft.dynamics.acceleration(0).dot(myUnitVec)/speedOfLight,aStdDev),
    'var':np.square(aStdDev)
}
gradInitial={'value':np.random.normal(myUnitVec.dot(mySpacecraft.dynamics.gradient(0)).dot(myUnitVec),1e-20), 'var':np.square(1e-20)}
gradInitial = None

myCorrelation = md.substates.CorrelationVector(
    myPulsar,
    nTaps,
    myPulsar.pulsarPeriod/(nTaps+1),
    signalTDOA=0,
    TDOAVar=0,
    measurementNoiseScaleFactor=1,
    processNoise=1e-5,
    centerPeak=True,
    peakLockThreshold=0.5,
    velocityNoiseScaleFactor=1,
    navProcessNoise=navProcessNoise,
    internalNavFilter='deep',
    vInitial=vInitial,
    aInitial=aInitial,
    gradInitial=gradInitial,
    peakEstimator='EK',
    )
myCorrelation.storeLastStateVectors=10

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
lastTUOTDOA = mySpacecraft.dynamics.position(0).dot(myUnitVec)/ myPulsar.speedOfLight()
timeUpdateOnlyT = []

vVec = [vInitial['value']*speedOfLight]
vStdVec = [np.sqrt(vInitial['var'])*speedOfLight]
aVec = [aInitial['value']*speedOfLight]
aStdVec = [np.sqrt(aInitial['var'])*speedOfLight]
if gradInitial:
    gradVec = [gradInitial['value']]
    gradTrueVec = [myUnitVec.dot(mySpacecraft.dynamics.gradient(0)).dot(myUnitVec)]
    gradStdVec = [np.sqrt(gradInitial['var'])]
tVec=[0]
myRangeFunction = mySpacecraft.dynamics.getRangeFunction(myPulsar.unitVec(),tFinal)
trueTDOA = [myRangeFunction(0)/myPulsar.speedOfLight()]
estTDOA = [0]
tdoaSTD = [0]

for photonMeas in photonMeasurements:
    arrivalT = photonMeas['t']['value']
    vMeas = mySpacecraft.dynamics.velocity(arrivalT) + np.random.normal(0,scale=np.sqrt(vVar),size=3)
    TUOTDOA = vMeas.dot(myUnitVec) * (arrivalT - lastT)/myPulsar.speedOfLight() + lastTUOTDOA
    lastTUOTDOA = TUOTDOA

    # dynamics = {
    #         'velocity': {'value': vMeas, 'var': np.eye(3)*vVar}
    #         }
    dynamics = None
    
    myFilter.timeUpdateEKF(arrivalT-lastT, dynamics=dynamics)
    # myFilter.timeUpdateEKF(arrivalT-lastT, dynamics=None)
#    if myCorrelation.peakLock is True:
#        myCorrelation.realTimePlot()
    #myFilter.timeUpdateEKF(photon-lastT)
    
    myFilter.measurementUpdateEKF(photonMeas, myPulsar.name)
        
    if (arrivalT-lastUpdateTime) > 1000:
        trueTDOA.append(myRangeFunction(arrivalT)/myPulsar.speedOfLight())
        estTDOA.append(myCorrelation.signalTDOA)

        tdoaSTD.append(np.sqrt(myCorrelation.TDOAVar))
        
        timeUpdateOnlyT.append(arrivalT)
        timeUpdateOnlyTDOA.append(TUOTDOA)
        
        vVec.append(myCorrelation.stateVector[nTaps]*speedOfLight)
        vStdVec.append(np.sqrt(myCorrelation.correlationVectorCovariance[nTaps,nTaps].value)*speedOfLight)
        if hasattr(myCorrelation, 'acceleration'):

            aVec.append(myCorrelation.stateVector[nTaps+1]*speedOfLight)
            aStdVec.append(np.sqrt(myCorrelation.correlationVectorCovariance[nTaps+1,nTaps+1].value)*speedOfLight)
        if hasattr(myCorrelation, 'gradient'):

            gradVec.append(myCorrelation.stateVector[nTaps+2])
            gradStdVec.append(np.sqrt(myCorrelation.correlationVectorCovariance[nTaps+2,nTaps+2].value))
            gradTrue = myUnitVec.dot(mySpacecraft.dynamics.gradient(arrivalT)).dot(myUnitVec)
            gradTrueVec.append(gradTrue)
        tVec.append(arrivalT)
        
        
        myCorrelation.realTimePlot()
        lastUpdateTime = int(arrivalT)
        correlationSubstate = myCorrelation
        myPulsarObject = myPulsar
        print()
        print("||=================================================||")
        print('Experiment Time: %f out of %f (s)' % (arrivalT, tFinal))
        print('Spacecraft Time: %f (s)' % (arrivalT))
        
        if hasattr(correlationSubstate, 'velocity'):
            vError = (
                (correlationSubstate.velocity * myPulsarObject.speedOfLight()) -
                mySpacecraft.dynamics.velocity(arrivalT).dot(myPulsarObject.unitVec())
            )
            vSigma = correlationSubstate.velocityStdDev * myPulsarObject.speedOfLight()
            print("VELOCITY ESTIMATE:")
            print('   True velocity: %s km/s' %(mySpacecraft.dynamics.velocity(arrivalT).dot(myPulsarObject.unitVec())))
            print('  Velocity error: %s km/s' %vError)
            print('  Velocity sigma: %s km/s' %vSigma)
            print()
        if hasattr(correlationSubstate, 'acceleration'):
            trueAcc = mySpacecraft.dynamics.acceleration(arrivalT).dot(myPulsarObject.unitVec())
            aError = (
                (correlationSubstate.acceleration * myPulsarObject.speedOfLight()) -
                trueAcc
            )
            aSigma = correlationSubstate.accelerationStdDev * myPulsarObject.speedOfLight()
            print("ACCELERATION ESTIMATE:")
            print('   True Acceleration: %s km/s^2' %trueAcc)
            print('  Acceleration error: %s km/s^2' %aError)
            print('  Acceleration sigma: %s km/s^2' %aSigma)
            print()
        if hasattr(correlationSubstate, 'gradient'):
            trueGrad = myPulsarObject.unitVec().dot(mySpacecraft.dynamics.gradient(arrivalT).dot(myPulsarObject.unitVec()))
            gradError = (
                (correlationSubstate.gradient) -
                trueGrad
            )
            gradSigma = correlationSubstate.gradientStdDev
            print("Gradient ESTIMATE:")
            print('   True Gradient: %s km/s^2/km' %trueGrad)
            print('  Gradient error: %s km/s^2/km' %gradError)
            print('  Gradient sigma: %s km/s^2/km' %gradSigma)
            print()

        # print('time: %f' % arrivalT)
        # print('True TDOA: %f' %(mySpacecraft.dynamics.position(arrivalT).dot(myUnitVec)/myPulsar.speedOfLight()))
        # print('Peak offset: %f' %(myCorrelation.peakCenteringDT))
        # print((myCorrelation.stateVector[myCorrelation.__filterOrder__]*speedOfLight - mySpacecraft.dynamics.velocity(arrivalT).dot(myUnitVec))/ mySpacecraft.dynamics.velocity(arrivalT).dot(myUnitVec))
        # print(np.sqrt(myCorrelation.correlationVectorCovariance[nTaps,nTaps].value)*speedOfLight)
    lastT = arrivalT

trueTDOA = np.array(trueTDOA)
estTDOA = np.array(estTDOA)
tdoaSTD = np.array(tdoaSTD)
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
plt.plot(tVec,np.array(vVec) - np.array([mySpacecraft.dynamics.velocity(t).dot(myUnitVec) for t in tVec]),label='velocity')
plt.plot(tVec,vStdVec,label='velocity std dev')
plt.plot(tVec,-np.array(vStdVec))
plt.legend()
plt.show(block=False)

plt.figure()
plt.plot(tVec,np.array(vVec),label='velocity estimated')
plt.plot(tVec,np.array([mySpacecraft.dynamics.velocity(t).dot(myUnitVec) for t in tVec]),label='velocity true')
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
    plt.loglog(tVec,np.abs(np.array(aVec) - np.array([mySpacecraft.dynamics.acceleration(myT).dot(myPulsar.unitVec()) for myT in tVec])),label='acceleration')
    plt.loglog(tVec,aStdVec,label='acceleration std dev')
    # plt.plot(tVec,-np.array(aStdVec))
    plt.legend()
    plt.show(block=False)

if hasattr(myCorrelation, 'gradient'):
    plt.figure()
    
    plt.loglog(tVec,np.abs(np.array(gradVec) - np.array(gradTrueVec)),label='gradient')
    plt.loglog(tVec,gradStdVec,label='gradient std dev')
    # plt.plot(tVec,-np.array(gradStdVec))
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
