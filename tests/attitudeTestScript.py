import numpy as np
import matplotlib.pyplot as plt
import sys
# from smartpanda import SmartPanda
from context import modest as me
from numpy import sin, cos, pi, sqrt
from numpy.random import exponential

np.random.seed(0)
#sys.path.append("/home/joel/Documents/astroSourceTracking/libraries")
# sys.path.append("/home/joel/Documents/astroSourceTracking/libraries/ModularFilter")

from SpaceGeometry import sidUnitVec, unitVector2RaDec
from QuaternionHelperFunctions import euler2quaternion, quaternion2euler, eulerAngleDiff

plt.close('all')
FOV = 1
useUnitVector=False
# Function defining angular velocity
#euler0 = np.random.uniform(-np.pi/2.1, np.pi/2.1, 3)

#euler0 = np.array([np.pi,0,0])
euler0 = np.array([3.046710761327998, 1.2100470620888801, 1.483962756901154])
#euler0 = np.array([0, 1.2100470620888801, 1.483962756901154])
def omega(t):
    
    # omegaT = np.array([sin(pi * t/4), cos(pi * t/12), sin(pi * t/16)])
    #omegaT = np.random.normal(np.zeros(3))
    omegaT = np.array([0, 0, 0])
    return(omegaT)
def attitude(t,returnQ=True):
    euler = np.array([euler0[0], euler0[1], euler0[2]])
    if returnQ:
        return euler2quaternion(euler)
    else:
        return euler
printFrequency=1.0
lastPrintTime=0

# Dynamics info
tCurrent = 0
timeStep = 0.01
tMax = 10

biasSTD = 1e-100
eulerT0True = attitude(0, returnQ=False)
biasTrue = np.random.normal(np.zeros(3), scale=biasSTD)
biasTrue=np.array([0,0,0])

rollErrorStd = 1e-2
RAErrorStd = 1e-2
DecErrorStd = 1e-2
biasErrorSTD = 1e-100
eulerT0Est = eulerT0True
eulerT0Est = np.array([
    np.random.normal(eulerT0True[0], rollErrorStd),
    np.random.normal(eulerT0True[1], DecErrorStd),
    np.random.normal(eulerT0True[2], RAErrorStd)
])
biasEst = np.random.normal(biasTrue, scale=biasErrorSTD)
biasEst=np.array([0,0,0])

q0 = euler2quaternion(eulerT0True)

QScalar = 1e-12
RScalar = 1e-6


# Initiate filters
myJPDAF = me.ModularFilter(measurementValidationThreshold=0, covarianceStorage='covariance')
myEKF = me.ModularFilter(covarianceStorage='covariance')
myML = me.ModularFilter(measurementValidationThreshold=1e-3,covarianceStorage='covariance')
myTUOnly = me.ModularFilter(covarianceStorage='covariance')
initialAttitudeCovariance = np.zeros([3,3])
initialAttitudeCovariance[0,0] = np.square(rollErrorStd)
initialAttitudeCovariance[1,1] = np.square(DecErrorStd)
initialAttitudeCovariance[2,2] = np.square(RAErrorStd)

# initialAttitudeCovariance = attitude(0).rotation_matrix.transpose().dot(initialAttitudeCovariance).dot(attitude(0).rotation_matrix)

# Initiate attitude stubstates
JPDAFAtt = me.substates.Attitude(
    euler2quaternion(eulerT0Est),
    initialAttitudeCovariance,
    np.zeros(3),
    np.eye(3) * np.square(biasSTD + biasErrorSTD),
    useUnitVector=useUnitVector
)

EKFAtt = me.substates.Attitude(
    euler2quaternion(eulerT0Est),
    initialAttitudeCovariance,
    np.zeros(3),
    np.eye(3) * np.square(biasSTD + biasErrorSTD),
    useUnitVector=useUnitVector
)

MLAtt = me.substates.Attitude(
    euler2quaternion(eulerT0Est),
    initialAttitudeCovariance,
    np.zeros(3),
    np.eye(3) * np.square(biasSTD + biasErrorSTD),
    useUnitVector=useUnitVector
)

TUOnlyAtt = me.substates.Attitude(
    euler2quaternion(eulerT0Est),
    initialAttitudeCovariance,
    np.zeros(3),
    np.eye(3) * np.square(biasSTD + biasErrorSTD),
    useUnitVector=useUnitVector
)

myJPDAF.addStates('attitude', JPDAFAtt)
myML.addStates('attitude', MLAtt)
myEKF.addStates('attitude', EKFAtt)
myTUOnly.addStates('attitude', TUOnlyAtt)

# Star and background info
backgroundFlux = 100
nStars = 3
starVecs = np.random.normal(np.zeros([nStars, 3]))
starCoordinates = np.zeros([nStars, 2])
fluxes = np.zeros(nStars + 1)
starArrivalTimes = np.zeros(nStars + 1)
bkgArrivalTime = exponential(1/backgroundFlux)

myNoise = me.signals.UniformNoiseXRaySource(backgroundFlux, detectorFOV=FOV*2)
myJPDAF.addSignalSource('background', myNoise)
myML.addSignalSource('background', myNoise)
myEKF.addSignalSource('background', myNoise)


starList = []
photonArrivals = []

# Generate signal objects, add them to filters
for i in range(nStars):
    fluxes[i] = 10
    starVecs[i] = starVecs[i]/np.linalg.norm(starVecs[i])
    starCoordinates[i] = unitVector2RaDec(starVecs[i])
    name = 'star%i' %i
    star = me.signals.StaticXRayPointSource(
        np.random.uniform(euler0[2]-(FOV*np.pi/180), euler0[2] + (FOV*np.pi/180)),
        np.random.uniform(-euler0[1]-(FOV*np.pi/180), -euler0[1] + (FOV*np.pi/180)),
        fluxes[i],
        name=name,
        useUnitVector=useUnitVector
    )
    # star = me.signals.StaticXRayPointSource(
    #     euler0[2] + 0.9*FOV*np.pi/180,
    #     -euler0[1],
    #     fluxes[i],
    #     name=name
    # )
    myJPDAF.addSignalSource('star%i' %i, star)

    myML.addSignalSource('star%i' %i, star)

    myEKF.addSignalSource('star%i' %i, star)

    myTUOnly.addSignalSource('star%i' %i, star)
    starList.append(star)
    photonArrivals = (
        photonArrivals +
        star.generatePhotonArrivals(
            tMax,
            attitude=attitude,
            AOA_StdDev=np.sqrt(RScalar),
            TOA_StdDev=1e-100
        )
    )
photonArrivals = (
    photonArrivals +
    myNoise.generatePhotonArrivals(
        tMax,
        attitude=attitude,
        AOA_StdDev=np.sqrt(RScalar),
        TOA_StdDev=1e-100
    )
)
photonArrivals = sorted(photonArrivals, key=lambda k: k['t']['value'])



# Lists for storing results
eulerAnglesTrue = []
JPDAFError = []
MLError = []
EKFError = []
TUError = []
for photonMeasurement in photonArrivals:
    nextArrivalTime = photonMeasurement['t']['value']
    # Propagate to next measurement time
    while tCurrent < nextArrivalTime:
        if tCurrent + timeStep < nextArrivalTime:
            currentDT = timeStep
        else:
            currentDT = nextArrivalTime - tCurrent
        ######################################
        # Dynamics of true system
        ######################################
        omegaCurrent = omega(tCurrent)

        diffRotation = euler2quaternion(omegaCurrent*currentDT)
        q0 = q0*diffRotation
        
        ######################################
        # Time Update:
        ######################################

        omegaMeas = omegaCurrent + np.random.normal(
            [0, 0, 0],
            [sqrt(QScalar), sqrt(QScalar), sqrt(QScalar)]
        ) + biasTrue

        dynamicsDict = {
            'omega': {'value': omegaMeas, 'var': QScalar},
            'gyroBias': {'var': 1e-10}
            }

        myJPDAF.timeUpdateEKF(currentDT, dynamicsDict)
        myML.timeUpdateEKF(currentDT, dynamicsDict)
        myEKF.timeUpdateEKF(currentDT, dynamicsDict)
        myTUOnly.timeUpdateEKF(currentDT, dynamicsDict)

        tCurrent = tCurrent + currentDT
        currentEATrue = np.array(quaternion2euler(q0))
        # eulerAnglesTrue.append(currentEATrue)
        
        eulerAnglesTrue.append({'t': tCurrent, 'eulerAngles': currentEATrue})
        JPDAFError.append({
            't': tCurrent,
            'eulerAngles': eulerAngleDiff(
                currentEATrue,
                myJPDAF.subStates['attitude']['stateObject'].eulerAngles()
                )
            }
        )

        MLError.append({
            't': tCurrent,
            'eulerAngles': eulerAngleDiff(
                currentEATrue,
                myML.subStates['attitude']['stateObject'].eulerAngles()
                )
            }
        )

        EKFError.append({
            't': tCurrent,
            'eulerAngles': eulerAngleDiff(
                currentEATrue,
                myEKF.subStates['attitude']['stateObject'].eulerAngles()
                )
            }
        )
        
        TUError.append({
            't': tCurrent,
            'eulerAngles': eulerAngleDiff(
                currentEATrue,
                myTUOnly.subStates['attitude']['stateObject'].eulerAngles()
                )
            }
        )
        attitudeMat = attitude(photonMeasurement['t']['value']).rotation_matrix
        myRaDec = me.utils.spacegeometry.unitVector2RaDec(
            attitudeMat.dot(JPDAFAtt.sidUnitVec(photonMeasurement))
        )
        photonMeasurement['TrueRA'] = myRaDec[0]
        photonMeasurement['TrueDEC'] = myRaDec[1]
    
    # if nextStarIndex < nStars:
    #     starVecBodyFrame = (
    #         q0.rotation_matrix.transpose().dot(starVecs[nextStarIndex])
    #         )
    # else:
    #     starVecBodyFrame = np.random.normal(np.zeros(3))
    #     starVecBodyFrame = starVecBodyFrame/np.linalg.norm(starVecBodyFrame)

    # starRa, starDec = unitVector2RaDec(starVecBodyFrame)
    # starRa = starRa + np.random.normal(0, np.sqrt(RScalar))
    # starDec = starDec + np.random.normal(0, np.sqrt(RScalar))

    # photonMeasurement['RA']['var'] = RScalar
    # photonMeasurement['DEC']['var'] = RScalar
    # photonMeasurement['t']['var'] = 1e-1000

    myEKF.measurementUpdateEKF(photonMeasurement, photonMeasurement['name'])
    # if photonMeasurement['name'] != 'background':
    #     1/0
    myML.measurementUpdateML(photonMeasurement)
        
    myJPDAF.measurementUpdateJPDAF(photonMeasurement)
    # photonMeasurement['associationProbabilities']=myJPDAF.computeAssociationProbabilities(photonMeasurement)

    # starArrivalTimes[nextStarIndex] = (
    #     starArrivalTimes[nextStarIndex] +
    #     exponential(1/fluxes[nextStarIndex])
    # )
    if tCurrent-lastPrintTime > printFrequency:
        print(np.floor(tCurrent))
        lastPrintTime = np.floor(tCurrent)
plt.figure()
plt.title('Euler angles')
plt.subplot(311)
plt.plot(
    [sv['t'] for sv in JPDAFAtt.stateVectorHistory],
    [sv['eulerAngles'][0] for sv in JPDAFAtt.stateVectorHistory],
    label='JPDAF'
)
plt.plot(
    [sv['t'] for sv in MLAtt.stateVectorHistory],
    [sv['eulerAngles'][0] for sv in MLAtt.stateVectorHistory],
    label='ML'
)
plt.plot(
    [sv['t'] for sv in EKFAtt.stateVectorHistory],
    [sv['eulerAngles'][0] for sv in EKFAtt.stateVectorHistory],
    label='Ideal'
)
# plt.plot(
#     [sv['t'] for sv in TUOnlyAtt.stateVectorHistory],
#     [sv['eulerAngles'][0] for sv in TUOnlyAtt.stateVectorHistory],
#     ls='-.'
# )
plt.plot(
    [eu['t'] for eu in eulerAnglesTrue],
    [eu['eulerAngles'][0] for eu in eulerAnglesTrue],
    ls='-',
    label='True'
)
plt.legend()

plt.subplot(312)
plt.plot(
    [sv['t'] for sv in JPDAFAtt.stateVectorHistory],
    [sv['eulerAngles'][1] for sv in JPDAFAtt.stateVectorHistory],
    label='JPDAF'
)
plt.plot(
    [sv['t'] for sv in MLAtt.stateVectorHistory],
    [sv['eulerAngles'][1] for sv in MLAtt.stateVectorHistory],
    label='ML'
)
plt.plot(
    [sv['t'] for sv in EKFAtt.stateVectorHistory],
    [sv['eulerAngles'][1] for sv in EKFAtt.stateVectorHistory],
    label='Ideal'
)
# plt.plot(
#     [sv['t'] for sv in TUOnlyAtt.stateVectorHistory],
#     [sv['eulerAngles'][1] for sv in TUOnlyAtt.stateVectorHistory],
#     ls='-.'
# )
plt.plot(
    [eu['t'] for eu in eulerAnglesTrue],
    [eu['eulerAngles'][1] for eu in eulerAnglesTrue],
    ls='-'
)


plt.subplot(313)
plt.plot(
    [sv['t'] for sv in JPDAFAtt.stateVectorHistory],
    [sv['eulerAngles'][2] for sv in JPDAFAtt.stateVectorHistory],
    label='JPDAF'
)
plt.plot(
    [sv['t'] for sv in MLAtt.stateVectorHistory],
    [sv['eulerAngles'][2] for sv in MLAtt.stateVectorHistory],
    label='ML'
)
plt.plot(
    [sv['t'] for sv in EKFAtt.stateVectorHistory],
    [sv['eulerAngles'][2] for sv in EKFAtt.stateVectorHistory],
    label='Ideal'
)
# plt.plot(
#     [sv['t'] for sv in TUOnlyAtt.stateVectorHistory],
#     [sv['eulerAngles'][2] for sv in TUOnlyAtt.stateVectorHistory],
#     ls='-.'
# )
plt.plot(
    [eu['t'] for eu in eulerAnglesTrue],
    [eu['eulerAngles'][2] for eu in eulerAnglesTrue],
    ls='-'
)

plt.show(block=False)

EKFT=[sv['t'] for sv in EKFAtt.stateVectorHistory]
rollSTD = np.array([sv['eulerSTD'][0] for sv in EKFAtt.stateVectorHistory])
pitchSTD = np.array([sv['eulerSTD'][1] for sv in EKFAtt.stateVectorHistory])
yawSTD = np.array([sv['eulerSTD'][2] for sv in EKFAtt.stateVectorHistory])

plt.figure()
plt.subplot(311)
# plt.plot(myJPDAF.subStates['attitude']['stateObject'].eulerAngleVec['t'],
#          myJPDAF.subStates['attitude']['stateObject'].eulerAngleVec['eulerSTD'][:,0]
#          )
plt.plot(
    [je['t'] for je in JPDAFError],
    [je['eulerAngles'][0] for je in JPDAFError],
    label='JPDAF'
)
plt.plot(
    [je['t'] for je in MLError],
    [je['eulerAngles'][0] for je in MLError],
    label='ML'
)
plt.plot(
    [je['t'] for je in EKFError],
    [je['eulerAngles'][0] for je in EKFError],
    label='Ideal'
)
plt.plot(
    EKFT,
    rollSTD, color=[0.5,0.5,0.5]
)
plt.plot(
    EKFT,
    -rollSTD, color=[0.5,0.5,0.5]
)
# plt.plot(
#     [je['t'] for je in TUError],
#     [je['eulerAngles'][0] for je in TUError],
#     label='Time update only'
# )

plt.legend()
plt.subplot(312)
plt.plot(
    [je['t'] for je in JPDAFError],
    [je['eulerAngles'][1] for je in JPDAFError],
    label='JPDAF'
)
plt.plot(
    [je['t'] for je in MLError],
    [je['eulerAngles'][1] for je in MLError],
    label='ML'
)
plt.plot(
    [je['t'] for je in EKFError],
    [je['eulerAngles'][1] for je in EKFError],
    label='Ideal'
)
plt.plot(
    EKFT,
    pitchSTD, color=[0.5,0.5,0.5]
)
plt.plot(
    EKFT,
    -pitchSTD, color=[0.5,0.5,0.5]
)

# plt.plot(
#     [je['t'] for je in TUError],
#     [je['eulerAngles'][1] for je in TUError],
#     label='Time update only'
# )

plt.subplot(313)
plt.plot(
    [je['t'] for je in JPDAFError],
    [je['eulerAngles'][2] for je in JPDAFError],
    label='JPDAF'
)
plt.plot(
    [je['t'] for je in MLError],
    [je['eulerAngles'][2] for je in MLError],
    label='ML'
)
plt.plot(
    [je['t'] for je in EKFError],
    [je['eulerAngles'][2] for je in EKFError],
    label='Ideal'
)
plt.plot(
    EKFT,
    yawSTD, color=[0.5,0.5,0.5]
)
plt.plot(
    EKFT,
    -yawSTD, color=[0.5,0.5,0.5]
)

# plt.plot(
#     [je['t'] for je in TUError],
#     [je['eulerAngles'][2] for je in TUError],
#     label='Time update only'
# )

plt.show(block=False)

me.plots.photonscatterplot.plotSourcesAndProbabilities(myJPDAF, photonArrivals, pointSize=100, ignoreBackground=False,plotAttitude=False)
