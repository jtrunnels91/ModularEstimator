import numpy as np
import matplotlib.pyplot as plt
import sys
# from smartpanda import SmartPanda
import modest as me

from numpy import sin, cos, pi, sqrt
from numpy.random import exponential

sys.path.append("/home/joel/Documents/astroSourceTracking/libraries")
# sys.path.append("/home/joel/Documents/astroSourceTracking/libraries/ModularFilter")

from SpaceGeometry import sidUnitVec, unitVector2RaDec
from QuaternionHelperFunctions import euler2quaternion, quaternion2euler, eulerAngleDiff

plt.close('all')

# Function defining angular velocity
def omega(t):
    # omegaT = np.array([sin(pi * t/4), cos(pi * t/12), sin(pi * t/16)])
    #omegaT = np.random.normal(np.zeros(3))
    omegaT = 0*np.array([cos(pi*t/4)*pi/4, 0, 0])
    return(omegaT)
def attitude(t,returnQ=True):
    euler = 0*np.array([sin(pi * t/4), 0, 0])
    if returnQ:
        return euler2quaternion(euler)
    else:
        return euler
printFrequency=1.0
lastPrintTime=0

# Dynamics info
tCurrent = 0
timeStep = 0.01
tMax = 100

biasSTD = 0.01
eulerT0True = np.array([0, 0, 0])
biasTrue = np.random.normal(np.zeros(3), scale=biasSTD)

rollErrorStd = 1e-9
RAErrorStd = 1e-1
DecErrorStd = 1e-1
biasErrorSTD = 1e-100

eulerT0Est = np.array([
    np.random.normal(eulerT0True[0], rollErrorStd),
    np.random.normal(eulerT0True[1], DecErrorStd),
    np.random.normal(eulerT0True[2], RAErrorStd)
])
biasEst = np.random.normal(biasTrue, scale=biasErrorSTD)

q0 = euler2quaternion(eulerT0True)

QScalar = 1e-9
RScalar = 1e-9


# Initiate filters
myJPDAF = me.ModularFilter(measurementValidationThreshold=0)
myEKF = me.ModularFilter()
myML = me.ModularFilter(measurementValidationThreshold=1e-3)
myTUOnly = me.ModularFilter()
initialAttitudeCovariance = np.zeros([3,3])
initialAttitudeCovariance[0,0] = np.square(rollErrorStd)
initialAttitudeCovariance[1,1] = np.square(DecErrorStd)
initialAttitudeCovariance[2,2] = np.square(RAErrorStd)


# Initiate attitude stubstates
JPDAFAtt = me.substates.Attitude(
    euler2quaternion(eulerT0Est),
    initialAttitudeCovariance,
    np.zeros(3),
    np.eye(3) * np.square(biasSTD + biasErrorSTD)
)

EKFAtt = me.substates.Attitude(
    euler2quaternion(eulerT0Est),
    initialAttitudeCovariance,
    np.zeros(3),
    np.eye(3) * np.square(biasSTD + biasErrorSTD)
)

MLAtt = me.substates.Attitude(
    euler2quaternion(eulerT0Est),
    initialAttitudeCovariance,
    np.zeros(3),
    np.eye(3) * np.square(biasSTD + biasErrorSTD)
)

TUOnlyAtt = me.substates.Attitude(
    euler2quaternion(eulerT0Est),
    initialAttitudeCovariance,
    np.zeros(3),
    np.eye(3) * np.square(biasSTD + biasErrorSTD)
)

myJPDAF.addStates('attitude', JPDAFAtt)
myML.addStates('attitude', MLAtt)
myEKF.addStates('attitude', EKFAtt)
myTUOnly.addStates('attitude', TUOnlyAtt)

# Star and background info
backgroundFlux = 100
nStars = 5
starVecs = np.random.normal(np.zeros([nStars, 3]))
starCoordinates = np.zeros([nStars, 2])
fluxes = np.zeros(nStars + 1)
starArrivalTimes = np.zeros(nStars + 1)
bkgArrivalTime = exponential(1/backgroundFlux)

myNoise = me.signals.UniformNoiseXRaySource(backgroundFlux)
myJPDAF.addSignalSource('background', myNoise)
myML.addSignalSource('background', myNoise)
myEKF.addSignalSource('background', myNoise)


starList = []
photonArrivals = []
# Generate signal objects, add them to filters
for i in range(nStars):
    fluxes[i] = 1
    starVecs[i] = starVecs[i]/np.linalg.norm(starVecs[i])
    starCoordinates[i] = unitVector2RaDec(starVecs[i])
    name = 'star%i' %i
    star = me.signals.StaticXRayPointSource(
        starCoordinates[i][0],
        starCoordinates[i][1],
        fluxes[i],
        name=name
    )
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
            'gyroBias': {'var': 1e-2}
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

    myJPDAF.measurementUpdateJPDAF(photonMeasurement)
    myML.measurementUpdateML(photonMeasurement)
    myEKF.measurementUpdateEKF(photonMeasurement, photonMeasurement['name'])
        

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
plt.plot(
    [sv['t'] for sv in TUOnlyAtt.stateVectorHistory],
    [sv['eulerAngles'][0] for sv in TUOnlyAtt.stateVectorHistory],
    ls='-.'
)
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
plt.plot(
    [sv['t'] for sv in TUOnlyAtt.stateVectorHistory],
    [sv['eulerAngles'][1] for sv in TUOnlyAtt.stateVectorHistory],
    ls='-.'
)
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
plt.plot(
    [sv['t'] for sv in TUOnlyAtt.stateVectorHistory],
    [sv['eulerAngles'][2] for sv in TUOnlyAtt.stateVectorHistory],
    ls='-.'
)
plt.plot(
    [eu['t'] for eu in eulerAnglesTrue],
    [eu['eulerAngles'][2] for eu in eulerAnglesTrue],
    ls='-'
)

plt.show(block=False)

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
    label='JPDAF'
)
plt.plot(
    [je['t'] for je in EKFError],
    [je['eulerAngles'][0] for je in EKFError],
    label='JPDAF'
)
plt.plot(
    [je['t'] for je in TUError],
    [je['eulerAngles'][0] for je in TUError],
    label='JPDAF'
)

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
    label='JPDAF'
)
plt.plot(
    [je['t'] for je in EKFError],
    [je['eulerAngles'][1] for je in EKFError],
    label='JPDAF'
)
plt.plot(
    [je['t'] for je in TUError],
    [je['eulerAngles'][1] for je in TUError],
    label='JPDAF'
)

plt.subplot(313)
plt.plot(
    [je['t'] for je in JPDAFError],
    [je['eulerAngles'][2] for je in JPDAFError],
    label='JPDAF'
)
plt.plot(
    [je['t'] for je in MLError],
    [je['eulerAngles'][2] for je in MLError],
    label='JPDAF'
)
plt.plot(
    [je['t'] for je in EKFError],
    [je['eulerAngles'][2] for je in EKFError],
    label='JPDAF'
)
plt.plot(
    [je['t'] for je in TUError],
    [je['eulerAngles'][2] for je in TUError],
    label='JPDAF'
)

plt.show(block=False)

