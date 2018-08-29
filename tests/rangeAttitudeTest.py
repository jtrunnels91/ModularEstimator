from context import modest as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from operator import itemgetter

from pulsarData.loadPulsarData import loadPulsarData
np.random.seed(0)


plt.close('all')
pulsarList = ['J0534+2200']
#pulsarList = ['J0030+0451', 'J0437-4715', 'B1937+21', 'B1957+20', 'B1821-24']
pulsarList = ['J0437-4715']
#pulsarList = ['B1957+20']
#pulsarList = ['B1937+21']
#pulsarList=['B1821-24']
pulsarDir = './pulsarData/'
pulsarCatalogFileName = 'pulsarCatalog.txt'
tFinal = 1000

orbitPeriod = 100/(2*np.pi)
orbitAmplitude = 1000
vVar = np.square(0.01) # km^2/s^2
omegaVar = np.square(1e-10) # rad^2/s^2
AOAVar = np.square(1e-10) # rad^2
initialAttitudeSigma= 1e-9 * np.pi/180.0 #rad

nTaps = 9

detectorArea = 1000  # cm^2
detectorFOV = 1
pulsarObjectDict = loadPulsarData(detectorArea=detectorArea)
pulsarPeriod = pulsarObjectDict[pulsarList[0]].pulsarPeriod
constantOffset = np.random.uniform(0, pulsarPeriod)


angularVelocity = [0, 0, 0]
pulsarRaDec = pulsarObjectDict[pulsarList[0]].__RaDec__

# pointSources = md.utils.accessPSC.xamin_coneSearch(
#     pulsarRaDec['RA'] * 180.0/np.pi,
#     pulsarRaDec['DEC'] * 180.0/np.pi,
#     FOV=detectorFOV,
#     catalog='rass2rxs',
#     fluxKey='PLaw_Flux'
#     #minSignificance=20
#     )
pointSources = md.utils.accessPSC.xamin_coneSearch(
    pulsarRaDec['RA'] * 180.0/np.pi,
    pulsarRaDec['DEC'] * 180.0/np.pi,
    FOV=detectorFOV,
    catalog='xmmslewcln',
    fluxKey='flux_b8'
    #minSignificance=20
    )


def attitude(t, returnQ=True):
    if hasattr(t, '__len__'):
        attitudeArray = []
        for i in range(len(t)):
            attitudeArray.append(attitude(t[i],returnQ))
        return attitudeArray
    else:
        #eulerAngles = [t * angularVelocity[0], t* angularVelocity[1], t* angularVelocity[2]]
        eulerAngles = [
            0,
            -pulsarRaDec['DEC'],
            pulsarRaDec['RA']
            ]
        if returnQ:
            return md.utils.euler2quaternion(eulerAngles)
        else:
            return(eulerAngles)

def omega(t):
    return(angularVelocity)

def position(t):
    return(
        np.array([
            orbitAmplitude * np.cos(t/orbitPeriod),
            orbitAmplitude * np.sin(t/orbitPeriod),
            0 * t
        ])
    )

def velocity(t):
    return(
        (orbitAmplitude/orbitPeriod) *
        np.array([
            -np.sin(t/orbitPeriod),
            np.cos(t/orbitPeriod),
            0 * t
            ]
        )
    )

corrSubstateDict = {}
photonMeasurements = []
covarianceStorageMethod='covariance'
#updateMethod = 'EKF'
updateMethod = 'JPDAF'

myFilter = md.ModularFilter(covarianceStorage=covarianceStorageMethod)

# myPointSource = md.signals.StaticXRayPointSource(
#     pulsarRaDec['RA'] + 0.001,
#     pulsarRaDec['DEC'],
#     photonEnergyFlux=1e-15,
#     name='Dummy'
#     )

pointSourceObjectDict = {}

for signalIndex in range(len(pointSources)):
    myRow = pointSources.iloc[signalIndex]
    myRa = md.utils.spacegeometry.hms2rad(hms=myRow['ra'])
    myDec = md.utils.spacegeometry.dms2rad(dms=myRow['dec'])
    try:
        myFlux = float(myRow['flux'])
        print('Initializing static point source %s.' %myRow['name'])
    except:
        print('Point source %s had invalid flux.  Skipping.' %myRow['name'])

    if myFlux > 0.0 and myFlux < 1e-10:

        if (
                (np.abs(pulsarRaDec['RA'] - myRa) > 1e-9) and
                (np.abs(pulsarRaDec['DEC'] - myDec) > 1e-9) and
                'null' not in myRow['name']
        ):
            
            pointSourceObjectDict[myRow['name']] = (
                md.signals.StaticXRayPointSource(
                    myRa,
                    myDec,
                    photonEnergyFlux=myFlux,
                    detectorArea=detectorArea,
                    name=myRow['name']
                    )
                )
            photonMeasurements+=pointSourceObjectDict[myRow['name']].generatePhotonArrivals(
                tFinal,
                attitude=attitude
                )
            try:
                if myFlux > 1e-14:
                    myFilter.addSignalSource(myRow['name'],pointSourceObjectDict[myRow['name']])
            except:
                print('The signal source %s has already been added.  Skipping.' %myRow['name'])
            

# Generate photon arrivals for each pulsar in the list
for pulsarName in pulsarList:
    photonMeasurements += pulsarObjectDict[pulsarName].generatePhotonArrivals(
            tFinal,
            position=position,
            attitude=attitude
        )

    corrSubstateDict[pulsarName] = md.substates.CorrelationVector(
        pulsarObjectDict[pulsarName],
        nTaps,
        pulsarObjectDict[pulsarName].pulsarPeriod/(nTaps+1),
        signalTDOA=0,
        TDOAVar=np.square(pulsarObjectDict[pulsarName].pulsarPeriod),
        measurementNoiseScaleFactor=3,
        processNoise=1e-15,
        centerPeak=True,
        peakLockThreshold=0.05,
        covarianceStorage=covarianceStorageMethod
        )

    myFilter.addSignalSource(pulsarObjectDict[pulsarName].name, pulsarObjectDict[pulsarName])
    myFilter.addStates(pulsarObjectDict[pulsarName].name, corrSubstateDict[pulsarName])



backgroundNoise = md.signals.UniformNoiseXRaySource(
    detectorArea=detectorArea,
    detectorFOV=detectorFOV
)

photonMeasurements += backgroundNoise.generatePhotonArrivals(tFinal)

photonMeasurements = sorted(photonMeasurements, key=lambda k: k['t']['value'])

initialAttitude = md.utils.euler2quaternion(
    attitude(0, returnQ=False) + np.random.normal(0, scale=initialAttitudeSigma, size=3)
    )
if covarianceStorageMethod == 'covariance':
    myAttitude = md.substates.Attitude(
        attitudeQuaternion=initialAttitude,
        attitudeErrorCovariance=np.eye(3)*np.square(initialAttitudeSigma),
        gyroBiasCovariance=np.eye(3)*1e-100,
        covarianceStorage=covarianceStorageMethod
    )
else:
    myAttitude = md.substates.Attitude(
        attitudeQuaternion=initialAttitude,
        attitudeErrorCovariance=np.eye(3)*initialAttitudeSigma,
        gyroBiasCovariance=np.eye(3)*np.sqrt(1e-100),
        covarianceStorage=covarianceStorageMethod
    )    
myFilter.addStates('attitude', myAttitude)
myFilter.addSignalSource('background', backgroundNoise)

# myMeas = {
#     't': {'value': 0}
# }
# myFilter.measurementUpdateEKF(myMeas, myPulsar.name)

lastUpdateTime = 0
lastT = 0

timeUpdateOnlyTDOA = []
timeUpdateOnlyT = []

for photonMeas in photonMeasurements:
    arrivalT = photonMeas['t']['value']
    vMeas = velocity(arrivalT) + np.random.normal(0,scale=np.sqrt(vVar),size=3)
    omegaMeas = omega(arrivalT) + np.random.normal(0,scale=np.sqrt(omegaVar),size=3)
    
    dynamics = {
        'velocity': {'value': vMeas, 'var': np.eye(3)*vVar},
        'omega': {'value': omega(arrivalT), 'var': np.eye(3) * omegaVar},
    }
    
    myFilter.timeUpdateEKF(arrivalT-lastT, dynamics=dynamics)
#    if myCorrelation.peakLock is True:
#        myCorrelation.realTimePlot()
    #myFilter.timeUpdateEKF(photon-lastT)
    # print(photonMeas)
    photonMeas['RA']['value'] = np.random.normal(photonMeas['RA']['value'], np.sqrt(AOAVar))
    photonMeas['DEC']['value'] = np.random.normal(photonMeas['DEC']['value'], np.sqrt(AOAVar))
    photonMeas['RA']['var'] = AOAVar
    photonMeas['DEC']['var'] = AOAVar
    photonMeas['t']['var'] = 1e-20
    photonMeas['t']['value'] -= constantOffset

    if updateMethod == 'EKF':
        myFilter.measurementUpdateEKF(photonMeas, photonMeas['name'])
    elif updateMethod == 'JPDAF':
        myFilter.measurementUpdateJPDAF(photonMeas)
    if (arrivalT-lastUpdateTime) > 100:
        lastUpdateTime = int(arrivalT)
        print('time: %f\test TDOA: %f\ttrue TDOA %f' %
              (
                  arrivalT,
                  constantOffset,
                  corrSubstateDict[pulsarList[0]].stateVectorHistory[-1]['signalTDOA']
              )
        )
        myFilter.realTimePlot()
        # for key in corrSubstateDict:
        #     corrSubstateDict[key].realTimePlot()
    lastT = arrivalT

# plt.figure()
# nPanels = len(corrSubstateDict)
# panelCounter = 0

# for key in corrSubstateDict:

#     subpanel=plt.subplot2grid((nPanels,1),(panelCounter,0))
#     panelCounter = panelCounter+1
#     myCorrelation = corrSubstateDict[key]
#     subpanel.set_title('Phase error %s' %key)
#     # subpanel.plot(
#     #     myCorrelation.stateVectorHistory['t'],
#     #     trueDelay
#     # )
#     C = myFilter.signalSources[key].speedOfLight()
#     subpanel.plot(
#         myCorrelation.stateVectorHistory['t'],
#         md.utils.spacegeometry.phaseError(
#             myCorrelation.stateVectorHistory['signalTDOA'],
#             constantOffset,
#             pulsarPeriod
#             ) * C
#     )
#     subpanel.plot(
#         myCorrelation.stateVectorHistory['t'],
#         np.sqrt(myCorrelation.stateVectorHistory['TDOAVar']) * C
#     )
#     subpanel.plot(
#         myCorrelation.stateVectorHistory['t'],
#         -np.sqrt(myCorrelation.stateVectorHistory['TDOAVar']) * C
#     )
#     if np.any(np.abs(subpanel.get_ylim()) > 2 * pulsarObjectDict[key].pulsarPeriod):
#         subpanel.set_ylim(-2*pulsarObjectDict[key].pulsarPeriod * C, 2*pulsarObjectDict[key].pulsarPeriod * C)
#     subpanel.grid()
#     subpanel.set_title(key)
# plt.show(block=False)


# scatterPlot = plt.figure()
# attitudeMatrix = attitude(0).rotation_matrix.transpose()
# # plt.title('Photon scatter plot')

# for signal in myFilter.signalSources:
#     if hasattr(myFilter.signalSources[signal], '__RaDec__'):
#         unitVec = attitudeMatrix.dot(myFilter.signalSources[signal].unitVec())
#         RaDec = md.utils.spacegeometry.unitVector2RaDec(unitVec)
#         plt.scatter(
#             RaDec[0],
#             RaDec[1],
#             marker='^'
#         )
# plt.scatter(
#         [p['RA']['value'] for p in photonMeasurements],
#         [p['DEC']['value'] for p in photonMeasurements],
#         marker='.', s=10, alpha=.2)

# plt.show(block=False)

# eulerErrorPlot = plt.figure()
# # plt.title('euler angle error')
# eulerAnglesTrue=np.array(attitude(myAttitude.stateVectorHistory['t'],returnQ=False))
# eulerAnglesEst=np.array(myAttitude.stateVectorHistory['eulerAngles'])

# for eulerAngleIndex in range(3):
#     subpanel=plt.subplot2grid((3,1),(eulerAngleIndex,0))
#     subpanel.plot(myAttitude.stateVectorHistory['t'],
#                   md.utils.QuaternionHelperFunctions.eulerAngleDiff(
#                       eulerAnglesEst[:,eulerAngleIndex],
#                       eulerAnglesTrue[:, eulerAngleIndex]
#                       )
#                   )
#     subpanel.grid()
# plt.show(block=False)
