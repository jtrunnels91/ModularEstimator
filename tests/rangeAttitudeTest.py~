from context import modest as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from operator import itemgetter

from pulsarData.loadPulsarData import loadPulsarData


plt.close('all')
#pulsarList = ['J0534+2200']
pulsarList = ['J0030+0451', 'J0437-4715', 'B1937+21', 'B1957+20', 'B1821-24']
#pulsarList = ['J0030+0451', 'J0437-4715']
pulsarDir = './pulsarData/'
pulsarCatalogFileName = 'pulsarCatalog.txt'
pulsarCatalog = pd.read_csv(pulsarDir + pulsarCatalogFileName)

tFinal = 100

orbitPeriod = 100/(2*np.pi)
orbitAmplitude = 0
vVar = np.square(1e-3)
nTaps = 9

detectorArea = 10000  # cm^2


def attitude(t, returnQ=True):
    if hasattr(t, '__len__'):
        attitudeArray = []
        for i in range(len(t)):
            attitudeArray.append(attitude(t[i],returnQ))
        return attitudeArray
    else:
        eulerAngles = [t/100, t*0, t*0]
        if returnQ:
            return md.utils.euler2quaternion(eulerAngles)
        else:
            return(eulerAngles)

def omega(t):
    return([1/100, 0, 0])

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

pulsarObjectDict = loadPulsarData(detectorArea=detectorArea)
corrSubstateDict = {}
photonMeasurements = []

myFilter = md.ModularFilter()


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
        pulsarObjectDict[pulsarName].pulsarPeriod/nTaps,
        signalTDOA=0,
        TDOAVar=np.square(pulsarObjectDict[pulsarName].pulsarPeriod),
        measurementNoiseScaleFactor=2,
        processNoise=1e-20,
        centerPeak=True,
        peakLockThreshold=0.1,
        )

    myFilter.addSignalSource(pulsarObjectDict[pulsarName].name, pulsarObjectDict[pulsarName])
    myFilter.addStates(pulsarObjectDict[pulsarName].name, corrSubstateDict[pulsarName])

photonMeasurements = sorted(photonMeasurements, key=lambda k: k['t']['value'])

myAttitude = md.substates.Attitude()
myFilter.addStates('attitude', myAttitude)

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
    
    dynamics = {
        'velocity': {'value': vMeas, 'var': np.eye(3)*vVar},
        'omega': {'value': omega(arrivalT), 'var': np.eye(3) * 1e-100},
    }
    
    myFilter.timeUpdateEKF(arrivalT-lastT, dynamics=dynamics)
#    if myCorrelation.peakLock is True:
#        myCorrelation.realTimePlot()
    #myFilter.timeUpdateEKF(photon-lastT)
    # print(photonMeas)
    photonMeas['RA']['var'] = 1e-6
    photonMeas['DEC']['var'] = 1e-6
    photonMeas['t']['var'] = 1e-20
    myFilter.measurementUpdateEKF(photonMeas, photonMeas['name'])
    #myFilter.measurementUpdateJPDAF(photonMeas)
    if (arrivalT-lastUpdateTime) > 5:
        lastUpdateTime = int(arrivalT)
        print('time: %f' % arrivalT)
        myFilter.realTimePlot()
        # for key in corrSubstateDict:
        #     corrSubstateDict[key].realTimePlot()
    lastT = arrivalT

plt.figure()
nPanels = len(corrSubstateDict)
panelCounter = 0
for key in corrSubstateDict:

    subpanel=plt.subplot2grid((nPanels,1),(panelCounter,0))
    panelCounter = panelCounter+1
    myCorrelation = corrSubstateDict[key]
    # subpanel.plot(
    #     myCorrelation.stateVectorHistory['t'],
    #     trueDelay
    # )
    subpanel.plot(
        myCorrelation.stateVectorHistory['t'],
        myCorrelation.stateVectorHistory['signalTDOA']
    )
    subpanel.plot(
        myCorrelation.stateVectorHistory['t'],
        np.sqrt(myCorrelation.stateVectorHistory['TDOAVar'])
    )
    subpanel.plot(
        myCorrelation.stateVectorHistory['t'],
        -np.sqrt(myCorrelation.stateVectorHistory['TDOAVar'])
    )
    if np.any(np.abs(subpanel.get_ylim()) > 2 * pulsarObjectDict[key].pulsarPeriod):
        subpanel.set_ylim(-2*pulsarObjectDict[key].pulsarPeriod, 2*pulsarObjectDict[key].pulsarPeriod)
    subpanel.set_title(key)
plt.show(block=False)
