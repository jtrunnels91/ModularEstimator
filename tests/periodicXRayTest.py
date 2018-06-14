import time
from context import modest as md
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter


t0 = time.time()

myProfile='./pulsarData/profiles/J0534+2200_profile.txt'
myPARFile = './pulsarData/PAR_files/ephem_J0534+2200_nancay_jodrell.par'

myFlux = 9.93e-9 # erg/cm^2/s

detectorArea = 100  # cm^2
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
angleVar = 1e-3
angleSTD = np.sqrt(angleVar)
myMeas = {
    't': {'value': 0, 'var': 1e-100},
    'RA': {'value': myPulsar.RaDec()['RA'] + np.random.normal(0, angleSTD), 'var': angleVar},
    'DEC': {'value': myPulsar.RaDec()['DEC'] + np.random.normal(0, angleSTD), 'var': angleVar}
}

state = 'a'

myCorr = md.substates.CorrelationVector(
    myPulsar,
    10,
    myPulsar.pulsarPeriod/10,
    signalTDOA=0,
    TDOAVar=1e-100
)
attitudeSubstate = md.substates.Attitude(
    attitudeErrorCovariance=np.eye(3)*1e-1,
    gyroBiasCovariance=np.eye(3)*1e-10
)
stateDict = {
    myPulsar.name: {
        'stateObject': myCorr,
        'length': myCorr.dimension()
    },
    'attitude': {
        'stateObject': attitudeSubstate,
        'length': attitudeSubstate.dimension()
    }
}

myPr = myPulsar.computeAssociationProbability(myMeas, stateDict)
print(myPr)

unitVec = myPulsar.unitVec()
speedOfLight = myPulsar.speedOfLight()

def position(t):
    return unitVec * speedOfLight * myPulsar.pulsarPeriod * (1/4)
    

periodSim = 1000
simTime = myPulsar.pulsarPeriod * periodSim

myPhotonArrivals = myPulsar.generatePhotonArrivals(simTime, position=position)

nHistBins = 200
hist, binEdges = np.histogram(np.mod([x['t']['value'] for x in myPhotonArrivals], myPulsar.pulsarPeriod), bins=nHistBins)

hist = (hist) * (nHistBins/simTime)
binCenters = (binEdges[:-1] + binEdges[1:])/2

myFig = plt.figure()
plt.step(binCenters, hist)
plt.show(block=False)

myPulsar.plot(figureHandle=myFig)
plt.show(block=False)


# Integral test
tSteps=50
nPeriods=3
tArray = np.linspace(0, myPulsar.pulsarPeriod * nPeriods, tSteps)
dT = myPulsar.pulsarPeriod * nPeriods / tSteps
myIntegral = []
for tIndex in range(len(tArray)):
    myIntegral.append(myPulsar.signalIntegral(tArray[tIndex] - dT/2.0, tArray[tIndex] + dT/2.0))

plt.figure()
plt.plot(tArray, myIntegral)

plt.plot(tArray, myPulsar.getSignal(tArray) * dT)
plt.show(block=False)

t1 = time.time()

total = t1-t0
print('total run-time: %f' %total)