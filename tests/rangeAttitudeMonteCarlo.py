from context import modest as md
#import matplotlib.pyplot as plt
import numpy as np
from pulsarData.loadPulsarData import loadPulsarData

from pypet import Environment, cartesian_product


def run4DOFSimulation(traj):

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

    for detectorArea in detectorAreaArray:
        for randomIndex in range(nSimulations):
            pulsarSourceObjectDict = loadPulsarData(detectorArea=detectorArea)

            myPulsarObject = pulsarSourceObjectDict[pulsarName]
            constantOffset = np.random.uniform(0, myPulsarObject.pulsarPeriod)

            pointSources = md.utils.accessPSC.xamin_coneSearch(
                pulsarRaDec['RA'] * 180.0/np.pi,
                pulsarRaDec['DEC'] * 180.0/np.pi,
                FOV=detectorFOV,
                catalog='xmmslewcln',
                fluxKey='flux_b8'
                #minSignificance=20
                )

            pointSourceObjectDict = {}

            photonMeasurements = []

            myFilter = md.ModularFilter()

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
                            (np.abs(pulsarRaDec['DEC'] - myDec) > 1e-9)
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
            photonMeasurements += myPulsarObject.generatePhotonArrivals(
                tFinal,
                position=position,
                attitude=attitude
            )

            correlationSubstate = md.substates.CorrelationVector(
                myPulsarObject,
                nTaps,
                myPulsarObject.pulsarPeriod/(nTaps+1),
                signalTDOA=0,
                TDOAVar=np.square(myPulsarObject.pulsarPeriod),
                measurementNoiseScaleFactor=1.0,
                processNoise=1e-12*detectorArea,
                centerPeak=True,
                peakLockThreshold=0.05,
            )

            myFilter.addSignalSource(myPulsarObject.name, myPulsarObject)
            myFilter.addStates(myPulsarObject.name, correlationSubstate)

            backgroundNoise = md.signals.UniformNoiseXRaySource(
                detectorArea=detectorArea,
                detectorFOV=detectorFOV
            )

            photonMeasurements += backgroundNoise.generatePhotonArrivals(tFinal)

            photonMeasurements = sorted(photonMeasurements, key=lambda k: k['t']['value'])

            initialAttitude = md.utils.euler2quaternion(
                attitude(0, returnQ=False) + np.random.normal(0, scale=initialAttitudeSigma, size=3)
                )
            myAttitude = md.substates.Attitude(
                attitudeQuaternion=initialAttitude,
                attitudeErrorCovariance=np.eye(3)*np.square(initialAttitudeSigma),
                gyroBiasCovariance=np.eye(3)*1e-100)
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

                #myFilter.measurementUpdateEKF(photonMeas, photonMeas['name'])
                myFilter.measurementUpdateJPDAF(photonMeas)
                if (arrivalT-lastUpdateTime) > 100:
                    lastUpdateTime = int(arrivalT)
                    estimatedDelay = correlationSubstate.stateVectorHistory[-1]['signalTDOA']
                    delayError = md.utils.spacegeometry.phaseError(
                        estimatedDelay,
                        constantOffset,
                        myPulsarObject.pulsarPeriod
                    )
                    print('Area: %i \tTime: %f \tTrue TDOA %f\tEst TDOA %f,Position error: %f km' %
                          (
                              detectorArea,
                              arrivalT,
                              constantOffset,
                              estimatedDelay,
                              delayError * myPulsarObject.speedOfLight()
                          )
                    )
                    myFilter.realTimePlot()
                    # for key in corrSubstateDict:
                    #     corrSubstateDict[key].realTimePlot()
                lastT = arrivalT

        # plt.close('all')


        # plt.figure()

        # C = myPulsarObject.speedOfLight()
        # plt.plot(
        #     correlationSubstate.stateVectorHistory['t'],
        #     (correlationSubstate.stateVectorHistory['signalTDOA'] - constantOffset) * C
        # )
        # plt.plot(
        #     correlationSubstate.stateVectorHistory['t'],
        #     np.sqrt(correlationSubstate.stateVectorHistory['TDOAVar']) * C
        # )
        # plt.plot(
        #     correlationSubstate.stateVectorHistory['t'],
        #     -np.sqrt(correlationSubstate.stateVectorHistory['TDOAVar']) * C
        # )
        # ax = plt.gca()
        # if np.any(np.abs(ax.get_ylim()) > 2 * myPulsarObject.pulsarPeriod):
        #     ax.set_ylim(-2*myPulsarObject.pulsarPeriod * C, 2*myPulsarObject.pulsarPeriod * C)
        # plt.grid()
        # plt.show(block=False)

        # plt.figure()
        # attitudeMatrix = attitude(0).rotation_matrix.transpose()

        # plt.scatter(
        #         [p['RA']['value'] for p in photonMeasurements],
        #         [p['DEC']['value'] for p in photonMeasurements],
        #         marker='.', s=10, alpha=.2)

        # for signal in myFilter.signalSources:
        #     if hasattr(myFilter.signalSources[signal], '__RaDec__'):
        #         unitVec = attitudeMatrix.dot(myFilter.signalSources[signal].unitVec())
        #         RaDec = md.utils.spacegeometry.unitVector2RaDec(unitVec)
        #         plt.scatter(
        #             RaDec[0],
        #             RaDec[1],
        #             marker='^'
        #         )

        # plt.show(block=False)

        # plt.figure()
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


tFinal = 3000
nSimulations = 1
        
pulsarName = 'J0437-4715'
#pulsarList = ['B1957+20']
#pulsarList = ['B1937+21']
#pulsarList=['B1821-24']
pulsarDir = './pulsarData/'
pulsarCatalogFileName = 'pulsarCatalog.txt'
basePulsarObjectDict = loadPulsarData(detectorArea=1)
pulsarRaDec = basePulsarObjectDict[pulsarName].__RaDec__

# Detector Information
detectorAreaArray = np.logspace(2,3,3)  # cm^2
nTaps = 15
detectorFOV = 1
AOAVar = np.square(1e-4) # rad^2

# Trajectory Information
orbitPeriod = 100/(2*np.pi)
orbitAmplitude = 0
vVar = np.square(1e-4) # km^2/s^2

angularVelocity = [0, 0, 0]
omegaVar = np.square(1e-6) # rad^2/s^2
initialAttitudeSigma = 0.05 * np.pi/180.0 #rad


