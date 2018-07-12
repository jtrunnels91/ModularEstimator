from context import modest as md
#import matplotlib.pyplot as plt
import numpy as np
from pulsarData.loadPulsarData import loadPulsarData

from pypet import Environment, cartesian_product, Trajectory




def run4DOFSimulation(traj):

    # pulsarDir = './pulsarData/'
    # pulsarCatalogFileName = 'pulsarCatalog.txt'

    pulsarObjectDict = loadPulsarData(detectorArea=traj.detectorArea)
    myPulsarObject = pulsarObjectDict[traj.pulsarName]
    pulsarRaDec = myPulsarObject.__RaDec__

    pointSources = md.utils.accessPSC.xamin_coneSearch(
        pulsarRaDec['RA'] * 180.0/np.pi,
        pulsarRaDec['DEC'] * 180.0/np.pi,
        FOV=traj.detectorFOV,
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
        return(traj.angularVelocity)

    def position(t):
        return(
            np.array([
                traj.orbitAmplitude * np.cos(t/traj.orbitPeriod),
                traj.orbitAmplitude * np.sin(t/traj.orbitPeriod),
                0 * t
            ])
        )

    def velocity(t):
        return(
            (traj.orbitAmplitude/traj.orbitPeriod) *
            np.array([
                -np.sin(t/traj.orbitPeriod),
                np.cos(t/traj.orbitPeriod),
                0 * t
                ]
            )
        )

    arrivalT = 1
    lastT = 0
    successfulRun = False
    vDrift = 0            
    pulsarUnitVector = myPulsarObject.unitVec()
    vMeas = velocity(arrivalT) + np.random.normal(0, scale=np.sqrt(traj.vVar), size=3)
    vDrift += vMeas.dot(pulsarUnitVector) * (arrivalT - lastT)
    
    arrivalT = 0
    lastT = 0
    
    while not successfulRun:
        # try:
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
                            detectorArea=traj.detectorArea,
                            name=myRow['name']
                            )
                        )
                    photonMeasurements+=pointSourceObjectDict[myRow['name']].generatePhotonArrivals(
                        traj.runtime,
                        attitude=attitude
                        )
                    try:
                        if myFlux > 1e-14:
                            myFilter.addSignalSource(myRow['name'],pointSourceObjectDict[myRow['name']])
                    except:
                        print('The signal source %s has already been added.  Skipping.' %myRow['name'])


        # Generate photon arrivals for each pulsar in the list
        photonMeasurements += myPulsarObject.generatePhotonArrivals(
            traj.runtime,
            position=position,
            attitude=attitude
        )

        if traj.scaleProcessNoise is True:
            processNoise = traj.processNoise * traj.detectorArea
        else:
            processNoise = traj.processNoise
        print('Pulsar object period:')
        print(myPulsarObject.pulsarPeriod)
        correlationSubstate = md.substates.CorrelationVector(
            myPulsarObject,
            traj.filterTaps,
            myPulsarObject.pulsarPeriod/(traj.filterTaps+1),
            signalTDOA=0,
            TDOAVar=myPulsarObject.pulsarPeriod,
            measurementNoiseScaleFactor=traj.measurementNoiseScaleFactor,
            processNoise=processNoise,
            centerPeak=True,
            peakLockThreshold=traj.peakLockThreshold,
        )

        myFilter.addSignalSource(myPulsarObject.name, myPulsarObject)
        myFilter.addStates(myPulsarObject.name, correlationSubstate)

        backgroundNoise = md.signals.UniformNoiseXRaySource(
            detectorArea=traj.detectorArea,
            detectorFOV=traj.detectorFOV
        )

        photonMeasurements += backgroundNoise.generatePhotonArrivals(traj.runtime)

        photonMeasurements = sorted(photonMeasurements, key=lambda k: k['t']['value'])

        initialAttitude = md.utils.euler2quaternion(
            attitude(0, returnQ=False) + np.random.normal(0, scale=traj.initialAttitudeSigma, size=3)
            )
        myAttitude = md.substates.Attitude(
            attitudeQuaternion=initialAttitude,
            attitudeErrorCovariance=np.eye(3)*np.square(traj.initialAttitudeSigma),
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

        constantOffset = traj.constantPhaseOffset * myPulsarObject.pulsarPeriod

        for photonMeas in photonMeasurements:
            arrivalT = photonMeas['t']['value']
            vMeas = velocity(arrivalT) + np.random.normal(0, scale=np.sqrt(traj.vVar), size=3)

            vDrift += vMeas.dot(pulsarUnitVector) * (arrivalT - lastT)

            omegaMeas = omega(arrivalT) + np.random.normal(0,scale=np.sqrt(traj.omegaVar),size=3)

            dynamics = {
                'velocity': {'value': vMeas, 'var': np.eye(3)*traj.vVar},
                'omega': {'value': omega(arrivalT), 'var': np.eye(3) * traj.omegaVar},
            }

            myFilter.timeUpdateEKF(arrivalT-lastT, dynamics=dynamics)
        #    if myCorrelation.peakLock is True:
        #        myCorrelation.realTimePlot()
            #myFilter.timeUpdateEKF(photon-lastT)
            # print(photonMeas)
            photonMeas['RA']['value'] = np.random.normal(photonMeas['RA']['value'], np.sqrt(traj.AOAVar))
            photonMeas['DEC']['value'] = np.random.normal(photonMeas['DEC']['value'], np.sqrt(traj.AOAVar))
            photonMeas['RA']['var'] = traj.AOAVar
            photonMeas['DEC']['var'] = traj.AOAVar
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
                print(
                    (
                        'Area: %i \tTime: %f\tTrue TDOA %f\t' +
                        'Est TDOA %f\tPhase Error %f\tVDrift %f'
                    )
                    %(
                        traj.detectorArea,
                        arrivalT,
                        constantOffset,
                        estimatedDelay,
                        delayError/myPulsarObject.pulsarPeriod,
                        vDrift
                    )
                )
                #myFilter.realTimePlot()
                # for key in corrSubstateDict:
                #     corrSubstateDict[key].realTimePlot()
            lastT = arrivalT
        successfulRun = True
        # except Exception as inst:
        #     print(type(inst))    # the exception instance
        #     print(inst.args)     # arguments stored in .args
        #     print(inst)
        #     print('Aborting run because of error... restarting.')
        
    traj.f_add_result(
        'correlationSubstate.$',
        correlationSubstate.getStateVector(),
        comment='Correlation vector substate'
    )
    traj.f_add_result(
        'attitudeSubstate.$',
        myAttitude.getStateVector(),
        comment='Attitude substate'
    )

    traj.f_add_result(
        'constantOffset.$',
        constantOffset,
        comment='Offset in seconds'
    )

    traj.f_add_result(
        'finalDelayError.$',
        md.utils.spacegeometry.phaseError(
            correlationSubstate.stateVectorHistory[-1]['signalTDOA'],
            constantOffset,
            myPulsarObject.pulsarPeriod
            ),
        comment='Final delay error estimate'
        )
    traj.f_add_result(
        'finalDelayVar.$',
        correlationSubstate.stateVectorHistory[-1]['TDOAVar'],
        comment='Variance of final TDOA estimate error'
        )
    
    traj.f_add_result(
        'finalEulerError.$',
        md.utils.eulerAngleDiff(
            myAttitude.stateVectorHistory[-1]['eulerAngles'],
            attitude(myAttitude.stateVectorHistory[-1]['t'], returnQ=False)
            ),
        comment='Final euler angle error'
        )
    traj.f_add_result(
        'eulerSTD.$',
        myAttitude.stateVectorHistory[-1]['eulerSTD'],
        comment='standard deviation of euler angle estimate error'
        )
    traj.f_add_result(
        'peakLock.$',
        correlationSubstate.peakLock,
        comment='indicates whether peak lock had been reached at end of run'
        )

    traj.f_add_result(
        'vDrift.$',
        vDrift,
        comment='Velocity integration drift over the course of the run, in km'
        )

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

env = Environment(
    filename='./MCResults/',
    trajectory='MonteCarloTest',
    add_time=True,
    git_repository='../.git',
    git_message='Pypet Monte Carlo Runs',
    file_title='More tests of pypet',
    overwrite_file=True
    )

traj = env.trajectory

# Monte Carlo simulation parameters
traj.f_add_parameter('runtime', 1000, comment='Length of simulation in seconds')


#traj.f_add_parameter('pulsarName', 'J0534+2200', comment='Name of the pulsar to run simulation for')
traj.f_add_parameter('pulsarName', 'J0437-4715', comment='Name of the pulsar to run simulation for')
#traj.f_add_parameter('pulsarName', 'B1957+20', comment='Name of the pulsar to run simulation for')
#traj.f_add_parameter('pulsarName', 'B1937+21', comment='Name of the pulsar to run simulation for')
#traj.f_add_parameter('pulsarName', 'B1821-24', comment='Name of the pulsar to run simulation for')
#traj.f_add_parameter('pulsarName', 'J0030+0451', comment='Name of the pulsar to run simulation for')


traj.f_add_parameter('filterTaps', 9, comment='Dimension of correlation vector')
traj.f_add_parameter('processNoise', 1e-15, comment='Process noise constant added to correlation vector')
traj.f_add_parameter('measurementNoiseScaleFactor', 3.0, comment='Tuning parameter for measurement noise')
traj.f_add_parameter('scaleProcessNoise', True, comment='Boolean sets whether the process noise is scaled by the detector area.')
traj.f_add_parameter('peakLockThreshold', 0.1, comment='How low the TDOA variance estimate must be in order to reach peak lock.  Unitless; it is defined in terms of the filter dT')


# Detector Information
traj.f_add_parameter('detectorArea', np.float64(100.0), comment='Detector area in cm^2')
traj.f_add_parameter('detectorFOV', 1, comment='Detector FOV in degrees (angle of half cone)')
traj.f_add_parameter('AOAVar', np.square(1e-4), comment='Angle of arrival measurement error variance in rad^2')

# Trajectory Information
traj.f_add_parameter('constantPhaseOffset', np.float64(0), comment='Constant phase delay added to photon arrivals')
traj.f_add_parameter('orbitPeriod', 100.0/(2*np.pi), comment='Period of orbit in seconds')
traj.f_add_parameter('orbitAmplitude', 0.0, comment='Amplitude of orbit in km')
traj.f_add_parameter('vVar', np.square(1), comment='Variance of velocity measurement in km^2/s^2')

# Attitude information
traj.f_add_parameter('angularVelocity', [0.0, 0.0, 0.0], comment='Angular velocity of detector in rad/s')
traj.f_add_parameter('omegaVar', np.square(1e-6), comment='Variance of angular velocity measurement in rad^2/s^2')
traj.f_add_parameter('initialAttitudeSigma', np.float64(0.1 * np.pi/180.0), comment='Variance of initial euler angle uncertainty in radians')

traj.f_explore(
    cartesian_product(
        {
            'detectorArea': np.logspace(2, 3),
            'constantPhaseOffset': np.random.uniform(low=-1.0, high=1.0, size=5)
        }
    )
)
# traj.f_explore(
#     cartesian_product(
#         {
#             'filterTaps': [7,9,11],
#             'constantPhaseOffset': np.random.uniform(low=0.0, high=1.0, size=20)
#         }
#     )
# )


env.run(run4DOFSimulation)
#md.plots.montecarloplots.plotNTapsVsError(traj)
md.plots.montecarloplots.plotAreaVsError(traj)
# md.plots.montecarloplots.plotKeyVsError(traj,'detectorArea',logx=True,logy=True)
