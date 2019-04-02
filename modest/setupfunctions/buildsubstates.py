import numpy as np
from .. import substates, signals, utils
from .. modularfilter import ModularFilter


## @fun buildPulsarCorrelationSubstate builds an correlation substate based on imported Traj
def buildPulsarCorrelationSubstate(
        traj,
        pulsarObject,
        mySpacecraft,
        ureg,
):
    tdoaStdDevThreshold = None
    velStdDevThreshold = None
    biasStateProcessNoiseStdDev = None
    artificialBiasMeasStdDev = None
    velocityNoiseScaleFactor = None
    tdoaNoiseScaleFactor = None
    tStart = mySpacecraft.tStart
    myPulsarPeriod = pulsarObject.getPeriod(tStart)
    internalNavFilter = traj.correlationFilter.internalNavFilter.INF_Type.value

    initialVelocityStdDev = (
        traj.correlationFilter.internalNavFilter.initialVelocityStdDev.value *
        ureg(traj.correlationFilter.internalNavFilter.initialVelocityStdDev.unit)
    ).to(ureg.speed_of_light).magnitude
    vInitial = (
        mySpacecraft.dynamics.velocity(mySpacecraft.tStart).dot(pulsarObject.unitVec()) *
        ureg.km/ureg.seconds
    ).to(ureg.speed_of_light).magnitude

    velocityNoiseScaleFactor = traj.correlationFilter.internalNavFilter.velocityNoiseScaleFactor.value


    # Import navigation process noise, and convert it to units of speed of light per time to the appropriate power
    navProcessNoise = (
        traj.correlationFilter.internalNavFilter.defaultNavProcessNoise.value *
        ureg(traj.correlationFilter.internalNavFilter.defaultNavProcessNoise.unit)
    )
    navProcessNoiseUnits = (
        ureg.speed_of_light *
        (ureg.seconds ** (navProcessNoise.dimensionality['[time]']+1))
    )
    navProcessNoise = navProcessNoise.to(navProcessNoiseUnits).magnitude

    if 'initialVelocityStdDev' in traj.correlationFilter.internalNavFilter:
        vStdDev = (
            traj.correlationFilter.internalNavFilter.initialVelocityStdDev.value *
            ureg(traj.correlationFilter.internalNavFilter.initialVelocityStdDev.unit)
        ).to(ureg.speed_of_light).magnitude
        
        vInitial = {
            'value': np.random.normal(
                (mySpacecraft.dynamics.velocity(mySpacecraft.tStart).dot(pulsarObject.unitVec()) *
                 ureg.km/ureg.seconds).to(ureg.speed_of_light).magnitude,
                vStdDev),
            'var': np.square(vStdDev)
        }
    else:
        vInitial=None

    if 'initialAccelerationStdDev' in traj.correlationFilter.internalNavFilter:
        aStdDev = (
            traj.correlationFilter.internalNavFilter.initialAccelerationStdDev.value *
            ureg(traj.correlationFilter.internalNavFilter.initialAccelerationStdDev.unit)
        ).to(ureg.speed_of_light/ureg.second).magnitude
        
        aInitial = {
            'value': np.random.normal(
                (mySpacecraft.dynamics.acceleration(mySpacecraft.tStart).dot(pulsarObject.unitVec()) * ureg.km/ureg.second**2).to(ureg.speed_of_light/ureg.second).magnitude,
                aStdDev),
            'var': np.square(aStdDev)
        }
        
    else:
        aInitial=None
        
    if traj.correlationFilter.internalNavFilter.INF_Type.value == 'external':
        internalNavFilter = ModularFilter()
        if traj.correlationFilter.internalNavFilter.biasState.useBiasState.value:
            navCov = np.eye(3)
            navCov[2,2] = myPulsarPeriod/12
            navX0 = np.array([0.0,0.0,0.0])
            biasStateTimeConstant = traj.correlationFilter.internalNavFilter.biasState.timeConstant.value
            biasStateProcessNoiseStdDev = (
                traj.correlationFilter.internalNavFilter.biasState.processNoiseStdDev.value *
                ureg(traj.correlationFilter.internalNavFilter.biasState.processNoiseStdDev.unit)
            ).to(ureg.speed_of_light).magnitude
        else:
            navCov = np.eye(2)
            navX0 = np.array([0.0,0.0])
            biasStateTimeConstant = None
            biasStateProcessNoiseVar = None

        tdoaStdDevThreshold = (
            traj.correlationFilter.internalNavFilter.tdoaUpdateStdDevThreshold.value *
            ureg(traj.correlationFilter.internalNavFilter.tdoaUpdateStdDevThreshold.unit)
        ).to(ureg.speed_of_light * ureg.seconds).magnitude

        velStdDevThreshold = (
            traj.correlationFilter.internalNavFilter.useVelocityStdDevThreshold.value *
            ureg(traj.correlationFilter.internalNavFilter.useVelocityStdDevThreshold.unit)
        ).to(ureg.speed_of_light).magnitude

        tdoaNoiseScaleFactor = traj.correlationFilter.internalNavFilter.tdoaNoiseScaleFactor.value

        navCov[1,1] = vInitial['var']
        navCov[0,0] = myPulsarPeriod/12

        navX0[1] = vInitial['value']

        if biasStateProcessNoiseStdDev:
            biasStateProcessNoiseVar = np.square(biasStateProcessNoiseStdDev)
        else:
            biasStateProcessNoiseVar = None

        navState = substates.oneDimensionalPositionVelocity.oneDPositionVelocity(
            'oneDPositionVelocity',
            {
                't': tStart,
                'stateVector': navX0,
                'position': 0,
                'biasState': 0,
                'positionStd': np.sqrt(myPulsarPeriod/12),
                'velocity': navX0[1],
                'velocityStd': 1,
                'covariance': navCov,
                'aPriori': True,
                'stateVectorID': -1
            },
            biasState=traj.correlationFilter.internalNavFilter.biasState.useBiasState.value,
            biasStateTimeConstant=biasStateTimeConstant,
            biasStateProcessNoiseVar=biasStateProcessNoiseVar,
            storeLastStateVectors=traj.correlationFilter.storeLastStateVectors.value,
        )

        internalNavFilter.addStates(
            'oneDPositionVelocity',
            navState
        )
        internalNavFilter.addSignalSource(
            'oneDPositionVelocity',
            signals.oneDimensionalObject.oneDObjectMeasurement('oneDPositionVelocity')
        )
        internalNavFilter.addSignalSource(
            '',
            None
        )        

    print("|||||||||||||||||||||||||||||||")
    print("inititalizine INF with type:")
    print(internalNavFilter)
    print(traj.correlationFilter.internalNavFilter.INF_Type.value)
    print("|||||||||||||||||||||||||||||||")

    # Import and initialize values for correlation filter
    processNoise = (
        traj.correlationFilter.processNoise.value
    )  # Unitless??

    nFilterTaps = traj.correlationFilter.filterTaps.value
    measurementNoiseScaleFactor = (
        traj.correlationFilter.measurementNoiseScaleFactor.value
    )
    peakLockThreshold = (
        traj.correlationFilter.peakLockThreshold.value
    )

    if 'peakFitPoints' in traj.correlationFilter:
        peakFitPoints = traj.correlationFilter.peakFitPoints.value
    else:
        peakFitPoints = 1
    
    if 'peakEstimator' in traj.correlationFilter:
        peakEstimator = traj.correlationFilter.peakEstimator.value
    else:
        peakEstimator = 'EK'
    
    correlationSubstate = substates.CorrelationVector(
        pulsarObject,
        nFilterTaps,
        myPulsarPeriod/(nFilterTaps+1),
        signalTDOA=0,
        TDOAVar=myPulsarPeriod/12,
        measurementNoiseScaleFactor=measurementNoiseScaleFactor,
        processNoise=processNoise,
        centerPeak=True,
        peakLockThreshold=peakLockThreshold,
        t=mySpacecraft.tStart,
        internalNavFilter=internalNavFilter,
        tdoaStdDevThreshold=tdoaStdDevThreshold,
        velStdDevThreshold=velStdDevThreshold,
        tdoaNoiseScaleFactor=tdoaNoiseScaleFactor,
        velocityNoiseScaleFactor=velocityNoiseScaleFactor,
        storeLastStateVectors=traj.correlationFilter.storeLastStateVectors.value,
        peakFitPoints=peakFitPoints,
        navProcessNoise=np.square(navProcessNoise),
        vInitial=vInitial,
        aInitial=aInitial,
        peakEstimator=peakEstimator
    )
    
    return correlationSubstate, vInitial['value']


## @fun buildPulsarCorrelationSubstate builds an correlation substate based on imported Traj
def buildAttitudeSubstate(
        traj,
        mySpacecraft,
        ureg,
):
    gyroBiasStdDev = (
        traj.dynamicsModel.gyroBiasStdDev.value *
        ureg(traj.dynamicsModel.gyroBiasStdDev.unit)
    ).to(ureg.rad/ureg.s).magnitude


    initialAttitudeStdDevRoll = (
        traj.dynamicsModel.initialAttitudeStdDev.roll.value *
        ureg(traj.dynamicsModel.initialAttitudeStdDev.roll.unit)
    ).to(ureg.rad).magnitude
    initialAttitudeStdDevRA = (
        traj.dynamicsModel.initialAttitudeStdDev.RA.value *
        ureg(traj.dynamicsModel.initialAttitudeStdDev.RA.unit)
    ).to(ureg.rad).magnitude
    initialAttitudeStdDevDEC = (
        traj.dynamicsModel.initialAttitudeStdDev.DEC.value *
        ureg(traj.dynamicsModel.initialAttitudeStdDev.DEC.unit)
    ).to(ureg.rad).magnitude

    initialAttitudeStdDev_DEG = (
        np.max([initialAttitudeStdDevDEC, initialAttitudeStdDevRA]) *
        ureg.rad
    ).to(ureg.deg).magnitude

    initialAttitudeEstimate = utils.euler2quaternion(
        mySpacecraft.dynamics.attitude(mySpacecraft.tStart, returnQ=False)
        +
        np.array(
            [
                np.random.normal(0, scale=initialAttitudeStdDevRoll),
                np.random.normal(0, scale=initialAttitudeStdDevRA),
                np.random.normal(0, scale=initialAttitudeStdDevDEC)
            ]
        ) 
    )

    attitudeCovariance = np.eye(3)
    attitudeCovariance[0, 0] = np.square(initialAttitudeStdDevRoll)
    attitudeCovariance[1, 1] = np.square(initialAttitudeStdDevDEC)
    attitudeCovariance[2, 2] = np.square(initialAttitudeStdDevRA)
    print(attitudeCovariance)

    if traj.attitudeFilter.updateMeasMat.value == 'unitVec':
        useUnitVec=True
    else:
        useUnitVec=False

    myAttitude = substates.Attitude(
        t=mySpacecraft.tStart,
        attitudeQuaternion=initialAttitudeEstimate,
        attitudeErrorCovariance=attitudeCovariance,
        gyroBiasCovariance=np.eye(3)*np.square(gyroBiasStdDev),
        useUnitVector=useUnitVec,
        storeLastStateVectors=traj.attitudeFilter.storeLastStateVectors.value        
    )

    return myAttitude
