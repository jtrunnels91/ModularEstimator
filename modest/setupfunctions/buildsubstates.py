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
    internalNavFilter = None

    initialVelocityStdDev = (
        traj.internalNavFilter.initialVelocityStdDev.value *
        ureg(traj.internalNavFilter.initialVelocityStdDev.unit)
    ).to(ureg.speed_of_light).magnitude
    vInitial = (
        mySpacecraft.dynamics.velocity(mySpacecraft.tStart).dot(pulsarObject.unitVec()) *
        ureg.km/ureg.seconds
    ).to(ureg.speed_of_light).magnitude
    vInitial_C = np.random.normal(vInitial, initialVelocityStdDev)

    velocityNoiseScaleFactor = traj.internalNavFilter.velocityNoiseScaleFactor.value
    
    if traj.internalNavFilter.useINF.value:
        internalNavFilter = ModularFilter()
        # internalNavFilter = None
        artificialBiasMeasStdDev = None
        if traj.internalNavFilter.useBiasState.value:
            navCov = np.eye(3)
            navCov[2,2] = myPulsarPeriod/12
            # navCov[2,2] = myPulsarObject.getPeriod(tStart)/12
            navX0 = np.array([0.0,0.0,0.0])
            biasStateTimeConstant = traj.internalNavFilter.biasStateTimeConstant.value
            biasStateProcessNoiseStdDev = (
                traj.internalNavFilter.biasStateProcessNoiseStdDev.value *
                ureg(traj.internalNavFilter.biasStateProcessNoiseStdDev.unit)
            ).to(ureg.speed_of_light).magnitude
            if traj.internalNavFilter.artificialBiasMeas:
                artificialBiasMeasStdDev = (
                    traj.internalNavFilter.artificialBiasMeas.noiseStdDev.value *
                    ureg(traj.internalNavFilter.artificialBiasMeas.noiseStdDev.unit)
                ).to(ureg.speed_of_light * ureg.seconds).magnitude
        else:
            navCov = np.eye(2)
            navX0 = np.array([0.0,0.0])
            biasStateTimeConstant = None
            biasStateProcessNoiseVar = None

        tdoaStdDevThreshold = (
            traj.internalNavFilter.tdoaUpdateStdDevThreshold.value *
            ureg(traj.internalNavFilter.tdoaUpdateStdDevThreshold.unit)
        ).to(ureg.speed_of_light * ureg.seconds).magnitude

        velStdDevThreshold = (
            traj.internalNavFilter.useVelocityStdDevThreshold.value *
            ureg(traj.internalNavFilter.useVelocityStdDevThreshold.unit)
        ).to(ureg.speed_of_light).magnitude

        tdoaNoiseScaleFactor = traj.internalNavFilter.tdoaNoiseScaleFactor.value

        navCov[1,1] = np.square(initialVelocityStdDev)
        # navCov[0,0] = myPulsarObject.getPeriod(tStart)/12
        navCov[0,0] = myPulsarPeriod/12

        navX0[1] = vInitial_C
        if artificialBiasMeasStdDev:
            biasMeasVar = np.square(artificialBiasMeasStdDev)
        else:
            biasMeasVar = None

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
            biasState=traj.internalNavFilter.useBiasState.value,
            artificialBiasMeas=traj.internalNavFilter.artificialBiasMeas.value,
            biasStateTimeConstant=biasStateTimeConstant,
            biasMeasVar=biasMeasVar,
            biasStateProcessNoiseVar=biasStateProcessNoiseVar
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


    defaultOneDAccelerationStdDev = (
        traj.internalNavFilter.defaultAccelerationStdDev.value *
        ureg(traj.internalNavFilter.defaultAccelerationStdDev.unit)
    ).to(ureg.speed_of_light).magnitude


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

    # Now initialize the correlation substate
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
        defaultOneDAccelerationVar=np.square(defaultOneDAccelerationStdDev),
        tdoaStdDevThreshold=tdoaStdDevThreshold,
        velStdDevThreshold=velStdDevThreshold,
        tdoaNoiseScaleFactor=tdoaNoiseScaleFactor,
        velocityNoiseScaleFactor=velocityNoiseScaleFactor
    )

    return correlationSubstate, vInitial_C


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
        useUnitVector=useUnitVec
    )

    return myAttitude
