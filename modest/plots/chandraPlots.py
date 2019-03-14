import numpy as np
import matplotlib.pyplot as plt, mpld3
import pickle
from .. import utils

def outputPlots(
        traj,
        ureg,
        estimatedT,
        estimatedTDOA,
        estimatedTDOAVar,
        navTDOA,
        navTDOAStd,
        navVel,
        navVelStd,
        navBiasState,
        trueTDOA,
        trueVel,
        velocityOnlyRangeTruncated,
        estimatedRoll,
        estimatedPitch,
        estimatedYaw,
        rollSigma,
        pitchSigma,
        yawSigma,
        mySpacecraft,
        saveOutput=True,
        outputDir=None,
):
    rad2deg = 180/np.pi
    estimatedRoll = np.array(estimatedRoll)
    estimatedPitch = np.array(estimatedPitch)
    estimatedYaw = np.array(estimatedYaw)
    rollSigma = np.array(rollSigma)
    pitchSigma = np.array(pitchSigma)
    yawSigma = np.array(yawSigma)

    trueAtt = np.array(mySpacecraft.dynamics.attitude(estimatedT+mySpacecraft.tStart,returnQ=False))

    trueAtt_DEG = trueAtt * rad2deg

    estimateAttitude_DEG = [estimatedRoll*rad2deg, estimatedPitch*rad2deg, estimatedYaw*rad2deg]
    estimatedAttitudeSigma_DEG = [rollSigma*rad2deg, pitchSigma*rad2deg, yawSigma*rad2deg]

    rollError_DEG = np.array(utils.eulerAngleDiff(estimatedRoll, trueAtt[:,0])) * rad2deg
    pitchError_DEG = np.array(utils.eulerAngleDiff(estimatedPitch, trueAtt[:,1])) * rad2deg
    yawError_DEG = np.array(utils.eulerAngleDiff(estimatedYaw, trueAtt[:,2])) * rad2deg

    attitudeError_DEG = [rollError_DEG, pitchError_DEG, yawError_DEG]



    attFig = plt.figure(figsize=(16,9))
    stdDevColor = [0.5, 0.5, 0.5]
    plt.subplot2grid((3,1), (0,0))
    plt.title('Roll error, standard dev')
    plt.plot(
        estimatedT-estimatedT[0],
        rollError_DEG
    )
    plt.plot(
        estimatedT-estimatedT[0],
        -rollSigma*rad2deg, color=stdDevColor
    )
    plt.plot(
        estimatedT-estimatedT[0],
        rollSigma*rad2deg, color=stdDevColor
    )
    plt.ylabel('deg')

    plt.subplot2grid((3,1), (1,0))
    plt.title('Pitch error, standard dev')

    plt.plot(
        estimatedT-estimatedT[0],
        pitchError_DEG
    )
    plt.plot(
        estimatedT-estimatedT[0],
        pitchSigma*rad2deg,
        color=stdDevColor
    )
    plt.plot(
        estimatedT-estimatedT[0],
        -pitchSigma*rad2deg, color=stdDevColor
    )
    plt.ylabel('deg')

    plt.subplot2grid((3,1), (2,0))
    plt.title('Yaw error, standard dev')
    plt.plot(
        estimatedT-estimatedT[0],
        yawError_DEG
    )
    plt.plot(
        estimatedT-estimatedT[0],
        yawSigma*rad2deg, color=stdDevColor
    )
    plt.plot(
        estimatedT-estimatedT[0],
        -yawSigma*rad2deg, color=stdDevColor
    )
    plt.ylabel('deg')
    plt.xlabel('time (s)')
    plt.subplots_adjust(hspace=.5)

    if saveOutput:
        mpld3.save_html(attFig, outputDir + '/attitude.html')

    estimatedTDOA = np.array(estimatedTDOA)
    estimatedTDOAVar = np.array(estimatedTDOAVar)

    if traj.internalNavFilter.useINF.value:
        navTDOA = np.array(navTDOA)
        navVel = np.array(navVel)
        navVelStd = np.array(navVelStd)

        navBiasState = np.array(navBiasState)
        navTDOAStd = np.array(navTDOAStd)

    trueTDOA = np.array(trueTDOA)
    trueVel = np.array(trueVel)

    truePos = trueTDOA - trueTDOA[0]
    estimatedPos = (estimatedTDOA * ureg.seconds * ureg.speed_of_light).to(ureg('km')).magnitude
    meanDiff = np.mean(estimatedPos - truePos)
    estimatedPos = estimatedPos - meanDiff
    estimatedPosStdDev = (
        np.sqrt(estimatedTDOAVar) * ureg.seconds * ureg.speed_of_light
    ).to(ureg.km).magnitude

    if traj.internalNavFilter.useINF.value:
        navPos = (navTDOA * ureg.seconds * ureg.speed_of_light).to(ureg('km')).magnitude
        meanNavDiff = np.mean(navPos - truePos)
        navPos = navPos - meanNavDiff

        navPosStd = (navTDOAStd * ureg.speed_of_light * ureg.seconds).to(ureg('km')).magnitude
        navVel = (navVel * ureg.speed_of_light).to(ureg('km/s')).magnitude
        navVelStd = (navVelStd * ureg.speed_of_light).to(ureg('km/s')).magnitude

        navBiasState = (navBiasState * ureg.seconds * ureg.speed_of_light).to(ureg('km')).magnitude

        navPosErrorStdDev = np.std(navPos - truePos)
        navVelErrorStdDev = np.std(navVel - trueVel)


    tdoaFigure = plt.figure(figsize=(16,9))
    plt.plot(
        estimatedT,
        truePos - estimatedPos,
        label=(
            'unfiltered delay error ($\sigma=%s$)'
            %np.std(truePos - estimatedPos)
            )
    )
    plt.plot(estimatedT, estimatedPosStdDev, ls='-.', color=[0.5,0.5,0.5], label = 'unfiltered standard deviation')
    plt.plot(estimatedT, -estimatedPosStdDev, ls='-.', color=[0.5,0.5,0.5])


    if traj.internalNavFilter.useINF.value:
        plt.plot(
            estimatedT,
            truePos - navPos,
            label=(
                'nav filter delay error, ($\sigma = %s$)'
                %np.std(truePos - navPos)
            )
        )
        plt.plot(
            estimatedT,
            navBiasState,
            label='bias state'
        )
        plt.plot(estimatedT, navPosStd, color=[0.9,0.9,0.9], label='nav filter standard deviation')
        plt.plot(estimatedT, -navPosStd, color=[0.9,0.9,0.9])

    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('(km)')
    plt.plot(
        estimatedT,
        trueTDOA -
        (velocityOnlyRangeTruncated)
        - trueTDOA[0],
        label='initial velocity error propagation'
    )
    if saveOutput:
        mpld3.save_html(tdoaFigure, outputDir + '/tdoa.html')

    if traj.internalNavFilter.useINF.value:
        velocityFigure = plt.figure(figsize=(16,9))
        plt.plot(
            estimatedT,
            trueVel - navVel,
            label=(
                'velocity error ($\sigma = %s$)'
                %np.std(trueVel - navVel)
            )
        )
        plt.plot(estimatedT, navVelStd, color=[0.5,0.5,0.5], label='velocity std dev')
        plt.plot(estimatedT, -navVelStd, color=[0.5,0.5,0.5])
        plt.legend()

        if saveOutput:
            mpld3.save_html(velocityFigure, outputDir + '/velocity.html')


    resultsDict = {}

    if traj.internalNavFilter.useINF.value:
        resultsDict['navFilterBiasState'] = {
            'value': navBiasState,
            'comment': 'Navigation filter measurement bias state estimate',
            'unit': 'km'
        }
        resultsDict['navFilterRange'] = {
            'value': navPos,
            'comment': 'Spacecraft range estimated by internal nav filter',
            'unit': 'km'
        }

        resultsDict['navFilterRangeStdDev'] = {
            'value': navPosStd,
            'comment': 'Spacecraft range standard deviation estimated by internal nav filter',
            'unit': 'km'
    }

        resultsDict['navFilterRangeErrorStdDev'] = {
            'value': navPosErrorStdDev,
            'comment': 'Standard deviation of spacecraft range estimate error',
            'unit': 'km'
        }


        resultsDict['navFilterVelocity'] = {
            'value': navVel,
            'comment': 'Spacecraft range as estimated by internal nav filter',
            'unit': 'km/s'
        }

        resultsDict['navFilterVelocityStdDev'] = {
            'value': navVelStd,
            'comment': 'Spacecraft velocity standard deviation estimated by internal nav filter',
            'unit': 'km/s'
        }

        resultsDict['navFilterVelocityErrorStdDev'] = {
            'value': navVelErrorStdDev,
            'comment':'Standard deviation of spacecraft velocity estimate error',
            'unit':'km/s'
        }



    resultsDict['trueRange'] = {
        'value': truePos,
        'comment': 'True Spacecraft range',
        'unit': 'km'
    }
    resultsDict['estimatedRangeUnfiltered'] = {
        'value': estimatedPos,
        'comment': 'Estimated spacecraft range (unfiltered)',
        'unit': 'km'
    }
    resultsDict['estimatedRangeUnfiltered'] = {
        'value': estimatedPosStdDev,
        'comment': 'Standard deviation of estimated spacecraft range (unfiltered)',
        'unit': 'km'
    }

    resultsDict['trueAttitude'] = {
        'value': trueAtt_DEG,
        'comment': 'True attitude solution',
        'unit': 'degrees'
    }
    resultsDict['estimatedAttitude'] = {
        'value': estimateAttitude_DEG,
        'comment': 'Attitude estimate',
        'unit': 'degrees'
    }
    resultsDict['attitudeEstimateError'] = {
        'value': attitudeError_DEG,
        'comment': 'Attitude estimate error',
        'unit': 'degrees'
    }

    resultsDict['attitudeEstimateStdDev'] = {
        'value': estimatedAttitudeSigma_DEG,
        'comment': 'Attitude estimate standard deviation',
        'unit': 'degrees'
    }

    if saveOutput:
        pickle.dump( resultsDict, open( outputDir + "/data.p", "wb" ) )
