import numpy as np
import matplotlib.pyplot as plt, mpld3
import pickle
from .. import utils

def outputPlots(
        useINF,
        resultsDict,
        saveOutput=True,
        outputDir=None,
        figureDict=None
):
    print()
    print("||=================================================||")    
    print("Plotting current results and saving output")
    print("||=================================================||")    
    print()

    if figureDict is None:
        attFig = plt.figure(2, figsize=(16,9))
    else:
        attFig = figureDict['attFigure']
    attFig.clear()
    plt.figure(2)
    
    estimatedT = resultsDict['estimatedT']['value']
    rollSigma = resultsDict['estimatedAttitudeSigma_DEG']['value'][0]
    pitchSigma = resultsDict['estimatedAttitudeSigma_DEG']['value'][1]
    yawSigma = resultsDict['estimatedAttitudeSigma_DEG']['value'][2]
    
    rollError = resultsDict['attitudeError_DEG']['value'][0]
    pitchError = resultsDict['attitudeError_DEG']['value'][1]
    yawError = resultsDict['attitudeError_DEG']['value'][2]

    if 'attitudeError_DEG_PO' in resultsDict:
        attPO = True
        rollError_PO = resultsDict['attitudeError_DEG_PO']['value'][0]
        pitchError_PO = resultsDict['attitudeError_DEG_PO']['value'][1]
        yawError_PO = resultsDict['attitudeError_DEG_PO']['value'][2]
    else:
        attPO = False
        
    estimatedPos = resultsDict['estimatedPos']['value']
    estimatedPosStdDev = resultsDict['estimatedPosStdDev']['value']
    estimatedPosStdDev_calc = resultsDict['estimatedPosStdDev_calc']['value']

    if 'navVel' in resultsDict:
        navVel = resultsDict['navVel']['value']
        navVelStd = resultsDict['navVelStd']['value']
    else:
        navVel = None
    if 'navAcc' in resultsDict:
        navAcc = resultsDict['navAcc']['value']
        navAccStd = resultsDict['navAccStd']['value']
    else:
        navAcc=None


    truePos = resultsDict['truePos']['value']
    trueVel = resultsDict['trueVel']['value']
    trueAcc = resultsDict['trueAcc']['value']

    velocityOnlyRange = resultsDict['velocityOnlyRange']['value']
                                     
    
    stdDevColor = [0.5, 0.5, 0.5]
    plt.subplot2grid((3,1), (0,0))
    plt.title('Roll error, standard dev')
    plt.plot(
        estimatedT - estimatedT[0],
        rollError
    )
    if attPO:
        plt.plot(
            estimatedT - estimatedT[0],
            rollError_PO
        )
    plt.plot(
        estimatedT-estimatedT[0],
        -rollSigma,
        color=stdDevColor
    )
    plt.plot(
        estimatedT-estimatedT[0],
        rollSigma,
        color=stdDevColor
    )
    plt.ylabel('deg')

    plt.subplot2grid((3,1), (1,0))
    plt.title('Pitch error, standard dev')

    plt.plot(
        estimatedT-estimatedT[0],
        pitchError
    )
    if attPO:
        plt.plot(
            estimatedT - estimatedT[0],
            pitchError_PO
        )
    
    plt.plot(
        estimatedT-estimatedT[0],
        pitchSigma,
        color=stdDevColor
    )
    plt.plot(
        estimatedT-estimatedT[0],
        -pitchSigma,
        color=stdDevColor
    )
    plt.ylabel('deg')

    plt.subplot2grid((3,1), (2,0))
    plt.title('Yaw error, standard dev')
    plt.plot(
        estimatedT-estimatedT[0],
        yawError
    )
    if attPO:
        plt.plot(
            estimatedT - estimatedT[0],
            yawError_PO
        )
    
    plt.plot(
        estimatedT-estimatedT[0],
        yawSigma,
        color=stdDevColor
    )
    plt.plot(
        estimatedT-estimatedT[0],
        -yawSigma,
        color=stdDevColor
    )
    plt.ylabel('deg')
    plt.xlabel('time (s)')
    plt.subplots_adjust(hspace=.5)

    if saveOutput:
        mpld3.save_html(attFig, outputDir + '/attitude.html')
    plt.show(block=False)
    
    if figureDict is None:
        tdoaFigure = plt.figure(3, figsize=(16,9))
    else:
        tdoaFigure = figureDict['tdoaFigure']
    tdoaFigure.clear()
    plt.figure(3)
    
    plt.plot(
        estimatedT,
        truePos - estimatedPos,
        label=(
            'unfiltered delay error ($\sigma=%s$)'
            %estimatedPosStdDev_calc
            )
    )
    plt.plot(estimatedT, estimatedPosStdDev, ls='-.', color=[0.5,0.5,0.5], label = 'unfiltered standard deviation')
    plt.plot(estimatedT, -estimatedPosStdDev, ls='-.', color=[0.5,0.5,0.5])


    # if useINF:
        # plt.plot(
        #     estimatedT,
        #     truePos - navPos,
        #     label=(
        #         'nav filter delay error, ($\sigma = %s$)'
        #         %navPosErrorStdDev
        #     )
        # )
        # plt.plot(
        #     estimatedT,
        #     navBiasState,
        #     label='bias state'
        # )
        # plt.plot(estimatedT, navPosStd, color=[0.9,0.9,0.9], label='nav filter standard deviation')
        # plt.plot(estimatedT, -navPosStd, color=[0.9,0.9,0.9])

    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('(km)')
    plt.plot(
        estimatedT,
        truePos -
        (velocityOnlyRange)
        - truePos[0],
        label='initial velocity error propagation'
    )
    if saveOutput:
        mpld3.save_html(tdoaFigure, outputDir + '/tdoa.html')
        # plt.close(tdoaFigure)
    plt.show(block=False)
    
    figureDict = {
        'tdoaFigure': tdoaFigure,
        'attFigure': attFig
    }
    
    velocityFigure = None
    if not np.any(navVel==None):
        if figureDict is None or 'velocityFigure' not in figureDict:
            velocityFigure = plt.figure(4, figsize=(16,9))
        else:
            velocityFigure = figureDict['velocityFigure']
        velocityFigure.clear()
        plt.figure(4)
        
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
            # plt.close(velocityFigure)
        # else:
        plt.show(block=False)
        figureDict['velocityFigure'] = velocityFigure
        
    if not np.any(navAcc==None):
        if figureDict is None or 'accelerationFigure' not in figureDict:
            accelerationFigure = plt.figure(5, figsize=(16,9))
        else:
            accelerationFigure = figureDict['accelerationFigure']
        accelerationFigure.clear()
        plt.figure(5)
        
        plt.plot(
            estimatedT,
            trueAcc - navAcc,
            label=(
                'acceleration error ($\sigma = %s$)'
                %np.std(trueAcc - navAcc)
            )
        )
        plt.plot(estimatedT, navAccStd, color=[0.5,0.5,0.5], label='acceleration std dev')
        plt.plot(estimatedT, -navAccStd, color=[0.5,0.5,0.5])
        plt.legend()

        if saveOutput:
            mpld3.save_html(accelerationFigure, outputDir + '/acceleration.html')
            # plt.close(velocityFigure)
        # else:
        plt.show(block=False)
        figureDict['accelerationFigure'] = accelerationFigure
        
        # figureDict = {
        #     'tdoaFigure': tdoaFigure,
        #     'velocityFigure': velocityFigure,
        #     'attFigure': attFig,
        #     'accelerationFigure': accelerationFigure
        # }

    return(figureDict)


def createResultsDict(
        mySpacecraft,
        ureg,
        estimatedT,
        tdoa,
        attitude,
        velocityOnlyRangeTruncated,
        attitudePO=None,
        useINF=False,
        saveOutput=True,
        outputDir=None
):
    rad2deg = 180/np.pi
    estimatedRoll = np.array(attitude['roll'])
    estimatedPitch = np.array(attitude['pitch'])
    estimatedYaw = np.array(attitude['yaw'])
    rollSigma = np.array(attitude['rollSigma'])
    pitchSigma = np.array(attitude['pitchSigma'])
    yawSigma = np.array(attitude['yawSigma'])

    trueAtt = np.array(mySpacecraft.dynamics.attitude(
        estimatedT+mySpacecraft.tStart, returnQ=False)
    )

    trueAtt_DEG = trueAtt * rad2deg

    estimateAttitude_DEG = [
        estimatedRoll*rad2deg,
        estimatedPitch*rad2deg,
        estimatedYaw*rad2deg
    ]

    
    estimatedAttitudeSigma_DEG = [
        rollSigma*rad2deg,
        pitchSigma*rad2deg,
        yawSigma*rad2deg
    ]

    rollError_DEG = np.array(utils.eulerAngleDiff(estimatedRoll, trueAtt[:,0])) * rad2deg
    pitchError_DEG = np.array(utils.eulerAngleDiff(estimatedPitch, trueAtt[:,1])) * rad2deg
    yawError_DEG = np.array(utils.eulerAngleDiff(estimatedYaw, trueAtt[:,2])) * rad2deg

    attitudeError_DEG = [rollError_DEG, pitchError_DEG, yawError_DEG]

    if attitudePO is not None:
        roll_PO = np.array(attitudePO['roll'])
        pitch_PO = np.array(attitudePO['pitch'])
        yaw_PO = np.array(attitudePO['yaw'])
        
        estimateAttitude_DEG_PO = [
            roll_PO*rad2deg,
            pitch_PO*rad2deg,
            yaw_PO*rad2deg
        ]

        rollError_DEG_PO = np.array(utils.eulerAngleDiff(roll_PO, trueAtt[:,0])) * rad2deg
        pitchError_DEG_PO = np.array(utils.eulerAngleDiff(pitch_PO, trueAtt[:,1])) * rad2deg
        yawError_DEG_PO = np.array(utils.eulerAngleDiff(yaw_PO, trueAtt[:,2])) * rad2deg

        attitudeError_DEG_PO = [rollError_DEG_PO, pitchError_DEG_PO, yawError_DEG_PO]
    else:
        attitudeError_DEG_PO = None
        
    estimatedTDOA = np.array(tdoa['TDOA'])
    estimatedTDOAStd = np.array(tdoa['TDOAStd'])

    trueTDOA = np.array([
        mySpacecraft.dynamics.position(t + mySpacecraft.tStart).dot(tdoa['unitVec']) for t in estimatedT
    ])
    trueVel = np.array([
        mySpacecraft.dynamics.velocity(t + mySpacecraft.tStart).dot(tdoa['unitVec']) for t in estimatedT
    ])
    
    trueAcc = np.array([
        mySpacecraft.dynamics.acceleration(t + mySpacecraft.tStart).dot(tdoa['unitVec']) for t in estimatedT
    ])

    truePos = trueTDOA - trueTDOA[0]
    

    if len(tdoa['vel'])>0:
        navVel = np.array(tdoa['vel'])
        navVelStd = np.array(tdoa['velStd'])
        navVelErrorStdDev = np.std(navVel - trueVel)
        
    if len(tdoa['acc'])>0:
        navAcc = np.array(tdoa['acc'])
        navAccStd = np.array(tdoa['accStd'])

    estimatedPos = (estimatedTDOA * ureg.seconds * ureg.speed_of_light).to(ureg('km')).magnitude
    if not np.any(tdoa['peakLock']):
        meanDiff = np.mean(estimatedPos - truePos)
    else:
        meanDiff = np.mean(
            [eP-tP for tP, eP, pL in zip(truePos, estimatedPos, tdoa['peakLock']) if pL]
        )
        
    estimatedPos = estimatedPos - meanDiff
    estimatedPosStdDev = (
        estimatedTDOAStd * ureg.seconds * ureg.speed_of_light
    ).to(ureg.km).magnitude


    estimatedPosStdDev_calc = np.std(
        [tP - eP for tP, eP, pL in zip(truePos, estimatedPos, tdoa['peakLock']) if pL]
    )
    
    
    resultsDict = {}

    if len(navVel) > 0:
        resultsDict['navVel'] = {
            'value': navVel,
            'comment': 'Spacecraft velocity as estimated by internal nav filter',
            'unit': 'km/s'
        }

        resultsDict['navVelStd'] = {
            'value': navVelStd,
            'comment': 'Spacecraft velocity standard deviation estimated by internal nav filter',
            'unit': 'km/s'
        }

        resultsDict['navVelErrorStdDev'] = {
            'value': navVelErrorStdDev,
            'comment':'Standard deviation of spacecraft velocity estimate error',
            'unit':'km/s'
        }
        
    if len(navAcc) > 0:
        resultsDict['navAcc'] = {
            'value': navAcc,
            'comment': 'Spacecraft acceleration as estimated by internal nav filter',
            'unit': 'km/s^2'
        }

        resultsDict['navAccStd'] = {
            'value': navAccStd,
            'comment': 'Spacecraft acceleration standard deviation estimated by internal nav filter',
            'unit': 'km/s^2'
        }
    resultsDict['velocityOnlyRange'] = {
        'value': velocityOnlyRangeTruncated,
        'comment':'Range from velocity propagation',
        'unit':'km'
    }



    resultsDict['truePos'] = {
        'value': truePos,
        'comment': 'True Spacecraft range',
        'unit': 'km'
    }
    resultsDict['trueVel'] = {
        'value': trueVel,
        'comment': 'True Spacecraft velocity',
        'unit': 'km/s'
    }
    resultsDict['trueAcc'] = {
        'value': trueAcc,
        'comment': 'True Spacecraft acceleration',
        'unit': 'km/s^2'
    }
    resultsDict['estimatedPos'] = {
        'value': estimatedPos,
        'comment': 'Estimated spacecraft range (unfiltered)',
        'unit': 'km'
    }
    resultsDict['estimatedPosStdDev'] = {
        'value': estimatedPosStdDev,
        'comment': 'Standard deviation of estimated spacecraft range (unfiltered)',
        'unit': 'km'
    }
    resultsDict['estimatedPosStdDev_calc'] = {
        'value': estimatedPosStdDev_calc,
        'comment': 'Standard deviation of estimated range (true)',
        'unit': 'km'
    }

    resultsDict['trueAtt_DEG'] = {
        'value': trueAtt_DEG,
        'comment': 'True attitude solution',
        'unit': 'degrees'
    }
    resultsDict['estimatedAttitude_DEG'] = {
        'value': estimateAttitude_DEG,
        'comment': 'Attitude estimate',
        'unit': 'degrees'
    }
    resultsDict['attitudeError_DEG'] = {
        'value': attitudeError_DEG,
        'comment': 'Attitude estimate error',
        'unit': 'degrees'
    }

    if attitudeError_DEG_PO is not None:
        resultsDict['attitudeError_DEG_PO'] = {
            'value': attitudeError_DEG_PO,
            'comment': 'Attitude estimate error from propagation only',
            'unit': 'degrees'
        }
    
    resultsDict['estimatedAttitudeSigma_DEG'] = {
        'value': estimatedAttitudeSigma_DEG,
        'comment': 'Attitude estimate standard deviation',
        'unit': 'degrees'
    }

    resultsDict['peakLock'] = {
        'value': tdoa['peakLock'],
        'comment': 'Indication of peak lock',
        'unit': ''
    }
    resultsDict['estimatedT'] = {
        'value': estimatedT,
        'comment': 'Time',
        'unit': 's'
    }
    
    if saveOutput:
        pickle.dump( resultsDict, open( outputDir + "/data.p", "wb" ) )

    return(resultsDict)
