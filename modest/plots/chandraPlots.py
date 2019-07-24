import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt, mpld3
import matplotlib.ticker as ticker

import pickle
from .. import utils

umnPrimaryColors = [[122 / 225, 0 / 225, 25 / 225],
                    [255 / 255, 204 / 225, 51 / 255]]

umnSecondaryDark = [[0 / 255, 61 / 255, 76 / 255],
                    [202 / 255, 119 / 255, 0 / 255],
                    [114 / 255, 110 / 255, 32 / 255],
                    [97 / 255, 99 / 255, 101 / 255],
                    [145 / 255, 120 / 255, 91 / 255],
                    [91 / 255, 0 / 255, 19 / 255],
                    [255 / 255, 183 / 255, 30 / 255]]


umnSecondaryLight = [[0 / 255, 185 / 255, 228 / 255],
                     [233 / 255, 131 / 255, 0 / 255],
                     [190 / 255, 214 / 255, 0 / 255],
                     [204 / 255, 204 / 255, 204 / 255],
                     [226 / 255, 211 / 255, 164 / 255],
                     [144 / 255, 0 / 255, 33 / 255],
                     [255 / 255, 222 / 255, 122 / 255]]


def outputPlots(
        useINF,
        resultsDict,
        saveOutput=True,
        outputDir=None,
        axisDict=None,
        plotPropagationError=False,
        scaleByStdDev=None,
        lineWeight=2,
        legendFont=14,
        legendLineLength=10,
        legendBorderPad=2,
        outputFormat='HTML',
        clearOldPlot=True,
        placeLegend=False,
        logErrorPlot=False,
        colorCounter=0,
        colorList=None
):
    if not colorList:
        colorList = umnSecondaryDark
        
    print()
    print("||=================================================||")    
    print("Plotting current results and saving output")
    print("||=================================================||")    
    print()
    legendDict = {}

    if axisDict == None:
        axisDict = {}

    if saveOutput and outputFormat == 'SVG':
        mp.rcParams['svg.fonttype'] = 'none'
        mp.rcParams['axes.unicode_minus'] = False
        plt.rc('text', usetex=False)

    if axisDict is None or 'attAxis' not in axisDict:
        attitudeFigure=plt.figure(figsize=(16,9))
        print("generating new attitude figure")
        if placeLegend:
            rollAxis = plt.subplot2grid((3,4), (0,0),colspan=3)
            pitchAxis = plt.subplot2grid((3,4), (1,0),colspan=3)
            yawAxis = plt.subplot2grid((3,4), (2,0),colspan=3)
        else:
            rollAxis = plt.subplot2grid((3,1), (0,0))
            pitchAxis = plt.subplot2grid((3,1), (1,0))
            yawAxis = plt.subplot2grid((3,1), (2,0))
    else:
        rollAxis = axisDict['attAxis']['roll']
        pitchAxis = axisDict['attAxis']['pitch']
        yawAxis = axisDict['attAxis']['yaw']
        
        # plt.sca(attAxis)

    if clearOldPlot:
        rollAxis.clear()
        pitchAxis.clear()
        yawAxis.clear()
        
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
    legendLabelList = []
    legendLineList = []
        
    # rollAxis.set_title(r'Roll error, \$\pm 1 \sigma\$ bounds')
    estLine, = rollAxis.plot(
        estimatedT - estimatedT[0],
        rollError,
        color=colorList[0 + colorCounter],
        lw=lineWeight
    )
    legendLineList.append(estLine)
    legendLabelList.append('attitude estimate error')
    if attPO and plotPropagationError:
        propLine,=rollAxis.plot(
            estimatedT - estimatedT[0],
            rollError_PO,
            color=colorList[1 + colorCounter],
            ls='dashdot',
            lw=lineWeight
        )
        legendLineList.append(propLine)
        legendLabelList.append('inertial propagation error')
        
    sigmaLine,=rollAxis.plot(
        estimatedT-estimatedT[0],
        -rollSigma,
        color=stdDevColor,
        ls='dotted',
        lw=lineWeight
    )
    legendLineList.append(sigmaLine)
    legendLabelList.append(r'\$\pm 1 \sigma\$')
        
    
    rollAxis.plot(
        estimatedT-estimatedT[0],
        rollSigma,
        color=stdDevColor,
        ls='dotted',
        lw=lineWeight
    )
    rollAxis.set_ylabel(r'roll error \$(deg)\$')
    rollAxis.grid()

    if scaleByStdDev:
        myStdDev = np.std(rollError)
        myMean = np.mean(rollError)
        rollAxis.set_ylim([-scaleByStdDev*myStdDev + myMean, scaleByStdDev*myStdDev + myMean])
    
    
    # if placeLegend:
    #     pitchAxis=plt.subplot2grid((3,4), (1,0),colspan=3)
    # else:
    #     pitchAxis=plt.subplot2grid((3,1), (1,0))
        
    # pitchAxis.set_title(r'Pitch error, \$\pm 1 \sigma\$ bounds')

    pitchAxis.plot(
        estimatedT-estimatedT[0],
        pitchError,
        color=colorList[0+colorCounter],
        lw=lineWeight
    )
    if attPO and plotPropagationError:
        pitchAxis.plot(
            estimatedT - estimatedT[0],
            pitchError_PO,
            color=colorList[1+ colorCounter],
            lw=lineWeight,
            ls='dashdot'
        )
    
    pitchAxis.plot(
        estimatedT-estimatedT[0],
        pitchSigma,
        color=stdDevColor,
        ls='dotted',
        lw=lineWeight
    )
    pitchAxis.plot(
        estimatedT-estimatedT[0],
        -pitchSigma,
        color=stdDevColor,
        ls='dotted',
        lw=lineWeight
    )
    pitchAxis.set_ylabel(r'pitch error \$(deg)\$')

    if scaleByStdDev:
        myStdDev = np.std(pitchError)
        myMean = np.mean(pitchError)
        pitchAxis.set_ylim([-scaleByStdDev*myStdDev + myMean, scaleByStdDev*myStdDev + myMean])
    
    pitchAxis.grid()
    
    # if placeLegend:
    #     yawAxis=plt.subplot2grid((3,4), (2,0),colspan=3)
    # else:
    #     yawAxis=plt.subplot2grid((3,1), (2,0))
        
    # yawAxis.set_title(r'Yaw error, \$\pm 1 \sigma\$ bounds')
    yawAxis.plot(
        estimatedT-estimatedT[0],
        yawError,
        color=colorList[0 + colorCounter],
        lw=lineWeight        
    )
    if attPO and plotPropagationError:
        yawAxis.plot(
            estimatedT - estimatedT[0],
            yawError_PO,
            color=colorList[1 + colorCounter],
            lw=lineWeight,
            ls='dashdot'            
        )
    
    yawAxis.plot(
        estimatedT-estimatedT[0],
        yawSigma,
        color=stdDevColor,
        ls='dotted',
        lw=lineWeight
    )
    yawAxis.plot(
        estimatedT-estimatedT[0],
        -yawSigma,
        color=stdDevColor,
        ls='dotted',
        lw=lineWeight    
    )
    yawAxis.grid()
    yawAxis.set_ylabel(r'yaw error \$(deg)\$')
    yawAxis.set_xlabel(r'time \$(s)\$')
    if scaleByStdDev:
        myStdDev = np.std(yawError)
        myMean = np.mean(yawError)
        yawAxis.set_ylim([-scaleByStdDev*myStdDev + myMean, scaleByStdDev*myStdDev + myMean])
    # plt.subplots_adjust(hspace=.5)


    rollAxis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.2f'))
    pitchAxis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.2f'))
    yawAxis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.2f'))

    rollAxis.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.f'))
    pitchAxis.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.f'))
    yawAxis.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.f'))
    
    rollAxis.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%2.2f'))
    pitchAxis.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%2.2f'))
    yawAxis.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%2.2f'))

    rollAxis.xaxis.set_minor_formatter(ticker.FormatStrFormatter('%0.f'))
    pitchAxis.xaxis.set_minor_formatter(ticker.FormatStrFormatter('%0.f'))
    yawAxis.xaxis.set_minor_formatter(ticker.FormatStrFormatter('%0.f'))
    plt.tight_layout()

    if placeLegend:
        # plt.subplot2grid((3,4), (3,0),colspan=3) 
        legendPlot = plt.subplot2grid((3,4), (0,3),rowspan=3)
        legendPlot.axis('off')
        myLegend=legendPlot.legend(
            legendLineList,
            legendLabelList,
            bbox_to_anchor=(1, 1),
            fancybox=True,
            shadow=True,
            handlelength=legendLineLength,
            borderpad=legendBorderPad,
        )
        plt.setp(myLegend.get_texts(), fontsize=30)
        
    else:
        legendDict['attitude'] = {'lines': legendLineList, 'labels': legendLabelList}
    
    if saveOutput:
        if outputFormat == 'HTML':
            mpld3.save_html(rollAxis.get_figure(), outputDir + '/attitude.html')
        elif outputFormat == 'SVG':
            plt.savefig(outputDir + '/acceleration.svg')
            
    plt.show(block=False)
    
    if axisDict is None or 'tdoaAxis' not in axisDict:
        tdoaFigure=plt.figure(figsize=(16,9))
        tdoaAxis = plt.gca()
    else:
        tdoaAxis = axisDict['tdoaAxis']
    legendLineList = []
    legendLabelList = []
    
    if clearOldPlot:
        tdoaAxis.clear()

    propOnlyRange = truePos - (velocityOnlyRange) - truePos[0]
    rangeError = truePos - estimatedPos
    if logErrorPlot:
        tdoaAxis.set_yscale('log')
        propOnlyRange = np.abs(propOnlyRange)
        rangeError = np.abs(rangeError)
        
    
    tdoaLine, = tdoaAxis.plot(
        estimatedT,
        rangeError,
        color=colorList[0+colorCounter],        
        lw=lineWeight
    )
    legendLineList.append(tdoaLine)
    legendLabelList.append(r'range error (\$\sigma=%.2f\$)'%estimatedPosStdDev_calc)

    sigmaLine, = tdoaAxis.plot(
        estimatedT,
        estimatedPosStdDev,
        color=stdDevColor,
        ls='dotted',
        lw=lineWeight        
    )
    legendLineList.append(sigmaLine)
    legendLabelList.append(r'estimated standard deviation (\$1\sigma\$)')

    if not logErrorPlot:
        tdoaAxis.plot(
            estimatedT,
            -estimatedPosStdDev,
            color=stdDevColor,
            ls='dotted',
            lw=lineWeight        
        )
    tdoaAxis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%2.f'))
    tdoaAxis.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.f'))
    
    tdoaAxis.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%2.f'))
    tdoaAxis.xaxis.set_minor_formatter(ticker.FormatStrFormatter('%0.f'))


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

    tdoaAxis.set_xlabel('time \$(s)\$')
    tdoaAxis.set_ylabel(r'TDOA error \$(km)\$')
        
    if plotPropagationError:
        tdoaPropOnlyLine, = tdoaAxis.plot(
            estimatedT,
            propOnlyRange,
            color=colorList[1 + colorCounter],
            ls='dashdot',
            lw=lineWeight
        )
        legendLineList.append(tdoaPropOnlyLine)
        legendLabelList.append('initial velocity error propagation')

    if scaleByStdDev:
        myStdDev = np.std(truePos-estimatedPos)
        myMean = np.mean(truePos-estimatedPos)
        tdoaAxis.set_ylim([-scaleByStdDev*myStdDev + myMean, scaleByStdDev*myStdDev + myMean])

    if placeLegend:
        tdoaAxis.legend(
            legendLineList,
            legendLabelList,
            bbox_to_anchor=(1, 1),
            fancybox=True,
            shadow=True,
            handlelength=legendLineLength,
            borderpad=legendBorderPad,
        )
    legendDict['tdoa'] = {'lines': legendLineList, 'labels': legendLabelList}
    
    tdoaAxis.grid()
    if saveOutput:
        if outputFormat == 'HTML':
            mpld3.save_html(tdoaAxis.get_figure(), outputDir + '/tdoa.html')
        elif outputFormat == 'SVG':
            plt.savefig(outputDir + '/tdoa.svg')
        # plt.close(tdoaFigure)
    plt.show(block=False)
    
    axisDict['tdoaAxis'] = tdoaAxis
    axisDict['attAxis'] = {'roll': rollAxis, 'pitch': pitchAxis, 'yaw': yawAxis}

    
    velocityFigure = None
    if not np.any(navVel==None):
        if axisDict is None or 'velocityAxis' not in axisDict:
            velocityFigure=plt.figure(figsize=(16,9))
            velocityAxis = plt.gca()
        else:
            velocityAxis = axisDict['velocityAxis']
        if clearOldPlot:
            velocityAxis.clear()
        velError = trueVel - navVel
        if logErrorPlot:
            velError = np.abs(velError)
            velocityAxis.set_yscale('log')
            
        velocityAxis.plot(
            estimatedT,
            velError,
            label=(
                r'velocity error (\$\sigma = %s\$)'
                %np.std(trueVel - navVel)                
            ),
            color=colorList[0 + colorCounter],
            lw=lineWeight
        )
        velocityAxis.plot(estimatedT, navVelStd,ls='dotted', lw=lineWeight, color=[0.5,0.5,0.5], label='velocity std dev')

        if not logErrorPlot:
            velocityAxis.plot(estimatedT, -navVelStd, color=[0.5,0.5,0.5],ls='dotted', lw=lineWeight,)
        if scaleByStdDev:
            myStdDev = np.std(trueVel-navVel)
            myMean = np.mean(trueVel-navVel)
            velocityAxis.set_ylim([-scaleByStdDev*myStdDev + myMean, scaleByStdDev*myStdDev + myMean])
        
        velocityAxis.legend()

        if saveOutput:
            if outputFormat == 'HTML':
                mpld3.save_html(velocityAxis.get_figure(), outputDir + '/velocity.html')
            elif outputFormat == 'SVG':
                plt.savefig(outputDir + '/velocity.svg')
            # plt.close(velocityFigure)
        # else:
        velocityAxis.grid()
        plt.show(block=False)
        axisDict['velocityAxis'] = velocityAxis
        
    legendLineList = []
    legendLabelList = []
        
    if not np.any(navAcc==None):
        if axisDict is None or 'accelerationAxis' not in axisDict:
            accelerationFigure=plt.figure(figsize=(16,9))
            accelerationAxis = plt.gca()
        else:
            accelerationAxis = axisDict['accelerationAxis']
            
        if clearOldPlot:
            accelerationAxis.clear()
        accError = trueAcc - navAcc

        if logErrorPlot:
            accError = np.abs(accError)
            accelerationAxis.set_yscale('log')
        accelerationAxis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.e'))
        accelerationAxis.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.e'))
        accLine, = accelerationAxis.plot(
            estimatedT,
            accError,
            label=(
                r'acceleration error'
            ),
            color=colorList[0 + colorCounter],
            lw=lineWeight            
        )
        legendLineList.append(accLine)
        legendLabelList.append(r'acceleration error')
        
        accSigmaLine, = accelerationAxis.plot(estimatedT, navAccStd, color=[0.5,0.5,0.5], label='acceleration std dev',ls='dotted', lw=lineWeight,)
        if not logErrorPlot:
            accelerationAxis.plot(estimatedT, -navAccStd, color=[0.5,0.5,0.5],ls='dotted', lw=lineWeight,)
        legendLineList.append(accSigmaLine)
        legendLabelList.append(r'estimated standard deviation')
        accelerationAxis.set_ylabel(r'Acceleration error \$\frac{km}{s^2}\$')
        accelerationAxis.set_xlabel(r'time \$(s)\$')

        if placeLegend:
            # plt.subplot2grid((3,4), (3,0),colspan=3) 
            legendPlot = plt.subplot2grid((3,4), (0,3),rowspan=3)
            legendPlot.axis('off')
            myLegend=legendPlot.legend(
                legendLineList,
                legendLabelList,
                bbox_to_anchor=(1, 1),
                fancybox=True,
                shadow=True,
                handlelength=legendLineLength,
                borderpad=legendBorderPad,
            )
            plt.setp(myLegend.get_texts(), fontsize=legendFont)

        else:
            legendDict['acceleration'] = {'lines': legendLineList, 'labels': legendLabelList}
        
        if scaleByStdDev:
            myStdDev = np.std(trueAcc-navAcc)
            myMean = np.mean(trueAcc-navAcc)
            accelerationAxis.set_ylim([-scaleByStdDev*myStdDev + myMean, scaleByStdDev*myStdDev + myMean])
        accelerationAxis.grid()
        if saveOutput:
            if outputFormat == 'HTML':
                mpld3.save_html(accelerationAxis.get_figure(), outputDir + '/acceleration.html')
            elif outputFormat == 'SVG':
                plt.savefig(outputDir + '/acceleration.svg')
            # plt.close(velocityFigure)
        # else:
        plt.show(block=False)
        axisDict['accelerationAxis'] = accelerationAxis
        
        # figureDict = {
        #     'tdoaFigure': tdoaFigure,
        #     'velocityFigure': velocityFigure,
        #     'attFigure': attFig,
        #     'accelerationFigure': accelerationFigure
        # }

    return(axisDict,legendDict)

def createStdDevHeader(
        inputRecord,
):
    
    myTableString = (
        r'\begin{tabular}{%' + '\n' +
        r'>{\raggedright}p{0.1\textwidth}%' + '\n' +
        r'>{\raggedright}p{0.15\textwidth}%' + '\n' +
        r'>{\raggedright}p{0.2\textwidth}%' + '\n' +
        r'>{\raggedright}p{0.12\textwidth}%' + '\n' +
        r'>{\raggedright}p{0.12\textwidth}%' + '\n' +
        r'p{0.12\textwidth}}%' + '\n' + r'\toprule'
    )
    
    myTableString += r'\multirow{3}{0.1\textwidth}{Target object} & '
    
    
    myTableString += (
        r'\multirow{3}{0.15\textwidth}{' +
        r'Initial velocity error  (' +
        inputRecord.parameters['parameters']['correlationFilter']['internalNavFilter']['initialVelocityStdDev']['unit'] +
        ')} & '
    )

    myTableString += (
        r'\multirow{3}{0.2\textwidth}{' +
        r'Angular velocity measurement error (' +
        inputRecord.parameters['parameters']['dynamicsModel']['omegaStdDev']['unit'] +
        ')} & '
    )
    
    myTableString += r'\multicolumn{3}{c}{Initial attitude (' + inputRecord.parameters['parameters']['dynamicsModel']['initialAttitudeStdDev']['DEC']['unit'] + r')} \\' + '\n'

    myTableString += r'\cmidrule(l){4-6}' + '\n'
    myTableString += r'& & & roll & pitch & yaw\\' + '\n'
    myTableString += r'& & &  &  & \\' + '\n'
    myTableString += r'\midrule\\' + '\n'
    return (myTableString)

def addInputToTable(
        resultsDict,        
        inputRecord,
        header
): 
    if 'pulsarName' in resultsDict:
        header += resultsDict['pulsarName'] + r' & '
    else:  
        header += inputRecord['parameters']['filesAndDirs']['targetObject']['value'] + r' & '
      
    header += r'\multirow{2}{*}{%.2e} &' %inputRecord['parameters']['correlationFilter']['internalNavFilter']['initialVelocityStdDev']['value']
    
    header += r'\multirow{2}{*}{%.2e} &' %inputRecord['parameters']['dynamicsModel']['omegaStdDev']['value']
    
    header += r'\multirow{2}{*}{%.2e} &' %(inputRecord['parameters']['dynamicsModel']['initialAttitudeStdDev']['roll']['value'])
    header += r'\multirow{2}{*}{%.2e} &' %(inputRecord['parameters']['dynamicsModel']['initialAttitudeStdDev']['DEC']['value'])
    header += r'\multirow{2}{*}{%.2e}' %(inputRecord['parameters']['dynamicsModel']['initialAttitudeStdDev']['RA']['value'])
    
    header += r'\\'
    header += '\n'
    if isinstance(inputRecord['parameters']['filesAndDirs']['observationID']['value'], list):
        header += r' (Obs. ID '
        firstObs = True
        for obsID in inputRecord['parameters']['filesAndDirs']['observationID']['value']:
            if not firstObs:
                header+=', '
            firstObs = False
            header+= r'%i' %obsID
            
        header += r')& & & & &\\'
        
    else:
        header += r' (Obs. ID %i)& & & & &\\' %inputRecord['parameters']['filesAndDirs']['observationID']['value']
    header += '\n'
    return header
   

def createResultsHeader(
        resultsDict
):
    myTableString = (
        r'\begin{tabular}{%' + '\n' +
        r'>{\raggedright}p{0.1\textwidth}%' + '\n' +
        r'>{\raggedright}p{0.15\textwidth}%' + '\n' +
        r'>{\raggedright}p{0.2\textwidth}%' + '\n' +
        r'>{\raggedright}p{0.12\textwidth}%' + '\n' +
        r'>{\raggedright}p{0.12\textwidth}%' + '\n' +
        r'p{0.12\textwidth}}%' + '\n' + r'\toprule'
    )
    
    # myTableString = (
    #     r'\begin{tabular}{%' + '\n' +
    #     r'>{\raggedright}p{0.15\textwidth}%' + '\n' +
    #     r'>{\raggedright}p{0.1\textwidth}%' + '\n' +
    #     r'>{\raggedright}p{0.2\textwidth}%' + '\n' +
    #     r'>{\raggedright}p{0.1\textwidth}%' + '\n' +
    #     r'>{\raggedright}p{0.1\textwidth}%' + '\n' +
    #     r'p{0.1\columnwidth}}%' + '\n' + r'\toprule'
    # )

    myTableString += r'\multirow{3}{0.15\textwidth}{Target object} & '
    
    myTableString += (
        r'\multirow{3}{0.1\textwidth}{' +
        r'Runtime (' +
        resultsDict['estimatedT']['unit'] +
        ')} & '
    )
    
    myTableString += (
        r'\multirow{3}{0.2\textwidth}{' +
        r'Range estimate error standard deviation (' +
        resultsDict['estimatedPosStdDev_calc']['unit'] +
        ')} & '
    )

    myTableString += r'\multicolumn{3}{c}{\multirow{2}{0.36\textwidth}{Attitude estimate error standard deviation (' + resultsDict['estimatedAttitudeSigma_DEG']['unit'] + r')}} \\' + '\n'

    myTableString += r'& & &  &  & \\' + '\n'
    myTableString += r'\cmidrule(l){4-6}' + '\n'

    myTableString += r'& & & roll & pitch & yaw\\' + '\n'
    myTableString += r'\midrule\\' + '\n'
    return (myTableString)

def addResultsToTable(
        resultsDict, inputs, header
):
    
    if 'pulsarName' in resultsDict:
        header += resultsDict['pulsarName'] + r' & '
    else:  
        header += inputs['parameters']['filesAndDirs']['targetObject']['value'] + r' & '
        
    header += r'\multirow{2}{*}{%.2e} &' %(resultsDict['estimatedT']['value'][-1] - resultsDict['estimatedT']['value'][0])
      
    header += r'\multirow{2}{*}{%.2e} &' %resultsDict['estimatedPosStdDev_calc']['value']
    header += r'\multirow{2}{*}{%.2e} &' %(np.std(resultsDict['attitudeError_DEG']['value'][0]) + np.abs(np.mean(resultsDict['attitudeError_DEG']['value'][0])))
    header += r'\multirow{2}{*}{%.2e} &' %(np.std(resultsDict['attitudeError_DEG']['value'][1]) + np.abs(np.mean(resultsDict['attitudeError_DEG']['value'][1])))
    header += r'\multirow{2}{*}{%.2e}' %(np.std(resultsDict['attitudeError_DEG']['value'][2]) + np.abs(np.mean(resultsDict['attitudeError_DEG']['value'][2])))
    header += r'\\'
    header += '\n'
    if isinstance(inputs['parameters']['filesAndDirs']['observationID']['value'], list):
        header += r' (Obs. ID '
        firstObs = True
        for obsID in inputs['parameters']['filesAndDirs']['observationID']['value']:
            if not firstObs:
                header+=', '
            firstObs = False
            header+= r'%i' %obsID
            
        header += r')& & & &\\'
        
    else:
        header += r' (Obs. ID %i)& & & &\\' %inputs['parameters']['filesAndDirs']['observationID']['value']
    header += '\n'
    return header


def createResultsDict(
        mySpacecraft,
        ureg,
        estimatedT,
        tdoa,
        attitude,
        velocityOnlyRangeTruncated,
        pulsarName,
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
    resultsDict['pulsarName'] = pulsarName
    
    if saveOutput:
        pickle.dump( resultsDict, open( outputDir + "/data.p", "wb" ) )

    return(resultsDict)
