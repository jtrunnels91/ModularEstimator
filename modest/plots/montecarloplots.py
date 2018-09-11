from pypet import Trajectory, pypetconstants
import numpy as np
import matplotlib.pyplot as plt
from math import isnan
plt.close('all')

def plotAreaVsAngleError(
        trajectory,
        resultsDir='/home/joel/Documents/pythonDev/modules/ModularFilter/tests/MCResults/',
        logx=True,
        logy=True,
        figure=None
        ):
    def meanSqrt(val):
        return np.mean(np.sqrt(val))
    def meanSTD(val):
        return np.std(np.mean(val))
    def meanAbs(val):
        return np.mean(np.abs(val))

    # This dictionary defines what results are to be plotted, and what operation
    # should be done to those results before plotting.
    plotResultsKeys = {
        # 'eulerErrorScatter': {
        #     'varName': 'finalEulerError',
        #     'function': np.abs,
        #     'name': 'angle error scatter',
        #     'plot': 'scatter'
        # },
        'eulerErrorSTDMeas': {
            'varName': 'finalEulerError',
            'function': np.std,
            'name': 'measured angle error \$\sigma\$',
            'plot': 'line'
        },
        'eulerErrorSTDEst': {
            'varName': 'eulerSTD',
            'function': meanSqrt,
            'name': 'estimated angle error \$\sigma\$',
            'plot': 'line'
        }
    }

    return plotTrajectory(
        trajectory,
        'detectorArea',
        plotResultsKeys,
        logx=logx,
        logy=logy,
        resultsDir=resultsDir
    )

    
def plotKeyVsError(
        trajectory,
        key,
        resultsDir='/home/joel/Documents/pythonDev/modules/ModularFilter/tests/MCResults/',
        logx=True,
        logy=True,
        figure=None
):
    def meanSqrt(val):
        return np.mean(np.sqrt(val))

    # This dictionary defines what results are to be plotted, and what operation
    # should be done to those results before plotting.
    plotResultsKeys = {
        'finalDelayErrorScatter': {
            'varName': 'finalDelayError',
            'function': np.abs,
            'name': 'result error scatter',
            'plot': 'scatter'
        },
        'finalDelayError': {
            'varName': 'finalDelayError',
            'function': np.std,
            'name': 'result error \$\sigma\$',
            'plot': 'line'
        },
        'finalDelayVar': {
            'varName': 'finalDelayVar',
            'function': meanSqrt,
            'name': 'estimated error \$\sigma\$',
            'plot': 'line'
        }
    }

    return plotTrajectory(
        trajectory,
        key,
        plotResultsKeys,
        logx=logx,
        logy=logy,
        resultsDir=resultsDir,
        figure=figure
    )

def plotNTapsVsError(trajectory, figure=None):
    def meanSqrt(val):
        return np.mean(np.sqrt(val))

    # This key defines which simulation parameter should be the ordinate axis of
    # the monte carlo simulation plot
    sortByKey = 'filterTaps'

    # This dictionary defines what results are to be plotted, and what operation
    # should be done to those results before plotting.
    plotResultsKeys = {
        'finalDelayErrorScatter': {
            'varName': 'finalDelayError',
            'function': np.abs,
            'name': 'result error scatter',
            'plot': 'scatter'
        },
        'finalDelayError': {
            'varName': 'finalDelayError',
            'function': np.std,
            'name': 'result error \$\sigma\$',
            'plot': 'line'
        },
        'finalDelayVar': {
            'varName': 'finalDelayVar',
            'function': meanSqrt,
            'name': 'estimated error \$\sigma\$',
            'plot': 'line'
        }
    }

    return plotTrajectory(trajectory, sortByKey, plotResultsKeys, figure=None, logx=False)

def plotAreaVsError(
        trajectory,
        rejectNonPeakLock=False,
        figure=None,
        resultsDir='/home/joel/Documents/pythonDev/modules/ModularFilter/tests/MCResults/'
):

    def meanSqrt(val):
        return np.mean(np.sqrt(val))

    # This key defines which simulation parameter should be the ordinate axis of
    # the monte carlo simulation plot
    sortByKey = 'detectorArea'

    if rejectNonPeakLock is True:
        rejectName = 'peakLock'
    else:
        rejectName = None
    # This dictionary defines what results are to be plotted, and what operation
    # should be done to those results before plotting.
    plotResultsKeys = {
        'finalDelayErrorScatter': {
            'varName': 'finalDelayError',
            'function': np.abs,
            'name': 'result error scatter',
            'plot': 'scatter',
            'reject': rejectName
        },
        'finalDelayError': {
            'varName': 'finalDelayError',
            'function': np.std,
            'name': 'result error \$\sigma\$',
            'plot': 'line',
            'reject': rejectName
        },
        'finalDelayVar': {
            'varName': 'finalDelayVar',
            'function': meanSqrt,
            'name': 'estimated error \$\sigma\$',
            'plot': 'line',
            'reject': rejectName
        }
    }

    return plotTrajectory(trajectory, sortByKey, plotResultsKeys, figure=figure, resultsDir=resultsDir)

def plotTrajWithFunction(
        traj,
        abscissa,
        ordinate,
        function=None,
        axis=None,
        plotOptions=None
    ):
    traj.f_load(load_results=pypetconstants.LOAD_DATA)

    if axis is None:
        axis = plt.gca()

    excludeNaN = False
    if plotOptions:
        if 'logx' in plotOptions and plotOptions['logx']:
            axis.set_xscale('log')
        if 'logy' in plotOptions and plotOptions['logy']:
            axis.set_yscale('log')
        if 'excludeNaN' in plotOptions and plotOptions['excludeNaN']:
            excludeNaN = True
    
    abscissaOrdinateDict = {}
    abscissaUnits = None
    ordinateUnits = None
    for run in traj.f_iter_runs(yields='idx'):
        traj.v_idx = run
        
        newAbscissaVal, abscissaUnits = checkForUnits(
            traj.parameters[abscissa],
            abscissaUnits
        )
        newOrdinateVal, ordinateUnits = checkForUnits(
            traj.results[ordinate][run],
            ordinateUnits
        )
        if not excludeNaN or not isnan(newOrdinateVal):
            if newAbscissaVal in abscissaOrdinateDict:
                abscissaOrdinateDict[newAbscissaVal].append(newOrdinateVal)
            else:
                abscissaOrdinateDict[newAbscissaVal] = [newOrdinateVal]

    abscissaList = []
    ordinateList = []
    for abscissaVal, ordinateVals in abscissaOrdinateDict.items():
        abscissaList.append(abscissaVal)
        if function is not None:
            ordinateList.append(function(ordinateVals))
        else:
            ordinateList.append(ordinateVals)
    myLine = axis.plot(abscissaList, ordinateList)
    
    if abscissaUnits is not None:
        axis.set_xlabel(abscissa + ' (' + abscissaUnits + ')')
    else:
        axis.set_xlabel(abscissa)
        
    if ordinateUnits is not None:
        axis.set_ylabel('(' + ordinateUnits + ')')
    return myLine
    

def checkForUnits(myVal, currentUnits):
    if currentUnits is not None:
        if myVal.unit != currentUnits:
            raise ValueError('Units must match')
        myVal = myVal.value
    else:
        try:
            currentUnits = myVal.unit
            myVal = myVal.value
        except:
            currentUnits = None
    return myVal, currentUnits
    

def scatterPlotTraj(
        traj,
        abscissa,
        ordinate,
        axis=None,
        plotOptions=None,
        function=None
    ):
    traj.f_load(load_results=pypetconstants.LOAD_DATA)
    
    excludeNaN = False
    if plotOptions:
        if 'logx' in plotOptions and plotOptions['logx']:
            axis.set_xscale('log')
        if 'logy' in plotOptions and plotOptions['logy']:
            axis.set_yscale('log')
        if 'excludeNaN' in plotOptions and plotOptions['excludeNaN']:
            excludeNaN = True
    
    if axis is None:
        axis = plt.gca()

    abscissaList = []
    ordinateList = []
    abscissaUnits = None
    ordinateUnits = None
    for run in traj.f_iter_runs(yields='idx'):
        traj.v_idx = run

        newAbscissaVal, abscissaUnits = checkForUnits(
            traj.parameters[abscissa],
            abscissaUnits
        )
        newOrdinateVal, ordinateUnits = checkForUnits(
            traj.results[ordinate][run],
            ordinateUnits
        )
        
        if not excludeNaN or not isnan(newOrdinateVal):

            abscissaList.append(newAbscissaVal)
            if function is not None:
                ordinateList.append(function(newOrdinateVal))
            else:
                ordinateList.append(newOrdinateVal)
                
    if abscissaUnits is not None:
        axis.set_xlabel(abscissa + ' (' + abscissaUnits + ')')
    else:
        axis.set_xlabel(abscissa)
        
    if ordinateUnits is not None:
        axis.set_ylabel('(' + ordinateUnits + ')')

    myPoints = axis.scatter(abscissaList, ordinateList)

    return myPoints

def plotTrajectory(
        trajPlot,
        sortByKey,
        plotResultsKeys,
        figure=None,
        resultsDir='/home/joel/Documents/pythonDev/modules/ModularFilter/tests/MCResults/',
        logx=True,
        logy=True
        ):
    if not isinstance(trajPlot, Trajectory):
        trajName = trajPlot.replace('.hdf5', '')
        trajPlot = Trajectory(trajName, add_time=False)
        MCFileName = resultsDir + '%s.hdf5' %trajName

        trajPlot.f_load(filename=MCFileName,load_parameters=2,load_results=2)
        trajPlot.v_auto_load = True
    else:
        trajPlot.f_load(load_results=pypetconstants.LOAD_DATA)

    if figure is None:
        figure = plt.figure()
    ax = plt.gca()
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    for resultsKey in plotResultsKeys:
        ordinateUnit = None
        resultsDict = {}

        varName = plotResultsKeys[resultsKey]['varName']
        if 'reject' in plotResultsKeys[resultsKey]:
            rejectName = plotResultsKeys[resultsKey]['reject']
        else:
            rejectName = None
        for run in trajPlot.f_iter_runs(yields='idx'):
            trajPlot.v_idx = run
            # print(traj[sortByKey])
            # print(traj.results[resultsKey][run])

            if trajPlot[sortByKey] in resultsDict:
                # newVal = trajPlot.results[varName][run][0]

                try:
                    newVal = trajPlot.results[varName][run].value
                    ordinateUnits = trajPlot.results[varName][run].unit
                except:
                    newVal = trajPlot.results[varName][run]
                try:
                    valueValid = not isnan(newVal)
                except:
                    valueValid = not np.any([isnan(subValue) for subValue in newVal])
                if valueValid:
                    if (
                            rejectName is not None and (trajPlot.results[rejectName][run])
                    ) or rejectName is None:
                        resultsDict[trajPlot[sortByKey]].append(
                            newVal
                        )
                        
            else:
                # resultsDict[trajPlot[sortByKey]] = [trajPlot.results[varName][run][0]]
                resultsDict[trajPlot[sortByKey]] = [trajPlot.results[varName][run]]

        processedResultsAbs = []
        processedResultsOrd = []
        abscissaUnits = None
        for sortKeyVal in resultsDict:
            currentResult = resultsDict[sortKeyVal]
            try:
                currentResult=[subResult.value for subResult in currentResult]
            except:
                currentResult = currentResult
            try:
                abscissaUnits = sortKeyVal.unit
                sortKeyVal = sortKeyVal.value
            except:
                abscissaUnits = None
                
            if plotResultsKeys[resultsKey]['plot'] == 'line':
                
                processedResultsAbs.append(sortKeyVal)
                if plotResultsKeys[resultsKey]['function'] is not None:
                    processedResultsOrd.append(
                        plotResultsKeys[resultsKey]['function'](currentResult)
                    )
                    
            elif plotResultsKeys[resultsKey]['plot'] == 'scatter':
                if plotResultsKeys[resultsKey]['function'] is not None:
                    scatterPointsOrd = plotResultsKeys[resultsKey]['function'](
                        currentResult
                    )
                else:
                    scatterPointsOrd = resultsDict[sortKeyVal]
                scatterPointsAbs = (np.ones(len(scatterPointsOrd)) * sortKeyVal).tolist()
                processedResultsAbs += scatterPointsAbs
                if not isinstance(scatterPointsOrd, list):
                    scatterPointsOrd = scatterPointsOrd.tolist()
                processedResultsOrd += scatterPointsOrd

        if plotResultsKeys[resultsKey]['plot'] == 'line':
            plt.plot(
                processedResultsAbs,
                processedResultsOrd,
                label=plotResultsKeys[resultsKey]['name']
            )

        elif plotResultsKeys[resultsKey]['plot'] == 'scatter':
            plt.scatter(
                processedResultsAbs,
                processedResultsOrd,
                label=plotResultsKeys[resultsKey]['name']
            )
            print('scatterplot!')


    sortByKey = sortByKey.replace('.value', '')
    try:
        myXLabel = sortByKey + ' (' + trajPlot[sortByKey].unit + ')'
    except:
        myXLabel = sortByKey
    plt.xlabel(myXLabel)
    plt.grid()
    plt.legend()
    plt.show(block=False)
    return figure
