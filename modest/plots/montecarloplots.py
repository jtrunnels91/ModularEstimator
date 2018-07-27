from pypet import Trajectory, pypetconstants
import numpy as np
import matplotlib.pyplot as plt
from math import isnan
plt.close('all')

def plotAreaVsAngleError(
        trajectory,
        resultsDir='/home/joel/Documents/pythonDev/modules/ModularFilter/tests/MCResults/',
        logx=True,
        logy=True
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

    return plotTrajectory(trajectory,
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
        logy=True

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

    return plotTrajectory(trajectory,
                   key,
                   plotResultsKeys,
                   logx=logx,
                   logy=logy,
                   resultsDir=resultsDir
    )

def plotNTapsVsError(trajectory):
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

    return plotTrajectory(trajectory, sortByKey, plotResultsKeys, logx=False)

def plotAreaVsError(
        trajectory,
        rejectNonPeakLock=False,
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

    return plotTrajectory(trajectory, sortByKey, plotResultsKeys, resultsDir=resultsDir)
    
def plotTrajectory(
        trajPlot,
        sortByKey,
        plotResultsKeys,
        resultsDir='/home/joel/Documents/pythonDev/modules/ModularFilter/tests/MCResults/',
        logx=True,
        logy=True
        ):
    if not isinstance(trajPlot, Trajectory):
        trajName = trajPlot
        trajPlot = Trajectory(trajName, add_time=False)
        MCFileName = resultsDir + '%s.hdf5' %trajName

        trajPlot.f_load(filename=MCFileName,load_parameters=2,load_results=2)
        trajPlot.v_auto_load = True
    else:
        trajPlot.f_load(load_results=pypetconstants.LOAD_DATA)

    myFigure = plt.figure()
    ax = plt.gca()

    for resultsKey in plotResultsKeys:
        print(resultsKey)
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
                newVal = trajPlot.results[varName][run][0]
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
                resultsDict[trajPlot[sortByKey]] = [trajPlot.results[varName][run][0]]

        processedResultsAbs = []
        processedResultsOrd = []
        for sortKeyVal in resultsDict:
            if plotResultsKeys[resultsKey]['plot'] is 'line':
                processedResultsAbs.append(sortKeyVal)
                if plotResultsKeys[resultsKey]['function'] is not None:
                    processedResultsOrd.append(
                        plotResultsKeys[resultsKey]['function'](resultsDict[sortKeyVal])
                    )
                    
            elif plotResultsKeys[resultsKey]['plot'] is 'scatter':
                if plotResultsKeys[resultsKey]['function'] is not None:
                    scatterPointsOrd = plotResultsKeys[resultsKey]['function'](
                        resultsDict[sortKeyVal]
                    )
                else:
                    scatterPointsOrd = resultsDict[sortKeyVal]
                scatterPointsAbs = (np.ones(len(scatterPointsOrd)) * sortKeyVal).tolist()
                processedResultsAbs += scatterPointsAbs
                if not isinstance(scatterPointsOrd, list):
                    scatterPointsOrd = scatterPointsOrd.tolist()
                processedResultsOrd += scatterPointsOrd

        if plotResultsKeys[resultsKey]['plot'] is 'line':
            plt.plot(
                processedResultsAbs,
                processedResultsOrd,
                label=plotResultsKeys[resultsKey]['name']
            )

        elif plotResultsKeys[resultsKey]['plot'] is 'scatter':
            print('scatterplot!')
            plt.scatter(
                processedResultsAbs,
                processedResultsOrd,
                label=plotResultsKeys[resultsKey]['name']
            )

    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    plt.grid()
    plt.legend()
    plt.show(block=False)
    return myFigure
