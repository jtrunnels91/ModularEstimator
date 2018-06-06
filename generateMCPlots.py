from pypet import Trajectory
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# Define the name of the file containing the data for plotting
trajName = 'MonteCarloTest_2018_06_06_14h31m45s'
trajName = 'MonteCarloTest_2018_06_06_16h06m04s'

def meanSqrt(val):
    return np.mean(np.sqrt(val))

traj = Trajectory(trajName, add_time=False)
MCFileName = './tests/MCResults/%s.hdf5' %trajName

traj.f_load(filename=MCFileName,load_parameters=2,load_results=2)
traj.v_auto_load = True

# This key defines which simulation parameter should be the ordinate axis of
# the monte carlo simulation plot
sortByKey = 'detectorArea'

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
# plotResultsKeys = {
#     'finalDelayError': {'function': np.std, 'name': 'result error \$\sigma\$'}
# }

    
plt.figure()
ax = plt.gca()

for resultsKey in plotResultsKeys:
    print(resultsKey)
    resultsDict = {}

    varName = plotResultsKeys[resultsKey]['varName']
    for run in traj.f_iter_runs(yields='idx'):
        traj.v_idx = run
        # print(traj[sortByKey])
        # print(traj.results[resultsKey][run])

        if traj[sortByKey] in resultsDict:
            resultsDict[traj[sortByKey]].append(traj.results[varName][run][0])
        else:
            resultsDict[traj[sortByKey]] = [traj.results[varName][run][0]]

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
            scatterPointsAbs = np.ones(len(scatterPointsOrd)) * sortKeyVal
            processedResultsAbs.append(scatterPointsAbs)
            processedResultsOrd.append(scatterPointsOrd)

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
    
ax.set_yscale('log')
ax.set_xscale('log')

plt.grid()
plt.legend()
plt.show(block=False)
