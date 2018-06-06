from pypet import Trajectory
import numpy as np
import matplotlib.pyplot as plt

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
    # 'finalDelayError': {'function': plt.scatter, 'name': 'result error \$\sigma\$'},
    'finalDelayError': {'function': np.std, 'name': 'result error \$\sigma\$'},
    'finalDelayVar': {'function': meanSqrt, 'name': 'estimated error \$\sigma\$'}
}
# plotResultsKeys = {
#     'finalDelayError': {'function': np.std, 'name': 'result error \$\sigma\$'}
# }

    
plt.figure()

for resultsKey in plotResultsKeys:
    resultsDict = {}

    for run in traj.f_iter_runs(yields='idx'):
        traj.v_idx = run
        print(traj[sortByKey])
        print(traj.results[resultsKey][run])

        if traj[sortByKey] in resultsDict:
            resultsDict[traj[sortByKey]].append(traj.results[resultsKey][run][0])
        else:
            resultsDict[traj[sortByKey]] = [traj.results[resultsKey][run][0]]

    processedResultsAbs = []
    processedResultsOrd = []
    for sortKeyVal in resultsDict:
        processedResultsAbs.append(sortKeyVal)
        processedResultsOrd.append(plotResultsKeys[resultsKey]['function'](resultsDict[sortKeyVal]))


    plt.loglog(processedResultsAbs, processedResultsOrd, label=plotResultsKeys[resultsKey]['name'])

plt.grid()
plt.legend()
plt.show(block=False)
