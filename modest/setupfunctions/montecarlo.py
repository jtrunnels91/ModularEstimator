from . import UserData
import numpy as np
import pickle
import multiprocessing as mp
import copy
np.random.seed()
def findUniqueParameters(resultsDict, parameterString):
    parameterList = parameterString.split('.')
    uniqueParameterValues = []

    for result in resultsDict:
        for parameter in parameterList:
            result = result[parameter]
        result = result['value']
        if result not in uniqueParameterValues:
            uniqueParameterValues.append(result)
    return uniqueParameterValues

def findExplorationParameters(myUserData):
    myExplorationParametersDict = {}
    for parameterName, parameter in myUserData.items():
        if ('start'     in parameter and
            'stop'      in parameter and
            'number'    in parameter and
            'rangeType' in parameter
        ):
            if parameter.rangeType == 'linear':
                myExplorationParameters = np.linspace(
                    parameter.start, parameter.stop, parameter.number)
            elif parameter.rangeType == 'log':
                myExplorationParameters = np.logspace(
                    parameter.start, parameter.stop, parameter.number)
            else:
                raise ValueError('Unrecougnized range type')
        elif 'valueList' in parameter:
            myExplorationParameters = parameter.valueList
        elif parameter.rangeType == 'randint':
            if 'upper' and 'lower' in parameter:
                myExplorationParameters = np.random.randint(
                    parameter.start, parameter.stop, parameter.number
                )
            else:
                myExplorationParameters = np.random.randint(
                    0, (2**32)-1, parameter.number
                )
        else:
            raise ValueError('Unrecougnized range type')

        myExplorationParametersDict[parameterName] = myExplorationParameters
    return myExplorationParametersDict


def executeSimulation(
        myExplorationParameters,
        myFunction,
        myUserData,
        outputFileName,
        resultList,
        currentKeyValueDict,
        totalExplorationParameters,
        useMultiProcessing=False,
        runSafe=False
):
    remainingParameters = dict(myExplorationParameters)
    key = next(iter(myExplorationParameters))
    value = myExplorationParameters[key]
    remainingParameters.pop(key)
    if len(remainingParameters) > 0:
        for subval in value:
            currentKeyValueDict[key] = subval
            setParameters(myUserData, key, subval)
            resultList = executeSimulation(
                remainingParameters,
                myFunction,
                myUserData,
                outputFileName,
                resultList,
                currentKeyValueDict,
                totalExplorationParameters,
                useMultiProcessing=useMultiProcessing,
                runSafe=runSafe
            )
    else:
        if useMultiProcessing:
            myPool = mp.Pool(mp.cpu_count())
            myParameterList = []
        
        for subval in value:
            currentKeyValueDict[key] = subval
            currentUserData = copy.deepcopy(setParameters(myUserData, key, subval))
            currentKeyValueDict['currentRun'] += 1
            print()
            print()
            print("||=================================================||")
            print("  MONTE CARLO SIMULATION EXECUTOR ")
            print("  Initializing process for run %i of %i " %(
                currentKeyValueDict['currentRun'], currentKeyValueDict['totalRuns']
            ))
            for currentValKey, currentVal in currentKeyValueDict.items():
                if currentValKey != 'currentRun' and currentValKey != 'totalRuns':
                    print("  %s = %s"  %(currentValKey, currentVal))
            if useMultiProcessing:
                print("  Running using multiprocessing")
            print("||=================================================||")

            if useMultiProcessing:
                myParameterList.append(
                    currentUserData
                )
                               
            else:
                if runSafe:
                    try:
                        singleResult = myFunction(currentUserData)
                    except:
                        singleResult = 'RUN FAILED'
                else:
                    singleResult = myFunction(currentUserData)
                resultList.append(
                    {
                        'parameters': currentUserData.toDict(),
                        'results': singleResult
                    }
                )
                with open( outputFileName, "wb" ) as myPickle:
                    pickle.dump(
                        {
                            'results':resultList,
                            'explorationParameters': totalExplorationParameters
                        },
                        myPickle
                    )
        if useMultiProcessing:
            newResults = myPool.map(myFunction, myParameterList)
            newResults = [
                {
                    'parameters': parameter.toDict(),
                    'results': result
                }
                for parameter, result in zip(myParameterList,newResults)
            ]
            resultList = resultList + newResults
            
            myPool.close()
            myPool.join()
            with open( outputFileName, "wb" ) as myPickle:
                pickle.dump(
                    {
                        'results':resultList,
                        'explorationParameters': totalExplorationParameters
                    },
                    myPickle
                )
            
    return resultList

    

def setParameters(myUserData, parameterString, newValue):
    if isinstance(parameterString, str):
        parameterList = parameterString.split('.')
    else:
        parameterList = parameterString

    modifiedUserData = myUserData
    for parameter in parameterList:
        modifiedUserData = modifiedUserData[parameter]
    if 'value' in modifiedUserData:
        modifiedUserData.value = newValue
    else:
        if isinstance(modifiedUserData, UserData):
            for key, subItem in modifiedUserData.items():
                setParameters(modifiedUserData, key, newValue)
        # for key, subItem in modifiedUserData.items():
        #     if 'value' in subItem:
        #         subItem.value = newValue
    return myUserData

def runSimulation(userData, function, outputFileName, useMultiProcessing = False, runSafe=False):
    exploreParameters = findExplorationParameters(userData.exploreParameters)
    totalRuns = 1
    for key, value in exploreParameters.items():
        totalRuns = totalRuns * len(value)
    currentStatusDict = {
        'totalRuns': totalRuns,
        'currentRun': 0
    }
    results=executeSimulation(
        exploreParameters,
        function,
        userData.parameters,
        outputFileName,
        [],
        currentStatusDict,
        exploreParameters,
        useMultiProcessing=useMultiProcessing,
        runSafe=runSafe
    )
    return results

