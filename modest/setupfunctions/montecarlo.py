from . import UserData
import numpy as np
import pickle
import multiprocessing as mp

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
        if ('start' and
            'stop' and
            'number' and
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
        useMultiProcessing=False
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
                totalExplorationParameters
            )
    else:
        if useMultiProcessing:
            output = mp.Queue()
            myProcessList = []
        
        for subval in value:
            currentKeyValueDict[key] = subval
            setParameters(myUserData, key, subval)
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
            print("||=================================================||")

            if useMultiProcessing:
                myProcessList.append(
                    mp.Process(target=myFunction, args=(myUserData, currentKeyValueDict, output))
                )
                               
            else:
                try:
                    singleResult = myFunction(myUserData, currentKeyValueDict)
                except:
                    singleResult = 'RUN FAILED'
                resultList.append(
                    {
                        'parameters': myUserData.toDict(),
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
            for p in myProcessList:
                p.start()
            for p in myProcessList:
                p.join()
            resultsList = resultList + [output.get() for p in myProcessList]
            
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

def runSimulation(userData, function, outputFileName, useMultiProcessing = False):
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
        useMultiProcessing=useMultiProcessing
    )
    return results

