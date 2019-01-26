from . import UserData
import numpy as np

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


def executeSimulation(myExplorationParameters, myFunction, myUserData):
    remainingParameters = dict(myExplorationParameters)
    key = next(iter(myExplorationParameters))
    value = myExplorationParameters[key]
    remainingParameters.pop(key)
    if len(remainingParameters) > 0:
        resultList = []
        for subval in value:
            setParameters(myUserData, key, subval)
            resultList += executeSimulation(remainingParameters, myFunction, myUserData)
    else:
        resultList = []
        for subval in value:
            setParameters(myUserData, key, subval)
            singleResult = myFunction(myUserData)
            resultList.append(
                {
                    'parameters': myUserData.toDict(),
                    'results': singleResult
                }
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

def runSimulation(userData, function):
    results=executeSimulation(
        findExplorationParameters(userData.exploreParameters),
        function,
        userData.parameters
    )
    return results

