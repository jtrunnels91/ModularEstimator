from . import UserData
import numpy as np

def findExplorationParameters(myUserData):
    if ('start' and
        'stop' and
        'number' and
        'rangeType' in myUserData
    ):
        if myUserData.rangeType == 'linear':
            myExplorationParameters = np.linspace(myUserData.start, myUserData.stop, myUserData.number)
        elif myUserData.rangeType == 'log':
            myExplorationParameters = np.logspace(myUserData.start, myUserData.stop, myUserData.number)
        else:
            raise ValueError('Unrecougnized range type')
    else:
        myExplorationParameters = {}
        for key, value in myUserData.items():
            newParameters = findExplorationParameters(value)
            if isinstance(newParameters, dict):
                subKey = next(iter(newParameters))
                newParameters = newParameters[subKey]
                newKey = key + '.' + subKey
            else:
                newKey = key
                newParameters
            myExplorationParameters[newKey] = newParameters
    return myExplorationParameters


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

