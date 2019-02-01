import matplotlib.pyplot as plt
import numpy as np
# from .. setupfunctions import montecarlo

def plotMCResults(
        resultsDict,
        abscissaDict,
        ordinateDict,
        plotType='line',
        axis=None,
        includeOnly=None
):

    if axis is None:
        myFig = plt.figure()
        axis = plt.gca()
    abscissaUnits=None
    ordinateUnits=None
    abscissaOrdinateValues ={}
    abscissaKeyList = abscissaDict['key'].split('.')
    for result in resultsDict:
        includeCurrent = False
        if not includeOnly:
            includeCurrent = True
        else:
            includeArray = []
            for criteria in includeOnly:
                includeParam = result
                for key in criteria[0]:
                    includeParam = includeParam[key]
                includeParam = includeParam['value']
                if includeParam == criteria[1]:
                    includeArray.append(True)
                else:
                    includeArray.append(False)
            includeCurrent = np.all(includeArray)
                
        if includeCurrent:
            abscissaValue = result['parameters']
            for abscissaKey in abscissaKeyList:
                abscissaValue = abscissaValue[abscissaKey]
            while 'value' not in abscissaValue:
                abscissaValue = abscissaValue[next(iter(abscissaValue))]
            abscissaUnits = abscissaValue['unit']
            abscissaValue = abscissaValue['value']
            ordinateValue = result['results'][ordinateDict['key']]
            if abscissaValue in abscissaOrdinateValues:
                abscissaOrdinateValues[abscissaValue].append(ordinateValue['value'])
            else:
                abscissaOrdinateValues[abscissaValue] = [ordinateValue['value']]
                if 'label' not in ordinateDict:
                    ordinateDict['label'] = ordinateValue['comment']
                ordinateUnits = ordinateValue['unit']
    if 'function' in ordinateDict:
        for abscissa, ordinate in abscissaOrdinateValues.items():
            ordinate = getattr(np,ordinateDict['function'])(ordinate)
            abscissaOrdinateValues[abscissa] = ordinate
    abscissaOrdinateList = [(abscissa, ordinate) for abscissa, ordinate in abscissaOrdinateValues.items()]

    abscissaOrdinateList.sort()
    abscissaList = [abscissa for abscissa, ordinate in abscissaOrdinateList]
    ordinateList = [ordinate for abscissa, ordinate in abscissaOrdinateList]

    
    if plotType == 'line':
        myLine = axis.plot(abscissaList, ordinateList)
        myLabel = ordinateDict['label']
    elif plotType == 'scatter':
        for counter in range(len(abscissaList)):
            abscissaList[counter] = np.ones(len(ordinateList[counter])) * abscissaList[counter]
        myLine = axis.scatter(abscissaList, ordinateList)
        myLabel = ordinateDict['label']

    return {
        'line': myLine,
        'label': myLabel,
        'axis': axis,
        'abscissaUnits': abscissaUnits,
        'ordinateUnits': ordinateUnits
    }
