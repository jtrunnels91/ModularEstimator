import matplotlib.pyplot as plt
import numpy as np
# from .. setupfunctions import montecarlo

def plotMCResults(
        resultsDict,
        abscissaDict,
        ordinateDict,
        plotType='line',
        axis=None
):

    if axis is None:
        myFig = plt.figure()
        axis = plt.gca()

    print(ordinateDict['label'])
    abscissaOrdinateValues ={}
    abscissaKeyList = abscissaDict['key'].split('.')
    for result in resultsDict:
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
            ordinate = ordinateDict['function'](ordinate)
            abscissaOrdinateValues[abscissa] = ordinate
    abscissaOrdinateList = [(abscissa, ordinate) for abscissa, ordinate in abscissaOrdinateValues.items()]

    abscissaOrdinateList.sort()
    abscissaList = [abscissa for abscissa, ordinate in abscissaOrdinateList]
    ordinateList = [ordinate for abscissa, ordinate in abscissaOrdinateList]
    print(ordinateDict['label'])


    if plotType is 'line':
        myLine = axis.plot(abscissaList, ordinateList)
        myLabel = ordinateDict['label']

    return {
        'line': myLine,
        'label': myLabel,
        'axis': axis,
        'abscissaUnits': abscissaUnits,
        'ordinateUnits': ordinateUnits
    }
