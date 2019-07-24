import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import moment
# from .. setupfunctions import montecarlo
from .. utils.spacegeometry import sigmaDeltaT_Theoretical

def stdMean(vals):
    return np.abs(np.mean(vals)) + np.std(vals)

def meanError(vals):
    return np.mean(np.abs(vals))

def varVar(vals):
    variance = np.var(vals)
    fourthCentral = moment(vals,4)
    number = len(vals)
    varianceVariance = (fourthCentral - (np.power(variance,2)*(number-3))/(number-1))/number
    return varianceVariance

def stdStd(vals):
    return np.sqrt(varVar(vals))

def uniformStdDev(pulsarPeriod):
    return pulsarPeriod/np.sqrt(12)

def plotMCResults(
        resultsDict,
        abscissaDict,
        ordinateDict,
        ureg,
        plotType='line',
        axis=None,
        includeOnly=None,
        color=None,
        lineStyle='-',
        marker='o',
        markerSize=2,
        includeErrorBars=False
):

    if axis is None:
        myFig = plt.figure()
        axis = plt.gca()
    abscissaUnits=None
    ordinateUnits=None
    abscissaOrdinateValues ={}

    abscissaKeyList = None
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

            if 'key' in ordinateDict:
                ordinateValue = result['results'][ordinateDict['key']]
            elif ordinateDict['function'] == 'sigmaDeltaT_Theoretical':
                pulsarPeriod = (
                    result['results']['pulsarPeriod']['value'] *
                    ureg(result['results']['pulsarPeriod']['unit'])
                ).to(ureg.seconds).magnitude
                pulsarFlux = (
                    result['results']['pulsarFlux']['value']
                )
                backgroundCountRate = result['results']['backgroundCountRate']['value']
                pulsedFraction = resultsDict[0]['results']['pulsedFraction']['value']
                
                pulsarFWHM = resultsDict[0]['results']['pulsarFWHM']['value']
                
                runtime = (
                    resultsDict[0]['parameters']['simulation']['runtime']['value'] *
                    ureg(resultsDict[0]['parameters']['simulation']['runtime']['unit'])
                ).to(ureg.seconds).magnitude
                         
                ordinateValue = {
                    'value': sigmaDeltaT_Theoretical(
                        pulsarPeriod,
                        pulsarFlux,
                        pulsedFraction,
                        pulsarFWHM,
                        1,
                        runtime,
                        backgroundFlux=backgroundCountRate
                    ),
                    'unit': 'seconds'
                }
                
            if abscissaValue in abscissaOrdinateValues:
                abscissaOrdinateValues[abscissaValue].append(ordinateValue['value'])
            else:
                abscissaOrdinateValues[abscissaValue] = [ordinateValue['value']]
                if 'label' not in ordinateDict:
                    ordinateDict['label'] = ordinateValue['comment']
                ordinateUnits = ordinateValue['unit']
    errorBars = []
    if 'function' in ordinateDict:
        for abscissa, ordinate in abscissaOrdinateValues.items():
            if ordinateDict['function'] == 'std + mean':
                # print(np.std(ordinate))
                if includeErrorBars:
                    myStdStd = stdStd(ordinate)
                    errorBars += [myStdStd*3]
                ordinate = np.std(ordinate) + np.abs(np.mean(ordinate))
                
            elif ordinateDict['function'] == 'uniformStdDev':
                ordinate = np.mean([uniformStdDev(subValue) for subValue in ordinate])
                
            elif ordinateDict['function'] == 'sigmaDeltaT_Theoretical':
                ordinate = np.mean(ordinate)
            else:
                ordinate = getattr(np,ordinateDict['function'])(ordinate)

            abscissaOrdinateValues[abscissa] = ordinate
    abscissaOrdinateList = [(abscissa, ordinate) for abscissa, ordinate in abscissaOrdinateValues.items()]

    abscissaOrdinateList.sort()
    abscissaList = [abscissa for abscissa, ordinate in abscissaOrdinateList]
    ordinateList = [ordinate for abscissa, ordinate in abscissaOrdinateList]

    print(ordinateDict['label'])
    # print(abscissaList)
    # print(ordinateList)
    if plotType == 'line':
        if color:
            myLine = axis.plot(abscissaList, ordinateList, color=color,lw=4, ls=lineStyle, marker=marker, ms=markerSize)
            if errorBars:
                axis.errorbar(abscissaList,ordinateList,yerr=errorBars)
        else:
            myLine = axis.plot(abscissaList, ordinateList)
        myLabel = ordinateDict['label']
    elif plotType == 'scatter':
        for counter in range(len(abscissaList)):
            abscissaList[counter] = np.ones(len(ordinateList[counter])) * abscissaList[counter]
        for subAbscissa, subOrdinate in zip(abscissaList, ordinateList):
            myLine = axis.scatter(subAbscissa, subOrdinate, color=color, marker=marker, s=markerSize)
        myLabel = ordinateDict['label']

    return {
        'line': myLine,
        'label': myLabel,
        'axis': axis,
        'abscissaUnits': abscissaUnits,
        'ordinateUnits': ordinateUnits
    }
