import matplotlib.pyplot as plt
import numpy as np

def plotSourcesAndProbabilities(modFilter, measurementList, pointSize=1):
    plt.figure()
    for signalName, signal in modFilter.signalSources.items():
        if hasattr(signal, 'RaDec'):
            myPoints=plt.scatter(signal.RaDec()['RA'], signal.RaDec()['DEC'], label=signalName,marker='*')
            myColor = myPoints.get_facecolor()
        else:
            myColor = [[0.75,0.75,0.75,1]]

        probArray = [
            singleMeas['associationProbabilities'][signalName]
            for singleMeas in measurementList
        ]
        probArray = np.array(probArray)
        # probArray = probArray - np.min(probArray)
        # probArray = probArray/np.max(probArray)
        
        myRaList = []
        myDecList = []
        myProbList = []
        for index, photonMeasurement in enumerate(photonMeasurementList):
            if probArray[index] > 0:
                myRaList.append(photonMeasurement['TrueRA'])
                myDecList.append(photonMeasurement['TrueDEC'])
                myProbList.append(probArray[index])

        myProbList = np.array(myProbList)
        plt.scatter(myRaList, myDecList, color=myColor, edgecolors=myColor, marker='.', s=myProbList*pointSize)

        c = np.asarray([
            list(myColor[0][0:3]) + [prob]
            for prob in myProbList]
        )

    euList=[
        svDict['eulerAngles']
        for svDict in myFilter.subStates['attitude']['stateObject'].stateVectorHistory
    ]
    plt.plot([eu[2] for eu in euList],[-eu[1] for eu in euList])
    plt.legend()
    plt.show(block=False)


def photonScatterPlot(
        photonMeasurements,
        probabilityAlpha=None,
        alpha=1.0,
        size=1,
        axis=None,
        color=None
):
    if axis is None:
        plt.figure()
        axis = plt.gca()
    if probabilityAlpha is None:
        if color is None:
            points = axis.scatter(
                [p['RA']['value'] for p in photonMeasurements],
                [p['DEC']['value'] for p in photonMeasurements],
                marker='.', s=size, alpha=alpha
            )
        else:
            points= axis.scatter(
                [p['RA']['value'] for p in photonMeasurements],
                [p['DEC']['value'] for p in photonMeasurements],
                marker='.', s=size, alpha=alpha, color=color
            )
    else:
        for p in photonMeasurements:
            
            prAlpha = np.float(alpha*p['associationProbabilities'][probabilityAlpha])
            if color is None:
                point = axis.scatter(
                    p['RA']['value'], p['DEC']['value'],
                    marker='.',
                    s=size,
                    alpha=prAlpha
                    )
                color = point.properties()['facecolor']
            else:
                point = axis.scatter(
                    p['RA']['value'], p['DEC']['value'],
                    marker='.',
                    s=size,
                    alpha=prAlpha,
                    color=color
                    )

    plt.show(block=False)
    
