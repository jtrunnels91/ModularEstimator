import numpy as np

def estimateTDOA(
        photonMeasurements,
        pulsarObject,
        maxMLEResolution=None,
        MLEBins=20
):
    pulsarPeriod = pulsarObject.getPeriod(photonMeasurements[0]['t']['value'])
    
    if not maxMLEResolution:
        maxMLEResolution = pulsarPeriod/100
    else:
        maxMLEResolution = pulsarPeriod * maxMLEResolution

    currentTimeResolution = pulsarPeriod/MLEBins
    tSearchLowerBound = 0
    tSearchUpperBound = currentTimeResolution * (MLEBins-1)
    while currentTimeResolution > maxMLEResolution:
        tSearchVector = np.linspace(tSearchLowerBound, tSearchUpperBound, MLEBins)
        likelihoodVector = np.zeros(MLEBins)

        for tSearchIndex in range(len(tSearchVector)):
            currentTimeOffset = tSearchVector[tSearchIndex]
            fluxValues = np.array([
                pulsarObject.signalIntegral(
                    photon['t']['value'] + currentTimeOffset,
                    photon['t']['value'] + currentTimeResolution + currentTimeOffset
                )
                for photon in photonMeasurements
            ])
            likelihoodVector[tSearchIndex] = (
                np.sum(
                    np.log(fluxValues)
                )
            )
        myArgMax = np.argmax(likelihoodVector)
        maxLikelihoodTDOA = tSearchVector[myArgMax]

        currentTimeResolution = currentTimeResolution/MLEBins

        tSearchLowerBound = maxLikelihoodTDOA
        tSearchUpperBound = maxLikelihoodTDOA + (currentTimeResolution * (MLEBins-1))

        
    return maxLikelihoodTDOA
