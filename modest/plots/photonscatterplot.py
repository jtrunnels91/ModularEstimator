import matplotlib.pyplot as plt
import numpy as np

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
    
