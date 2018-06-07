import matplotlib.pyplot as plt

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
            print(p['associationProbabilities'][probabilityAlpha])
            if color is None:
                point = axis.scatter(
                    p['RA']['value'], p['DEC']['value'],
                    marker = '.',
                    size=size,
                    alpha=alpha * p['associationProbabilities'][probabilityAlpha]
                    )
                color = point[-1].get_color()
            else:
                point = axis.scatter(
                    p['RA']['value'], p['DEC']['value'],
                    marker='.',
                    size=size,
                    alpha= (alpha * p['associationProbabilities'][probabilityAlpha]),
                    color=color
                    )
                
                

    
