from . signalsource import SignalSource
from .. utils import covarianceContainer
from scipy.linalg import block_diag
import numpy as np

class oneDObjectMeasurement(SignalSource):
    def __init__(self, objectID):
        self.objectID = objectID
        return

    def computeAssociationProbability(self, measurement, stateDict, validationThreshold=0):
        myMeasMat = stateDict[self.objectID]['stateObject'].getMeasurementMatrices(measurement, source=self)
        dY = None
        R = None
        H = None
        for key in myMeasMat['dY']:
            if H is None:
                H = myMeasMat['H'][key]
                R = myMeasMat['R'][key]
                dY = myMeasMat['dY'][key]
            else:
                H = np.vstack([H, myMeasMat['H'][key]])
                R = block_diag(R, myMeasMat['R'][key])
                dY = np.append(dY, myMeasMat['dY'][key])

        if dY is not None:
            P = stateDict[self.objectID]['stateObject'].covariance()
            Pval = P.convertCovariance('covariance').value
            # if P.form == 'cholesky':
            #     Pval = P.value.dot(P.value.transpose())
            # elif P.form == 'covariance':
            #     Pval = P.value
            # else:
            #     raise ValueError('Unrecougnized covariance specifier %s' %P.form)
            S = H.dot(Pval).dot(H.transpose()) + R

            myProbability = mvn.pdf(dY, cov=S)
        else:
            myProbability = 0
        return myProbability
