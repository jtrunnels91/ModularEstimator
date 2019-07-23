## @file SignalSource.py holds the SignalSource base class
#
from abc import ABCMeta, abstractmethod
from scipy.stats import multivariate_normal as mvn
from scipy.linalg import block_diag
import numpy as np


class SignalSource():
    __metaclass__ = ABCMeta
    nextSignalID = 0
    
    def __init__(
            self,
    ):
        self.__signalID__ = SignalSource.nextSignalID
        SignalSource.nextSignalID += 1
        return

    def signalID(self):
        return self.__signalID__

    @abstractmethod
    def computeAssociationProbability(
            self,
            measurement,
            stateDict,
            validationThreshold=0
    ):
        
        # myMeasMat = substate.getMeasurementMatrices(measurement, source=self)
        # dY = None
        # R = None
        # H = None
        # for key in myMeasMat['dY']:
        #     if H is None:
        #         H = myMeasMat['H'][key]
        #         R = myMeasMat['R'][key]
        #         dY = myMeasMat['dY'][key]
        #     else:
        #         H = np.vstack([H, myMeasMat['H'][key]])
        #         R = block_diag(R, myMeasMat['R'][key])
        #         dY = np.append(dY, myMeasMat['dY'][key])

        # if dY is not None:
        #     P = substate.covariance()
        #     Pval = P.convertCovariance('covariance').value
        #     S = H.dot(Pval).dot(H.transpose()) + R

        #     myProbability = mvn.pdf(dY, cov=S)
        # else:
        #     myProbability = 0
        # return myProbability
        pass
