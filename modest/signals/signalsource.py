## @file SignalSource.py holds the SignalSource base class
#
from abc import ABCMeta, abstractmethod


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
        pass
