## @file SignalSource.py holds the SignalSource base class
#

class SignalSource():
    nextSignalID = 0
    
    def __init__(
            self,
    ):
        self.__signalID__ = SignalSource.nextSignalID
        SignalSource.nextSignalID += 1
        return

    def signalID(self):
        return self.__signalID__
