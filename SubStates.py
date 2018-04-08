from abc import ABCMeta, abstractmethod

## @class SubState is designed as a parent-class for 
#
class SubState():
    
    __metaclass__ = ABCMeta
    
    def __init__(
            self,
            stateDimension=None,
            stateVectorHistory=None
    ):
        if stateDimension is None:
            raise ValueError(
                "Must pass a state dimension."
                )
        self.__dimension__ = stateDimension
        self.stateVectorHistory = stateVectorHistory
        return
    
    ## @fun dimension returns the dimension of the sub-state vector
    #
    # @param self The object pointer
    #
    # @return Returns the dimension of state vector
    def dimension(
            self
            ):
        return(self.__dimension__)

    @abstractmethod
    def timeUpdate():
        pass
