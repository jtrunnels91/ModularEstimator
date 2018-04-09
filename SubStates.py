## @package SubStates
# This package contains the SubState class.

from abc import ABC, abstractmethod

## @class SubState
# @brief This is an abstract base class for objects used as sub-states in
# State.ModularFilter.
#
# @details SubState is an abstract base class that specifies the methods which
# are required for an object to function as a sub-state of State.ModularFilter.
#
# Some of these methods are implemented and most likely do not need to be
# reimplemented in a derived class implementation (for example the #dimension and covariance methods.
#
# Other methods may have a rudimentary implementation that may be suitable for some derived classes, but not others, depending on the specific functionality of the derived class (for instance getStateVector and storeStateVector).
#
# Finally, some methods are specifically tagged as abstract methods and are not implemented at all.  These methods must be implemented in the derived class.  This is usually because there is no way to implement even a rudimentary version of what the method is supposed to do without having some knowledge of what kind of substate the derived class contains (for instance timeUpdate and getMeasurementMatrices).
#
# In any case, the documentation for each method of SubState contains a generalized description of what functionality the implementation should provide in a derived class.
class SubState(ABC):
    
    ## @fun #__init__ initializes a SubState object
    # 
    def __init__(
            self,
            stateDimension=None,
            stateVectorHistory=None
    ):
        if stateDimension is None:
            raise ValueError(
                "Must pass a state dimension."
                )
        
        ## @brief Stores the length of the state vector as seen by
        # the ModularFilter.  See the #dimension function for details on
        # implementation.
        self.__dimension__ = stateDimension

        
        ## @brief Stores the time-history of the sub-state state vector.
        self.stateVectorHistory = stateVectorHistory
        return
    
    @abstractmethod
    def getStateVector(self):
        pass

    def storeStateVector(self, svDict):
        self.stateVectorHistory.append(svDict)
        return
    
    ## @fun #covariance returns the SubState covariance matrix
    #
    # @details
    # The #covariance method returns the covariance of the estimate of the
    # substate.
    #
    # @todo Currently, this method only returns the covariance of the most
    # recent state estimate.  Ideally, there should be an optional time
    # parameter which would allow the user to get the covaraince matrix at a
    # specified time (or the closest to that specified time).
    #
    # @param self The object pointer
    #
    # @return Returns the covaraince matrix
    def covariance(self):
        return self.stateVectorHistory[-1]['covariance']

    ## @fun #dimension returns the dimension of the sub-state vector
    #
    # @details The #dimension method returns the dimension of the sub-state
    # vector estimated by the SubState.  This is the dimension as seen by the
    # ModularFilter estimator.
    #
    # The default implementation is to return the class variable
    # #__dimension__, which is saved at initialization.  This is designated as
    # a "protected" variable, and should not change during the course of the
    # SubState's lifetime.  If child class overwrites this implementation,
    # care should be taken to ensure that the value returned by #dimension
    # does not change over SubState object lifetime.
    #
    # For SubState objects with auxilary states, or other quantities related
    # to the state vector but not directly estimated by the ModularFilter,
    # #dimension should not count these states as part of the total dimension.
    #
    # @param self The object pointer
    #
    # @return Returns the dimension of state vector
    def dimension(
            self
            ):
        return(self.__dimension__)

    @abstractmethod
    def timeUpdate(self, dT, dynamics=None):
        pass

    @abstractmethod
    def getMeasurementMatrices(self, measurement, source=None):
        pass
