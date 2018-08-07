## @module SubState
# This file contains the SubState class.


from abc import ABCMeta, abstractmethod
import numpy as np
# import matplotlib as mp
import matplotlib.pyplot as plt
from smartpanda import SmartPanda

## @class SubState
# @brief This is an abstract base class for objects used as sub-states in
# State.ModularFilter.
#
# @details SubState is an abstract base class that specifies the methods which
# are required for an object to function as a sub-state of State.ModularFilter.
#
# Some of these methods are implemented and most likely do not need to be
# reimplemented in a derived class implementation (for example the #dimension
# and #covariance methods.
#
# Other methods may have a rudimentary implementation that may be suitable for
# some derived classes, but not others, depending on the specific
# functionality of the derived class (for instance #getStateVector and
# #storeStateVector).
#
# Finally, some methods are specifically tagged as abstract methods and are
# not implemented at all.  These methods must be implemented in the derived
# class.  This is usually because there is no way to implement even a
# rudimentary version of what the method is supposed to do without having some
# knowledge of what kind of substate the derived class contains (for instance
# #timeUpdate and #getMeasurementMatrices).
#
# In any case, the documentation for each method of SubState contains a
# generalized description of what functionality the implementation should
# provide in a derived class.
class SubState():
    __metaclass__ = ABCMeta

    ## @fun #__init__ initializes a SubState object
    #
    # @details The #__init__ method is responsible for initializing a
    # generalized SubState object.  The essential functions of #__init__ are
    # to store the dimension of the state, and to initialize a time-history of
    # the state in a SmartPanda object.
    #
    # If no values are passed for the initial state estimate dictionary, they
    # will be initialized to the following default values.
    #
    # - 'stateVector': A length #dimension array of zeros
    # - 'covariance': An (#dimension x #dimension) identity matrix
    # - 't': 0
    #
    # @param self The object pointer
    # @param stateDimension The dimension of the sub-state state vector
    # @param stateVectorHistory A dictionary containing the initial state.
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

        if stateVectorHistory is None:
            stateVectorHistory = {
                't': 0,
                'stateVector': np.zeros(stateDimension),
                'covariance': np.eye(stateDimension),
                'stateVectorID': 0
                }
                
        # Check to verify that the dictionary contains the correct keys
        if 't' not in stateVectorHistory:
            raise ValueError(
                "State vector history must contain time key, labeled \"t\""
                )
        if 'stateVector' not in stateVectorHistory:
            raise ValueError(
                "State vector history must contain state vector key, labeled" +
                "\"stateVector\""
                )

        if 'stateVectorID' not in stateVectorHistory:
            raise ValueError(
                "State vector history must contain state vector id key, labeled \"stateVectorID\""
                )
            
        
        ## @brief Stores the time-history of the sub-state state vector.
        self.stateVectorHistory = SmartPanda(data=stateVectorHistory)

        ## @brief Stores handle for real-time plotting        
        self.RTPlotHandle = None

        self.RTPlotData = None

        return

    ##
    # @name Mandatory SubState Functions
    # The following functions are functions which are required for the
    # SubState to function as a sub-state in State.ModularFilter.
    # @{
    
    ## @fun #getStateVector returns the most recent value of the state vector
    #
    # @details
    # The #getStateVector method is responsible for returning a dictionary
    # object containing, at minimim, the following items:
    #
    # - 'stateVector': A length #dimension array containing the most recent
    # state vector estimate
    # - 'covariance': A (#dimension x #dimension) array containing the most
    # recent covariance matrix
    # - 'aPriori': A boolean indicating if the most recent estimate is the
    # - result of a time update (aPriori=True) or a measurement update (aPriori=False)
    #
    # This function can be used as-is, or can be overloaded to perform
    # additional tasks specific to the substate.
    #
    # @param self The object pointer
    #
    # @returns The dictionary containing the state vector, covariance matrix,
    # and aPriori status
    def getStateVector(self):
        lastSV = self.stateVectorHistory.getDict(-1)

        return(lastSV)

    ## @fun #storeStateVector stores the most recent value of the state vector.
    #
    # @details
    # The #storeStateVector method is responsible for storing a dictionary
    # containing the most recent state estimate.  In SubState implementation,
    # the functionality is minimal: the new dictionary is simply appeneded to
    # the SmartPanda list of state vector estimates.  However, in some derived
    # classes, it may be nescessary to implement additional functionality.
    # This is particularly true if there are derived quantities that need to
    # be calculated from the updated state vector (for instance, calculating
    # the attitude quaternion from the attitude error states).  Also in some
    # cases, the actual value of the state vector may need to be "tweaked" by
    # the SubState derived class.
    #
    # If an alternative implementation is written for a derived class, it
    # should still call this implementation, or at least make sure that it
    # stores the current state estimate in #stateVectorHistory.
    #
    # @param self The object pointer
    # @param svDict A dictionary containing the current state estimate.
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
        return self.stateVectorHistory.getDict(-1)['covariance']

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

    ## @fun #timeUpdate returns time-update matrices
    #
    # @details The #timeUpdate method is responsible for returning the EKF
    # time update measurement matrices.  Specifically, it returns the state
    # update matrix \f$\mathbf{F}\f$ and the process noise matrix
    # \f$\mathbf{Q}\f$, following the standard
    # <a href="https://en.wikipedia.org/wiki/Extended_Kalman_filter">
    # Extended Kalman Filter</a> time update equations:
    #
    # \f[\sv[timeIndex=k+1, aPriori=True] =
    # \mathbf{F} \sv[timeIndex=k, aPriori=False] \f]
    # \f[\svVar[timeIndex=k+1, aPriori=True] =
    # \mathbf{F} \svVar[timeIndex=k+1, aPriori=True] \mathbf{F}^{T}  + \mathbf{Q} \f]
    #
    # Because these matrices are nescessarily specific to the type of substate
    # being updated, there is no default implementation in the SubState class.
    # Rather, each derived class must implement this method as appropriate for
    # the dynamics of the state being modeled.
    #
    # In addition, some substates may require additional operations to occur
    # at a time update.  For instance, if a substate includes auxillary values
    # (for instance, the attitude quaternion derived from the attitude error
    # state), it may need to be time-updated seperately from the other states.
    # In this case, the local implementation of the #timeUpdate function is
    # the place to do these updates.
    #
    # @param self The object pointer
    # @param dT The ellapsed time over which the time update occurs
    # @param dynamics A dictionary containing any dynamics infomation which
    # may be needed to update the state, for instance, measured accelerations
    # or angular velocities.
    #
    # @returns A dictionary containing, at minimum, the following items:
    # - "F": The state time-update matrix
    # - "Q": The process noise matrix
    @abstractmethod
    def timeUpdate(self, dT, dynamics=None):
        pass

    @abstractmethod
    def getMeasurementMatrices(self, measurement, source=None):
        pass
    ## @}


    """
    Plotting Functions
    """
    ##
    # @name Plotting Functions
    # These functions provide generalized plotting capabilities
    # @{

    def initializeRealTimePlot(
            self,
            plotHandle=None,
            axisHandle=None
            ):

        if plotHandle is None:
            self.RTPlotHandle = plt.figure()
        else:
            self.RTPlotHandle = plotHandle

        if axisHandle is None:
            self.RTPaxisHandle = plt.gca()
        else:
            self.RTPaxisHandle = axisHandle
            
        xAxis = np.linspace(0, self.__dimension__ - 1, self.__dimension__)

        self.RTPlotData, = plt.plot(
            xAxis,
            np.zeros(self.__dimension__)
            )

        # plt.grid()
        
        plt.show(block=False)
        return
        
    def realTimePlot(
            self,
            normalized=True
    ):
        if self.RTPlotHandle is None:
            self.initializeRealTimePlot()

        stateDict = self.getStateVector()
        yAxis = stateDict['stateVector']

        if normalized is True:
            self.RTPaxisHandle.set_ylim([0, 1.1])
            yAxis = yAxis - np.min(yAxis)
            yAxis = yAxis/np.max(yAxis)

        if 'xAxis' in stateDict:
            xAxis = stateDict['xAxis']
            self.RTPaxisHandle.set_xlim([np.min(xAxis), np.max(xAxis)])
        else:
            xAxis = np.linspace(0, self.__dimension__ - 1, self.__dimension__)

        self.RTPlotData.set_data(xAxis, yAxis)
        self.RTPlotHandle.canvas.draw()
        self.RTPlotHandle.canvas.flush_events()

        return

    # @}
