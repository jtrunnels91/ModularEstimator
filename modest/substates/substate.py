## @module SubState
# This file contains the SubState class.


from abc import ABCMeta, abstractmethod
import numpy as np
# import matplotlib as mp
import matplotlib.pyplot as plt

class SubState():
    r"""
    This is an abstract base class for objects used as sub-states in
    State.ModularFilter.

    SubState is an abstract base class that specifies the methods which are required for an object to function as a sub-state of State.ModularFilter.

    Some of these methods are implemented and most likely do not need to be
    reimplemented in a derived class implementation (for example the #dimension
    and #covariance methods.

    Other methods may have a rudimentary implementation that may be suitable for
    some derived classes, but not others, depending on the specific
    functionality of the derived class (for instance #getStateVector and
    #storeStateVector).

    Finally, some methods are specifically tagged as abstract methods and are
    not implemented at all.  These methods must be implemented in the derived
    class.  This is usually because there is no way to implement even a
    rudimentary version of what the method is supposed to do without having some
    knowledge of what kind of substate the derived class contains (for instance
    :meth:`timeUpdate` and :meth:`getMeasurementMatrices`).

    In any case, the documentation for each method of SubState contains a
    generalized description of what functionality the implementation should
    provide in a derived class.
    """
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
            stateVectorHistory=None,
            storeLastStateVectors=0,
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
        # self.stateVectorHistory = SmartPanda(data=stateVectorHistory)
        self.stateVectorHistory = [stateVectorHistory]
        self.timeList = [stateVectorHistory['t']]

        ## @brief Stores handle for real-time plotting        
        self.RTPlotHandle = None
        self.THPlotHandle = None

        self.RTPlotData = None

        self.storeLastStateVectors = storeLastStateVectors
        """ 
        (int) Determines how far back the state vector history may go.  If zero, then the entire state vector history is stored
        """

        return

    ##
    # @name Mandatory SubState Functions
    # The following functions are functions which are required for the
    # SubState to function as a sub-state in State.ModularFilter.
    # @{
    
    def getStateVector(self, t=None):
        r"""
        getStateVector returns the most recent value of the state vector
    
        The getStateVector method is responsible for returning a dictionary object containing, at minimim, the following items:
    
        - 'stateVector': A length :attr:`dimension` array containing the most recent state vector estimate
        - 'covariance': A (:attr:`dimension` x :attr:`dimension`) array containing the most recent covariance matrix
        - 'aPriori': A boolean indicating if the most recent estimate is the
        - result of a time update (aPriori=True) or a measurement update (aPriori=False)
    
        This function can be used as-is, or can be overloaded to perform additional tasks specific to the substate.
        
        Args:
         t (int): This is an optional argument if the user wants a state vector from a time other than the current time.
    
        Returns:
         dict: A dictionary containing the state vector, covariance matrix, and aPriori status
        """
        # lastSV = self.stateVectorHistory.getDict(-1)
        if t is None:
            stateVector = self.stateVectorHistory[-1]
        else:
            timeIndex = np.searchsorted(self.timeList, t)
            stateVector = self.stateVectorHistory[timeIndex]

        return(stateVector)

    def storeStateVector(self, svDict):
        r"""
        storeStateVector stores the most recent value of the state vector.
    
        The storeStateVector method is responsible for storing a dictionary
        containing the most recent state estimate.  In SubState implementation,
        the functionality is minimal: the new dictionary is simply appeneded to
        the list of state vector estimates.  However, in some derived
        classes, it may be nescessary to implement additional functionality.
        This is particularly true if there are derived quantities that need to
        be calculated from the updated state vector (for instance, calculating
        the attitude quaternion from the attitude error states).  Also in some
        cases, the actual value of the state vector may need to be "tweaked" by
        the SubState derived class.
        
        If an alternative implementation is written for a derived class, it
        should still call this implementation, or at least make sure that it
        stores the current state estimate in :attr:`stateVectorHistory`.
        
        Args:
         svDict (dict):  A dictionary containing the current state estimate.
        """
        self.stateVectorHistory.append(svDict)

        # Check to see if state vector history is too long; if so truncate to
        # the last storeLastStateVectors values
        if self.storeLastStateVectors > 0:
            if len(self.stateVectorHistory) > self.storeLastStateVectors:
                self.stateVectorHistory = self.stateVectorHistory[
                    len(self.stateVectorHistory)-self.storeLastStateVectors:
                ]
        self.timeList.append(svDict['t'])
        return
    
    def covariance(self):
        r"""
        :meth:`covariance` returns the SubState covariance matrix
    
        The covariance method returns the covariance of the estimate of the
        substate.
    
        todo: Currently, this method only returns the covariance of the most
        recent state estimate.  Ideally, there should be an optional time
        parameter which would allow the user to get the covaraince matrix at a
        specified time (or the closest to that specified time).
        
        Returns:
         covarianceContainer: Returns the covaraince matrix
        """
        # return self.stateVectorHistory.getDict(-1)['covariance']
        return self.stateVectorHistory[-1]['covariance']

    def dimension(
            self
            ):
        r"""
        dimension returns the dimension of the sub-state vector

        The dimension method returns the dimension of the sub-state
        vector estimated by the SubState.  This is the dimension as seen by the
        ModularFilter estimator.

        The default implementation is to return the class variable
        :attr:`__dimension__`, which is saved at initialization.  This is designated as
        a "protected" variable, and should not change during the course of the
        :class:`SubState`'s lifetime.  If child class overwrites this implementation,
        care should be taken to ensure that the value returned by #dimension
        does not change over SubState object lifetime.

        For SubState objects with auxilary states, or other quantities related
        to the state vector but not directly estimated by the ModularFilter,
        #dimension should not count these states as part of the total dimension.

        Returns:
         int: The dimension of state vector
        """
        return(self.__dimension__)

    @abstractmethod
    def timeUpdate(self, dT, dynamics=None):
        r"""
        timeUpdate returns time-update matrices

        The :meth:`timeUpdate` method is responsible for returning the EKF
        time update matrices.  Specifically, it returns the state
        update matrix :math:`\mathbf{F}` and the process noise matrix
        :math:`\mathbf{Q}`, following the standard
        `Extended Kalman Filter <https://en.wikipedia.org/wiki/Extended_Kalman_filter>`_
        time update equations:
        
        .. math::
            \mathbf{x}_k^- = \mathbf{F}\mathbf{x}_j^+
        .. math::
            \mathbf{P}_k^- = \mathbf{F} \mathbf{P}_k^- \mathbf{F}^T + \mathbf{Q}
         

        Because these matrices are nescessarily specific to the type of substate
        being updated, there is no default implementation in the SubState class.
        Rather, each derived class must implement this method as appropriate for
        the dynamics of the state being modeled.

        In addition, some substates may require additional operations to occur
        at a time update.  For instance, if a substate includes auxillary values
        (for instance, the attitude quaternion derived from the attitude error
        state), it may need to be time-updated seperately from the other states.
        In this case, the local implementation of the #timeUpdate function is
        the place to do these updates.

        Args:
         dT (float): The ellapsed time over which the time update occurs
         dynamics (dict): A dictionary containing any dynamics infomation which may be needed to update the state, for instance, measured accelerations or angular velocities.

        Returns: 
         (dict)
         A dictionary containing, at minimum, the following items:
          - "F": The state time-update matrix
          - "Q": The process noise matrix
        """
        
        pass

    @abstractmethod
    def getMeasurementMatrices(self, measurement, source=None):
        r"""
        getMeasurementMatrices returns matrices needed to perform a measurement update

        The :meth:`getMeasurementMatrices` method is responsible for returning
        the EKF measurement update matrices.  Specifically, it returns the
        measurement matrix :math:`\mathbf{H}`, the measurement noise matrix
        :math:`\mathbf{R}`, and the measurement residual vector
        :math:`\mathbf{\delta}\mathbf{y}`, folloing the standard 
        `Extended Kalman Filter <https://en.wikipedia.org/wiki/Extended_Kalman_filter>`_ 
        measurement update equations:
        
        .. math::
            \mathbf{\delta y} = \mathbf{y} - h(\mathbf{x}_k^-, \mathbf{w}_k)

        .. math::

            \mathbf{H}_k^-= \frac{\partial h}{ \partial \mathbf{x}}

        .. math::

            \mathbf{S}_k^- = \mathbf{H}_k \mathbf{P}_k^- \mathbf{H}_k^T + \mathbf{R}

        .. math::

            \mathbf{K}_k^- = \mathbf{P}_k^- \mathbf{H}_k^T \mathbf{S}_k^{-1} 

        Because these matrices are nescessarily specific to the type of substate
        being updated, there is no default implementation in the SubState class.
        Rather, each derived class must implement this method as appropriate for
        the measurement of the state being modeled.  Additionally, the
        measurement update may be different depending on which signal source
        is generating the measurement, so the substate must have measurement
        matrix generation implemented for all the expected signal sources.

        In addition, some substates may require additional operations to occur
        at a measurement update.  For instance, if a substate includes
        auxillary values (for instance, the attitude quaternion derived from
        the attitude error state), it may need to be updated seperately after
        the global state vector has been updated. In this case, the local
        implementation of the :meth:`timeUpdate` function is the place to do
        these updates.

        Note that there is no time associated with the measurement; the filter
        assumes that the measurement is occuring at the current time. 
        Therefore it is the user's responsibility to time-update the state to
        the current time before doing the measurement update.

        Args:
         measurement (dict): A dictionary containing the measurement value(s)
         source (str): A key uniquely identifying the source of origin of the measurement

        Returns: 
         (dict)
         A dictionary containing, at minimum, the following items:
          - "F": The state time-update matrix
          - "Q": The process noise matrix
        """
        
        pass


    """
    Plotting Functions
    """
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
            try:
                yAxis = yAxis/np.max(yAxis)
            except:
                pass

        if 'xAxis' in stateDict:
            xAxis = stateDict['xAxis']
            self.RTPaxisHandle.set_xlim([np.min(xAxis), np.max(xAxis)])
        else:
            xAxis = np.linspace(0, self.__dimension__ - 1, self.__dimension__)

        self.RTPlotData.set_data(xAxis, yAxis)
        self.RTPlotHandle.canvas.draw()
        self.RTPlotHandle.canvas.flush_events()

        return


    def timeHistoryPlot(
            self
    ):
        if self.THPlotHandle is None:
            self.initializeTimeHistoryPlot()
            
        for stateCounter in range(self.dimension()):
            self.THPlotDataList[stateCounter]['x'].append(self.stateVectorHistory[-1]['t'])
            self.THPlotDataList[stateCounter]['y'].append(
                self.stateVectorHistory[-1]['stateVector'][stateCounter]
            )
            
            self.THPlotObjectList[stateCounter].set_data(
                self.THPlotDataList[stateCounter]['x'],
                self.THPlotDataList[stateCounter]['x']
            )
            
        self.THPlotHandle.canvas.draw()
        self.THPlotHandle.canvas.flush_events()
        
        return
    def initializeTimeHistoryPlot(
            self,
            plotHandle=None
    ):
        if plotHandle is None:
            self.THPlotHandle = plt.figure()
        else:
            self.THPlotHandle = plotHandle
        # if axisHandle is None:
        #     self.THPaxisHandle = plt.gca()
        # else:
        #     self.THPaxisHandle = axisHandle

        self.THPaxisList = []
        self.THPlotDataList = []
        self.THPlotObjectList = []
        for stateCounter in range(self.dimension()):
            newAxis = plt.subplot2grid(
                (self.dimension(), 1), (stateCounter, 0)
            )
            self.THPaxisList.append(newAxis)
            newAxisX = [self.stateVectorHistory[0]['t'], self.stateVectorHistory[-1]['t']]
            newAxisY = [
                self.stateVectorHistory[0]['stateVector'][stateCounter],
                self.stateVectorHistory[-1]['stateVector'][stateCounter]
            ]
            newPlot, = plt.plot(newAxisX, newAxisY)
            self.THPlotObjectList.append(newPlot)
            self.THPlotDataList.append(
                {
                    'x': newAxisX,
                    'y': newAxisY
                }
            )
                
        plt.tight_layout()
        
        plt.show(block=False)
        return
        
        
