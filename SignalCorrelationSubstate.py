import numpy as np
from scipy.linalg import block_diag
from numpy import sin, cos, arcsin, arccos, arctan2, square, sqrt, abs, power
from numpy.linalg import norm
from pyquaternion import Quaternion

import sys
import os
sys.path.append("/home/joel/Documents/astroSourceTracking/libraries")
from SpaceGeometry import *
from Signals import PointSource
from SmartPanda import SmartPanda


class CorrelationFilter():
    def __init__(
            self,
            signal,
            filterOrder,
            dT,
            correlationVector,
            correlationVectorCovariance,
            signalDelay=0,
            delayVar=0,
            centerPeak=True
            ):

        # Store the signal to be tracked
        self.signal = signal

        # Store information about the correlation vector
        self.filterOrder = filterOrder
        self.dT = dT
        
        self.cHat = correlationVector
        self.PHat = correlationVectorCovariance

        self.delay = signalDelay
        self.delayVar = delayVar

        self.centerPeak = True

        return

    ###########################################################################
    # Functions Required To Function as a "Substate"                          #
    #                                                                         #
    # The following functions are required in order for this class to be used #
    # as a substate in ModularFilter.  The inside of the functions may be     #
    # changed or updated, but their "black box" behavior must remain the      #
    # same; i.e. they must still perform the same essential functions and     #
    # return the same things.                                                 #
    ###########################################################################

    def dimension(self):
        return(self.filterOrder)

    def getStateVector(self):
        return

    def storeStateVector(
            self,
            deltaXPlus,
            PPlus,
            time,
            aPriori=False
            ):
        return

    def covariance(
            self
            ):
        return

    def timeUpdate(
            self,
            dT,
            dynamics=None
            ):
        return

    def getMeasurementMatrices(
            self,
            measurement,
            source=None
            ):
        return
    
    
    ###########################################################################
    # Functions Specific to This Class                                        #
    #                                                                         #
    # The following remaining functions are not required in order for this    #
    # class to be used as a SubState, and may be changed as needed,           #
    # including inputs and outputs.                                           #
    ###########################################################################
