## @package State
# This package contains the ModularFilter class.
# @author Joel Runnels
# @date 2018
# @copyright GNU General Public License
#
# @note
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see
# <a href="http://www.gnu.org/licenses/">GNU GPL</a>.

import numpy as np
from scipy.linalg import block_diag
from numpy import sin, cos, arcsin, arccos, arctan2, square, sqrt, abs, power
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
# from pyquaternion import Quaternion

import sys
import os
sys.path.append("/home/joel/Documents/astroSourceTracking/libraries")


## @class ModularFilter
# @details
# This class is designed to facilitate a variety of estimation algorithms in a
# modular way, i.e. in a way that allows maximum amount of flexibility with
# minimum amount of code duplication.
#
# The idea behind this class is that all of the functions which are "generic"
# to an estimation algorithm can be written just once, as a part of this class.
# Then the code contained here can be implemented on a variety of different
# estimation problems without having to duplicate code.
#
# The basic estimation model implemented here works as follows.  The overall
# state, i.e. all of the quantities to be estimated, are represented by
# SubStates.SubState objects. Measurements of these states are represented by
# Signals.SignalSource objects.  The ModularFilter class is responsible for doing time-updates and measurement-updates on these states.
# These objects are responsible for doing all
# of the things that are particular to those states, and those signals,
# respectively.  For instance, generation of time-update and measurement
# update matrices is handled by the SubStates.SubState objects.
class ModularFilter():
    def __init__(
            self,
            measurementValidationThreshold=1e-3,
            time=0
    ):
        self.plotHandle=None

        self.totalDimension = 0
        self.covarianceMatrix = np.zeros([0, 0])

        self.subStates = {}
        self.signalSources = {}
        self.tCurrent = time

        self.measurementValidationThreshold = measurementValidationThreshold

        self.measurementList = []
        self.lastMeasurementID = None

        self.lastStateVectorID = 0
        
        return


    """
    addStates is a utility function used to add a state to the joint estimator.
    The following inputs are required:
    -name: The name by which the state can be referenced (must be unique)
    -stateObject: An object that contains the sub-state and is responsible for 
    performing various tasks related to the substate.
    """
    def addStates(
            self,
            name,
            stateObject
    ):

        # Check to see whether the name is unique
        if name in self.subStates:
            raise ValueError(
                'The name "%s" has already been used for a state.  If you ' +
                'want to remove that state, you can use the removeStates ' +
                'function.  If you want to replace it, you can use the ' +
                'replaceStates function.' %name
                )

        newSlice = slice(self.totalDimension, self.totalDimension + stateObject.dimension())
        
        self.totalDimension = (
            self.totalDimension + stateObject.dimension()
            )

        self.covarianceMatrix = block_diag(
            self.covarianceMatrix,
            stateObject.covariance()
        )

        self.subStates[name] = {
            'index': newSlice,
            'length': stateObject.dimension(),
            'stateObject': stateObject
            }
        return

    """
    addSignalSource is a utility function used to add a signal source to the 
    joint estimator.

    The following inputs are required:
    -name: The name by which the signal source can be referenced (must be 
    unique)
    -signalSourceObject: An object that contains the signal source and is
    responsible for performing various tasks related to the substate.
    """
    def addSignalSource(
            self,
            name,
            signalSourceObject
    ):
        # Check to see whether the name is unique
        if name in self.signalSources:
            raise ValueError(
                'The name "%s" has already been used for a signal source.  ' +
                'If you want to remove that signal source, you can use the ' +
                'removeSignalSource function.  If you want to replace it, ' +
                'you can use the replaceSignalSource function.'
                )

        self.signalSources[name] = signalSourceObject
        
        return

    def timeUpdateEKF(
            self,
            dT,
            dynamics=None
            ):

        F = np.zeros([self.totalDimension, self.totalDimension])
        Q = np.zeros([self.totalDimension, self.totalDimension])

        # Assemble time-update matrix and process noise matrix based on
        # dynamics.
        for stateName in self.subStates:
            timeUpdateMatrices = (
                self.subStates[stateName]['stateObject'].timeUpdate(
                    dT,
                    dynamics=dynamics
                )
                )
            
            mySlice = self.subStates[stateName]['index']

            F[mySlice, mySlice] = timeUpdateMatrices['F']
            Q[mySlice, mySlice] = timeUpdateMatrices['Q']

        xMinus = F.dot(self.getGlobalStateVector())
        
        PMinus = F.dot(self.covarianceMatrix).dot(F.transpose()) + Q

        self.tCurrent = self.tCurrent + dT
        
        self.covarianceMatrix = PMinus
        
        self.storeGlobalStateVector(xMinus, PMinus, aPriori=True)
        
        return (xMinus, PMinus)
    
    """
    computeAssociationProbabilities is responsible for computing the
    probability that a measurement originated from each of the signal sources
    being tracked by the estimator.  The function assumes that each signal
    source object has a member function which will compute the probability of
    origination, given the measurement and the prior state values.
    Additionally, it assumes that the objects take a validation threshold as
    an optional argument.

    The following inputs are required:
    - measurement: The measurement for which the probability is to be computed.
    Most likely will be a dict, with a time key and a value key.
    """
    def computeAssociationProbabilities(
            self,
            measurement
    ):

        probabilityDict = {}
        probabilitySum = 0
        for signalKey in self.signalSources:
            currentProbability = (
                self.signalSources[signalKey].computeAssociationProbability(
                    measurement,
                    self.subStates,
                    validationThreshold=self.measurementValidationThreshold
                )
            )

            if currentProbability < 0:
                raise ValueError(
                    'Got negative probability for measurement %s, signal source %s'
                    %(measurement, signalKey)
                    )
                                 
            probabilityDict[signalKey] = currentProbability
            probabilitySum = probabilitySum + currentProbability

        for probabilityKey in probabilityDict:
            probabilityDict[probabilityKey] = (
                probabilityDict[probabilityKey] / probabilitySum
                )
        # print(probabilityDict)
        return (probabilityDict)

    """
    measurementUpdateEKF is a standard Extended Kalman Filter measurement
    update.  This function only works when the signal source is known, which
    may or may not be a realistic assumption, depending on the problem.
    Alternatively, it can be used for comparison to other data association
    methods.

    The following inputs are required:
    - measurement: A dict with all the relevant measurable quanties labeled in
    a way that the substates and signal source objects understand.
    - sourceName: A string that contains the name of the signal source from
    which the measurement originated.  This must be the name of one of the
    signal sources that have been added to the estimator.
    """    
    def measurementUpdateEKF(
            self,
            measurement,
            sourceName
            ):

        if 'ID' not in measurement:
            if self.lastMeasurementID is None:
                self.lastMeasurementID = -1
            measurement['ID'] = self.lastMeasurementID + 1

        if measurement['ID'] in self.measurementList:
            raise Warning(
                'Measurement ID %s has already been used to update state.'
                % measurement['ID']
            )
        else:
            self.measurementList.append(measurement['ID'])
        self.lastMeasurementID = measurement['ID']
        
        xMinus = self.getGlobalStateVector()
        PMinus = self.covarianceMatrix

        updateDict = self.localStateUpdateMatrices(
            measurement,
            sourceName,
            xMinus,
            PMinus
            )

        xPlus = updateDict['xPlus']
        PPlus = updateDict['PPlus']
        
        self.covarianceMatrix = PPlus
        self.storeGlobalStateVector(xPlus, PPlus, aPriori=False)
        return (xPlus, PPlus)

    def measurementUpdateML(
            self,
            measurement
            ):
        signalAssociationProbability = (
            self.computeAssociationProbabilities(measurement)
            )

        del signalAssociationProbability['noise']

        maxLikelihoodSignal = max(
            signalAssociationProbability,
            key=signalAssociationProbability.get
        )

        xMinus = self.getGlobalStateVector()
        PMinus = self.covarianceMatrix

        if (
                signalAssociationProbability[maxLikelihoodSignal] >
                self.measurementValidationThreshold
                ):
                
            updateDict = self.localStateUpdateMatrices(
                measurement,
                maxLikelihoodSignal,
                xMinus,
                PMinus
                )

            xPlus = updateDict['xPlus']
            PPlus = updateDict['PPlus']
        else:
            xPlus = xMinus
            PPlus = PMinus
            
        self.covarianceMatrix = PPlus
        self.storeGlobalStateVector(xPlus, PPlus, aPriori=False)
        return (xPlus, PPlus)
        
        return
        
    def measurementUpdateJPDAF(
            self,
            measurement
    ):
        print('Called JPDAF')
        signalAssociationProbability = (
            self.computeAssociationProbabilities(measurement)
            )

        measurement['associationProbabilities']=signalAssociationProbability

        #print(signalAssociationProbability)
        xMinus = self.getGlobalStateVector()
        PMinus = self.covarianceMatrix

        xPlus = np.zeros(self.totalDimension)
        PPlus = np.zeros([self.totalDimension, self.totalDimension])

        validAssociationsDict = {}
        for signalName in signalAssociationProbability:
            currentPR = signalAssociationProbability[signalName]

            if (
                    currentPR >
                    self.measurementValidationThreshold
            ):
                updateDict = self.localStateUpdateMatrices(
                    measurement,
                    signalName,
                    xMinus,
                    PMinus
                    )

                xPlus = (
                    xPlus + (currentPR * updateDict['xPlus'])
                    )

                PPlus = (
                    PPlus + (currentPR * updateDict['PPlus'])
                )

                # If the signal association was valid, store it in a dict so
                # that we can go back and compute the spread of means term
                validAssociationsDict[signalName] = updateDict

        try:
            np.linalg.cholesky(PPlus)
        except:
            print('PPlus = %s' %PPlus)
            raise ValueError('JPDAF measurement matrix not positive semi-definite (pre spread-of-means term')
        
        # Initialize Spread Of Means matrix
        spreadOfMeans = np.zeros([self.totalDimension, self.totalDimension])
        # Compute the "spread of means" term
        for signalName in validAssociationsDict:
            currentPR = signalAssociationProbability[signalName]

            # Compute the difference between the jointly-updated state vector,
            # and the locally updated state vector.
            xDiff = xPlus - validAssociationsDict[signalName]['xPlus']

            spreadOfMeans = (
                spreadOfMeans +
                currentPR * np.outer(xDiff, xDiff)
            )
            
        PPlus = PPlus + spreadOfMeans

        try:
            np.linalg.cholesky(PPlus)
        except:
            print('PPlus = %s' %PPlus)
            raise ValueError('JPDAF measurement matrix not positive semi-definite (post spread-of-means term')
            
        self.covarianceMatrix = PPlus
        
        self.storeGlobalStateVector(xPlus, PPlus, aPriori=False)
        # print(signalAssociationProbability)
        return (xPlus, PPlus, measurement)

    def getGlobalStateVector(
            self
            ):
        globalStateVector = np.zeros(self.totalDimension)
        
        for stateName in self.subStates:
            globalStateVector[self.subStates[stateName]['index']] = (
                self.subStates[stateName]['stateObject'].getStateVector()['stateVector']
                )
        return(globalStateVector)

    def storeGlobalStateVector(
            self,
            globalStateVector,
            covariance,
            aPriori=False
            ):
        newSVID = self.lastStateVectorID + 1
        for stateName in self.subStates:
            mySlice = self.subStates[stateName]['index']
            svDict = {
                'stateVector': globalStateVector[mySlice],
                'covariance': covariance[mySlice, mySlice],
                't': self.tCurrent,
                'aPriori': aPriori,
                'stateVectorID': newSVID
                }

            self.subStates[stateName]['stateObject'].storeStateVector(svDict)
        self.lastStateVectorID = newSVID
        return

    """
    localStateUpdateMatrices
    This function is responsible for assembling a sub-component of the global
    measurement matrix, assuming that the signal in question originated from a
    single signal source.

    Inputs:
    - measurement: A dictionary containing all measured quantities being used
    in the update
    - signalSource: A string refering to the signal source which is being
    assumed to be the origin.
    """
    def localStateUpdateMatrices(
            self,
            measurement,
            signalSourceName,
            xMinus,
            PMinus
            ):
        measurementDimensions = {}
        measurementMatrixDict = {}
        residualDict = {}
        varianceDict = {}

        totaldYLength = 0

        # This loop iterates through each of the substate blocks to retrieve
        # the measurement matrix, the measurement residual, and the
        # measurement variance.
        for stateName in self.subStates:
            localMeasurementMatrices = (
                self.subStates[stateName]['stateObject'].getMeasurementMatrices(
                    measurement,
                    source=self.signalSources[signalSourceName]
                    )
                )

            localHDict = localMeasurementMatrices['H']
            localdYDict = localMeasurementMatrices['dY']
            localRDict = localMeasurementMatrices['R']

            # In this loop, we check to see what measurements (or inferred
            # measurements) the state is associating with its update.  Since
            # we don't know ahead of time what the dimensions of the inferred
            # measurements might be, we also use this loop to gather those
            # dimensions.  If two states return values for the same inferred
            # measurement, the lengths are checked to ensure that they are
            # equal.  If they are not, an error is thrown.
            for key in localdYDict:
                if localdYDict[key] is not None:
                    if key in measurementDimensions:
                        if (
                                len(localdYDict[key]) !=
                                measurementDimensions[key]['length']
                                ):
                            raise ValueError(
                                'Inferred measurement lengths do not ' +
                                'match.  Previous inferred measurement ' +
                                'length was %i, new inferred ' +
                                'measurement length is %i.  \n\n' +
                                'This problem is caused by different ' +
                                'substate objects producing different ' +
                                'lengths of inferred measurements for ' +
                                'the same base measurement type.  This ' +
                                'probably means that there is an ' +
                                'inconsistency in how the substate ' +
                                'objects are interpreting measurements.'
                                % (measurementDimensions[key]['length'],
                                   len(localdYDict[key]))
                                )
                    else:
                        newdYLength = totaldYLength + len(localdYDict[key])
                        
                        measurementDimensions[key] = {
                            'length': len(localdYDict[key]),
                            'index': slice(
                                totaldYLength,
                                newdYLength
                                )
                            }
                        totaldYLength = newdYLength

            measurementMatrixDict[stateName] = localHDict
            residualDict[stateName] = localdYDict
            varianceDict[stateName] = localRDict

        totalHMatrix = np.zeros([totaldYLength, self.totalDimension])
        totalRMatrix = np.zeros([totaldYLength, totaldYLength])
        totaldYMatrix = np.zeros(totaldYLength)
        
        for stateName in self.subStates:
            localHDict = measurementMatrixDict[stateName]
            localdYDict = residualDict[stateName]
            localRDict = varianceDict[stateName]

            for key in measurementDimensions:
                if ((key in localdYDict) and
                    (localdYDict[key] is not None)
                    ):
                    
                    # Check the measurement matrix for proper dimenisions
                    if (localHDict[key].shape !=
                       (
                           measurementDimensions[key]['length'],
                           self.subStates[stateName]['length']
                           )
                        ):
                        raise ValueError(
                            'State %s returned a measurement matrix (H) ' +
                            'with incompatible dimensions.\nExpected ' +
                            'Dimensions: (%i, %i)\nReceived Dimensions: %s.'
                            %(measurementDimensions[key]['length'],
                              self.subStates[stateName]['length'],
                              localHDict[key].shape)
                            )

                    # Next check the measurement residual matrix for proper
                    # dimensions.
                    if (localRDict[key].shape !=
                        (measurementDimensions[key]['length'],
                         measurementDimensions[key]['length'])):
                        
                        raise ValueError(
                            'State %s returned a measurement noise matrix ' +
                            '(Q) with incompatible dimensions.\n Expected' +
                            'Dimensions: (%i, %i)\nReceived Dimensions: %s.'
                            % (
                                localRDict[key].shape,
                                measurementDimensions[key]['length'],
                                measurementDimensions[key]['length']
                            )
                        )
                    
                    # No need to check the measurement residual length, since
                    # we already checked those in the previous loop.

                    # Populate the total matricies with submatrices
                    totalHMatrix[
                        measurementDimensions[key]['index'],
                        self.subStates[stateName]['index']
                    ] = localHDict[key]

                    totalRMatrix[
                        measurementDimensions[key]['index'],
                        measurementDimensions[key]['index'],
                    ] = (
                        totalRMatrix[
                            measurementDimensions[key]['index'],
                            measurementDimensions[key]['index'],
                        ] + localRDict[key]
                    )
                    
                    totaldYMatrix[
                        measurementDimensions[key]['index']
                    ] = (
                        totaldYMatrix[
                            measurementDimensions[key]['index']
                        ] + localdYDict[key]
                    )

        S = totalHMatrix.dot(PMinus).dot(totalHMatrix.transpose()) + totalRMatrix
        K = PMinus.dot(totalHMatrix.transpose()).dot(ModularFilter.covarianceInverse(S))
        # K = PMinus.dot(totalHMatrix.transpose()).dot(np.linalg.inv(S))

        IminusKH = np.eye(self.totalDimension) - K.dot(totalHMatrix)

        xPlus = xMinus + K.dot(totaldYMatrix)
        PPlus = (
            IminusKH.dot(PMinus).dot(IminusKH.transpose()) +
            K.dot(totalRMatrix).dot(K.transpose())
            )
        return({
            'xPlus': xPlus,
            'PPlus': PPlus,
            'H': totalHMatrix,
            'R': totalRMatrix,
            'dY': totaldYMatrix
            })
    @staticmethod
    def covarianceInverse(P):
        cholPInv = np.linalg.inv(np.linalg.cholesky(P))
        return (cholPInv.transpose()).dot(cholPInv)

    def initializeRealTimePlot(
            self,
            plotHandle=None
    ):
        if plotHandle is None:
            self.plotHandle = plt.figure()
        else:
            self.plotHandle = plotHandle

        panelIterator = 0
        for substate in self.subStates:
            newAxis = plt.subplot2grid(
                (len(self.subStates), 1), (panelIterator, 0)
            )
            newAxis.set_title(substate)
            self.subStates[substate]['stateObject'].initializeRealTimePlot(
                plotHandle=self.plotHandle,
                axisHandle=newAxis
            )
            panelIterator = panelIterator+1
        plt.tight_layout()

    def realTimePlot(
            self,
            normalized=True
    ):
        if self.plotHandle is None:
            self.initializeRealTimePlot()
        for substate in self.subStates:
            self.subStates[substate]['stateObject'].realTimePlot(normalized)
