"""
.. module::modularfilter
:synopsis: This package contains the ModularFilter class.

.. moduleauthor:: Joel Runnels

Note:
 This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program.  If not, see http://www.gnu.org/licenses/

"""
# 
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

## @package modularfilter
# This package contains the ModularFilter class.
# @author Joel Runnels
# @date 2018
# @copyright GNU General Public License
#
# @note

import numpy as np
from scipy.linalg import block_diag, ldl
from numpy import sin, cos, arcsin, arccos, arctan2, square, sqrt, abs, power
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
# from pyquaternion import Quaternion
from math import isnan
from . utils import covarianceContainer
import sys
import os
sys.path.append("/home/joel/Documents/astroSourceTracking/libraries")


class ModularFilter():
    """ 
    This class is designed to facilitate a variety of estimation algorithms in
    a modular way, i.e. in a way that allows maximum amount of flexibility with
    minimum amount of code duplication.

    The idea behind this class is that all of the functions which are "generic"
    to an estimation algorithm can be written just once, as a part of this
    class. Then the code contained here can be implemented on a variety of
    different estimation problems without having to duplicate code.

    The basic estimation model implemented here works as follows.  The overall
    state, i.e. all of the quantities to be estimated, are represented by
    :class:`~modest.substates.substate.SubState` objects. Measurements of
    these states are represented by 
    :class:`~modest.signals.signalsource.SignalSource` objects.  The
    ModularFilter class is responsible for doing time-updates and
    measurement-updates on these states. These objects are responsible for
    doing all of the things that are particular to those states, and those
    signals, respectively.  For instance, generation of time-update and
    measurement update matrices is handled by the
    :class:`~modest.substates.substate.SubState` objects.

    Attributes
    ----------
       totalDimension (int):Dimension of joint state vector (sum of all substate dimensions)
       covarianceMatrix(covarianceContainer): Object which stores covariance matrix
       subStates(dict): Dictionary of all substate objects
       signalSources(dict): Dictionary of all signal source objects
       tCurrent(float): Current filter time
       measurementValidationThreshold(float): Probability below which a signal source is rejected
       measurementList(list): A list of measurements used by the filter
       lastStateVectorID(int): A tag used to specify what the "last" state vector was
       covarianceStroage(str): String specifying how state error covariance is to be stored
       plotHandle: Handle for real-time state plotting
    """
    
    def __init__(
            self,
            measurementValidationThreshold=1e-3,
            time=0,
            covarianceStorage='covariance'
    ):
        """
        __init__ does the house-keeping work to initialize a ModularFilter
        
        Args: 
            time (float): The "starting time" of the filter (default=0)
            covarianceStorage (str): Specify how covariance is to be stored, whether as the full covariance matrix ("covariance"), or as the square-root representation ("cholesky")
            measurementValidationThreshold (float): This is a value that specifies the minimum probability an association must have in order to be included in a joint measurement update
        """
        
        self.covarianceStorage=covarianceStorage
        self.plotHandle=None

        self.totalDimension = 0
        self.covarianceMatrix = covarianceContainer(np.zeros([0, 0]), covarianceStorage)

        self.subStates = {}
        self.signalSources = {}
        self.tCurrent = time

        self.measurementValidationThreshold = measurementValidationThreshold

        self.measurementList = []
        self.lastMeasurementID = None

        self.lastStateVectorID = 0
        
        return


    def addStates(
            self,
            name,
            stateObject
    ):
        """
        addStates is a utility function used to add a state to the ModularFilter.
        
        Args: 
            name (str): The name by which the state can be referenced (must be unique)
            stateObject (substate): An object that contains the sub-state and is responsible for performing various tasks related to the substate.
        """


        # Check to see whether the name is unique
        if name in self.subStates:
            raise ValueError(
                ('The name %s has already been used for a state.  If you ' +
                'want to remove that state, you can use the removeStates ' +
                'function.  If you want to replace it, you can use the ' +
                'replaceStates function.') % name
                )

        newSlice = slice(self.totalDimension, self.totalDimension + stateObject.dimension())
        
        self.totalDimension = (
            self.totalDimension + stateObject.dimension()
            )

        otherCovariance = stateObject.covariance()
        if otherCovariance.form != self.covarianceMatrix.form:
            otherCovariance = otherCovariance.convertCovariance(self.covarianceMatrix.form)

        self.covarianceMatrix = covarianceContainer(
            block_diag(
                self.covarianceMatrix.value,
                otherCovariance.value
            ),
            self.covarianceMatrix.form
        )

        self.subStates[name] = {
            'index': newSlice,
            'length': stateObject.dimension(),
            'stateObject': stateObject
            }
        return

    def addSignalSource(
            self,
            name,
            signalSourceObject
    ):
        """
        addSignalSource is a utility function used to add a signal source to the joint estimator.
        
        Args:
         name (str): The name by which the signal source can be referenced (must be unique)
         signalSourceObject (SignalSource): An object that contains the signal source and is responsible for performing various tasks related to the substate.
        """
        # Check to see whether the name is unique
        if name in self.signalSources:
            raise ValueError(
                'The name %s has already been used for a signal source.  ' +
                'If you want to remove that signal source, you can use the ' +
                'removeSignalSource function.  If you want to replace it, ' +
                'you can use the replaceSignalSource function.'
                %name
                )

        self.signalSources[name] = signalSourceObject
        
        return

    def timeUpdateEKF(
            self,
            dT,
            dynamics=None
            ):
        """
        Performs a standard extended Kalman filter time-update using information found in dynamcs, and over time-interval dT.

        Following the general framework for the ModularFilter, this function only performs the tasks that are universal across all the substates.  It is up to the substates to process the dynamics infromation and build the appropriate time-update matrices ($F$ and $Q$).  This is done by calling :meth:`~modest.substates.substate.SubState.timeUpdate` for each substate.

        Once the substate time update matrices are received, they are combined to form a single set of time-update matrices, and then the state vector and covariance are updated.  The updated states and covariances are then passed back to the substates for them to do any nescessary post-processing.

        This function allows for a variable-length time update, so it is up to the user to determine what length of time is appropriate.  Often, the time-update will simply update the state to the next measurement time, if the time between measurements is short.  If the time between measurements is longer, then the length of time-update will most likely depend upon what time-regime the dynamics dictionary is valid over.  For example if the dynamics dict contains acceleration information, the user should specify a dT over which the acceleration is *approximately* constant.
        
        While values are returned containing the time-updated state vector and covariance, these are also automatically stored and dissimenated to the substates for additional processing, so the user does not *need* to do anything with these returned values.  They are returned merely for convenience and monitoring by the user as desired.

        Args:
         dT (float): The amount of time over which the time-update is occuring
         dynamics (dict): A dictionary containing all information about the dynamics during the time update 
        Returns:
         numpy.array, covarianceContainer: The time-updated state vector and covariance
        
        Example: ::

            myFilter.timeUpdateEKF(
                0.01, 
                dynamics={
                'acceleration': {'value': [0, 0, -9.81], 'var' [0.01, 0.01, 0.01]}
                }
            )
        This command would  pass the dynamics dict containing a three-dimensional acceleration vector and associated variances to the substates, and perform a time update from the current time to time pluse 0.01.  Note that it is up to the substates to decide what to do (if anything) with the dynamics information.
        """
        
        F = np.zeros([self.totalDimension, self.totalDimension])
        Q = covarianceContainer(
            np.zeros([self.totalDimension, self.totalDimension]), 'covariance'#self.covarianceMatrix.form
        )

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

            # try:
            #     np.linalg.cholesky(timeUpdateMatrices['Q'])
            # except:
            #     raise ValueError(
            #         'Process noise matrix Q for substate %s not positive semi-definite'
            #         %stateName
            #     )
            F[mySlice, mySlice] = timeUpdateMatrices['F']
            Q[mySlice, mySlice] = timeUpdateMatrices['Q']

        # try:
        #     np.linalg.cholesky(Q)
        # except:
        #     raise ValueError('Q matrix in EKF time update not positive semidefinite')
        
        xMinus = F.dot(self.getGlobalStateVector())

        if self.covarianceMatrix.form == 'covariance':
            # Standard Kalman Filter equation
            PMinus = F.dot(self.covarianceMatrix.value).dot(F.transpose()) + Q.value
            
        elif self.covarianceMatrix.form == 'cholesky':
            # Square root filter time update equation based on Gram-Schmidt
            # orthogonalization.  See Optimal State Estimation (Simon),
            # Page 162-163 for derivation.
            M = np.vstack([
                self.covarianceMatrix.value.transpose().dot(F.transpose()),
                Q.convertCovariance('cholesky').value.transpose()
                ]
            )
            T = np.linalg.qr(M)
            PMinus = T[1].transpose()
            # if PMinus[0,0] < 0:
            #     PMinus = -PMinus
                    
            
        # try:
        #     np.linalg.cholesky(PMinus)
        # except:
        #     raise ValueError('PMinus matrix in EKF time update not positive semidefinite')

        self.tCurrent = self.tCurrent + dT
        
        self.covarianceMatrix = covarianceContainer(PMinus, self.covarianceMatrix.form)
        
        self.storeGlobalStateVector(xMinus, self.covarianceMatrix, aPriori=True)
        
        return (xMinus, PMinus)
    
    def computeAssociationProbabilities(
            self,
            measurement
    ):
        """
        computeAssociationProbabilities is responsible for computing the
        probability that a measurement originated from each of the signal sources
        being tracked by the estimator.

        The function assumes that each signal source object has a member function which will compute the probability of origination, given the measurement and the prior state values, :meth:`~modest.signals.signalsource.SignalSource`. Additionally, it assumes that the objects take a validation threshold as an optional argument.
        
        It is up to each individual signal source to determine it's own probability of association given the current state vector.  For some signal sources the probability may be zero, particularly if the measurement does not contain anything that could have possibly originated from that signal source (e.g. a signal source that measures only velocity would have zero probability of association with a position measurement).

        Args:
         measurement (dict): The measurement for which the probability is to be computed.

        Example: ::

            myFilter.computeAssociationProbabilities(
                {
                'position': {'value': [0, 0, 100], 'var' [0.01, 0.01, 0.01]}
                }
            )

    """
        

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
            elif isnan(currentProbability):
                raise ValueError(
                    'Got NaN probability for measurement %s, signal source %s'
                    %(measurement, signalKey)
                    )
            probabilityDict[signalKey] = currentProbability
            probabilitySum = probabilitySum + currentProbability
        for probabilityKey in probabilityDict:
            if probabilitySum > 0:
                probabilityDict[probabilityKey] = (
                    probabilityDict[probabilityKey] / probabilitySum
                )
            else:
                probabilityDict[probabilityKey] = (
                    probabilityDict[probabilityKey]
                )

        # print(probabilityDict)
        return (probabilityDict)


    def measurementUpdateEKF(
            self,
            measurement,
            sourceName
            ):
        """
        measurementUpdateEKF performs a standard Extended Kalman Filter measurement
        update.  

        This function only works when the signal source is known, which
        may or may not be a realistic assumption, depending on the problem.
        Alternatively, it can be used for comparison to other data association
        methods.

        As with other update methods, the updated state and covariance are returned.  However the function also stores and distributes the state and covariance to the substates, so the user does not need to do anything with the returned state and covariance.

        Note that there is no time associated with the measurement; the filter
        assumes that the measurement is occuring at the current time. 
        Therefore it is the user's responsibility to time-update the state to
        the current time before doing the measurement update.

        Args:
         measurement (dict): Dictionary containing all the relevant measurable quanties labeled in a way that the substates and signal source objects understand.
         sourceName (str): A string that contains the name of the signal source from which the measurement originated.  This must be the name of one of the signal sources that have been added to the estimator.

        Returns:
         numpy.array, covarianceContainer: The measurement-updated state vector and covariance

        Example: ::

            myFilter.computeAssociationProbabilities(
                {
                'position': {'value': [0, 0, 100], 'var' [0.01, 0.01, 0.01]},
                },
                'range1'
            )

        """
        
        if 'ID' not in measurement:
            if self.lastMeasurementID == None:
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
        """
        measurementUpdateML performs a maximum-likelhood measurement update

        The first step is to assemble the global state vector from each sub-state vector.  This is performed using :meth:`getGlobalStateVector`.

        This function first computes the probability of association of the measurement with each signal source using :meth:`computeAssociationProbabilities`.  Whichever signal is most likely is assumed to be the correct measurement source and is used to update the state (assuming the probability is above :attr:`measurementValidationThreshold`).

        Finally, the updated state vector and covariance matrix is distributed to the substates using :meth:`storeGlobalStateVector`.

        As with other update methods, the updated state and covariance are returned.  However the function also stores and distributes the state and covariance to the substates, so the user does not need to do anything with the returned state and covariance.

        Note that there is no time associated with the measurement; the filter
        assumes that the measurement is occuring at the current time. 
        Therefore it is the user's responsibility to time-update the state to
        the current time before doing the measurement update.

        
        Args:
         measurement (dict): Dictionary containing all the relevant measurable quanties labeled in a way that the substates and signal source objects understand.

        Returns:
         measurement (dict): Dictionary containing all the relevant measurable quanties labeled in a way that the substates and signal source objects understand.

        Example: ::

            myFilter.computeAssociationProbabilities(
                {
                'position': {'value': [0, 0, 100], 'var' [0.01, 0.01, 0.01]},
                }
            )

        """    

        signalAssociationProbability = (
            self.computeAssociationProbabilities(measurement)
            )

        del signalAssociationProbability['background']

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
        """
        measurementUpdateJPDAF performs a measurement using the joint probabilistic data association framework.

        The first step is to assemble the global state vector from each sub-state vector.  This is performed using :meth:`getGlobalStateVector`.

        This function then computes the probability of association of the measurement with each signal source using :meth:`computeAssociationProbabilities`.  Associations with probabilities lower than :attr:`measurementValidationThreshold`.

        Finally, the updated state vector and covariance matrix is distributed to the substates using :meth:`storeGlobalStateVector`.

        As with other update methods, the updated state and covariance are returned.  However the function also stores and distributes the state and covariance to the substates, so the user does not need to do anything with the returned state and covariance.

        Note that there is no time associated with the measurement; the filter
        assumes that the measurement is occuring at the current time. 
        Therefore it is the user's responsibility to time-update the state to
        the current time before doing the measurement update.
        
        Args:
         measurement (dict): Dictionary containing all the relevant measurable quanties labeled in a way that the substates and signal source objects understand.

        Returns:
         measurement (dict): Dictionary containing all the relevant measurable quanties labeled in a way that the substates and signal source objects understand.

        Example: ::

            myFilter.computeAssociationProbabilities(
                {
                'position': {'value': [0, 0, 100], 'var' [0.01, 0.01, 0.01]},
                }
            )

        """    
        
        
        signalAssociationProbability = (
            self.computeAssociationProbabilities(measurement)
            )
        # print(signalAssociationProbability)
        measurement['associationProbabilities']=signalAssociationProbability

        #print(signalAssociationProbability)
        xMinus = self.getGlobalStateVector()
        PMinus = self.covarianceMatrix

        xPlus = np.zeros(self.totalDimension)
        PPlus = None
        # print('Started measurement update')

        validAssociationsDict = {}
        for signalName in signalAssociationProbability:
            currentPR = signalAssociationProbability[signalName]
            if isnan(currentPR):
                print(signalAssociationProbability)
                raise ValueError('Received NaN for probability')
            elif currentPR < 0:
                print(signalAssociationProbability)
                raise ValueError('Received negaitve probability')
            elif currentPR > 1:
                print(signalAssociationProbability)
                raise ValueError('Received probability greater than 1')

            if (
                    currentPR >
                    self.measurementValidationThreshold
            ):
                # currentPR = 1
                # signalAssociationProbability[signalName]=1
                try:
                    updateDict = self.localStateUpdateMatrices(
                        measurement,
                        signalName,
                        xMinus,
                        PMinus
                    )
                except:
                    print(measurement)
                    print(signalName)
                    print(xMinus)
                    print(PMinus)
                    raise ValueError('Computed NaN state update')

                if np.any([isnan(stateVal) for stateVal in updateDict['xPlus']]):
                    raise ValueError(
                        'The following signal association computed a NaN ' +
                        'state vector:\n' +
                        'Association: %s \n'
                        #'Association Probability: %s\n' +
                        % signalName
                    )
                xPlus = (
                    xPlus + (currentPR * updateDict['xPlus'])
                    )

                if self.covarianceMatrix.form == 'covariance':
                    if PPlus is not None:

                        # print('PPlus is not none')
                        # print(signalName)
                        PPlus = (
                            PPlus + (currentPR * updateDict['PPlus'].value)
                        )
                    else:
                        # print('PPlus is none')
                        # print(signalName)
                        # print(PMinus.value - updateDict['PPlus'].value)
                        
                        PPlus = currentPR * updateDict['PPlus'].value
                        
                elif self.covarianceMatrix.form == 'cholesky':
                    # If we're doing square root filtering, then we can't
                    # simply add the square roots of covariance together.
                    # Rather we have to stack them, then do the QR factorization.
                    if PPlus is not None:
                        PPlus = np.vstack(
                            [PPlus,
                             (np.sqrt(currentPR) * updateDict['PPlus'].value).transpose()
                            ]
                        )
                    else:
                        PPlus = (np.sqrt(currentPR) * updateDict['PPlus'].value).transpose()
                else:
                    raise ValueError('Unrecougnized covariance storage method')

                # If the signal association was valid, store it in a dict so
                # that we can go back and compute the spread of means term
                validAssociationsDict[signalName] = updateDict
            # else:
            #     currentPR = 0
            #     signalAssociationProbability[signalName]=0


        # Here, we compute the spread of means.  Note that we compute it the
        # same way regardless of which covariance storage we're using.  If
        # square root filtering, we'll just take the cholesky decomposition
        # after the computation is finished.
        #
        # Also note that we only need to compute the spread of means term if
        # there was more than one valid association.  Otherwise we essentially
        # just have the standard KF
        spreadOfMeans=None
        if len(validAssociationsDict) > 1:
            # Initialize Spread Of Means matrix
            # spreadOfMeans = np.zeros([self.totalDimension, self.totalDimension])
            # Compute the "spread of means" term
            for signalName in validAssociationsDict:
                currentPR = signalAssociationProbability[signalName]

                # Compute the difference between the jointly-updated state vector,
                # and the locally updated state vector.
                xDiff = xPlus - validAssociationsDict[signalName]['xPlus']
                # print(np.linalg.norm(xDiff))
                if spreadOfMeans is not None:
                    if PMinus.form == 'covariance':
                        spreadOfMeans = (
                            spreadOfMeans +
                            currentPR * np.outer(xDiff, xDiff)
                            # np.outer(xDiff, xDiff)
                        )
                    elif PMinus.form == 'cholesky':
                        spreadOfMeans = np.vstack(
                            [spreadOfMeans,
                             np.vstack([
                                 xDiff * np.sqrt(currentPR),
                                 # xDiff,
                                 np.zeros([self.totalDimension-1, self.totalDimension])]
                             )
                            ]
                        )
                else:
                    if PMinus.form == 'covariance':
                        spreadOfMeans = currentPR * np.outer(xDiff, xDiff)
                        # spreadOfMeans = np.outer(xDiff, xDiff)
                    elif PMinus.form == 'cholesky':
                        spreadOfMeans = np.vstack([
                            xDiff * np.sqrt(currentPR),
                            # xDiff,
                            np.zeros([self.totalDimension-1, self.totalDimension])]
                        )
            if PMinus.form == 'covariance':
                PPlus = PPlus + spreadOfMeans
            elif PMinus.form == 'cholesky':
                PPlus = np.vstack([PPlus, spreadOfMeans])

        if PPlus is not None:
            if self.covarianceStorage == 'cholesky':
                QR = np.linalg.qr(PPlus)
                PPlus = QR[1].transpose()
                if PPlus[0,0] < 0:
                    PPlus = - PPlus
                PPlus = covarianceContainer(PPlus, 'cholesky')
            elif self.covarianceMatrix.form == 'covariance':
                PPlus = covarianceContainer(PPlus, 'covariance')
        else:
            xPlus = xMinus
            PPlus = PMinus    

        self.covarianceMatrix = PPlus
        
        self.storeGlobalStateVector(xPlus, PPlus, aPriori=False)
        # print(signalAssociationProbability)
        return (xPlus, PPlus, measurement, validAssociationsDict, spreadOfMeans)

    def getGlobalStateVector(
            self
            ):
        """
        This function collects the substate vectors from each substate, and assembles the global state vector.

        """
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
            if np.any([isnan(stateVal) for stateVal in globalStateVector[mySlice]]):
                print(globalStateVector)
                print('A priori: %s' %aPriori)
                raise ValueError('NaN state vector for substate %s' %stateName)
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
    given signal source.

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
        # try:
        #     np.linalg.cholesky(PMinus)
        # except:
        #     raise ValueError(
        #         'PMinus is not positive semidefinite going into ' +
        #         'measurement update. Signal source %s'
        #         %signalSourceName
        #     )
        
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
                        if hasattr(localdYDict[key], '__len__'):
                            localdYLength = len(localdYDict[key])
                        else:
                            localdYLength = 1

                        newdYLength = totaldYLength + localdYLength
                        measurementDimensions[key] = {
                            'length': localdYLength,
                            'index': slice(
                                totaldYLength,
                                newdYLength
                                )
                            }
                        totaldYLength = newdYLength

            measurementMatrixDict[stateName] = localHDict
            residualDict[stateName] = localdYDict
            varianceDict[stateName] = localRDict

            # for key, subComponentR in localRDict.items():
            #     if subComponentR is not None:
                    # try:
                    #     np.linalg.cholesky(subComponentR)
                    # except:
                    #     print('KEY:')
                    #     print(key)
                    #     print(stateName)
                    #     print('R MATRIX:')
                    #     print(subComponentR)
                    #     raise ValueError(
                    #         'Received a non positive-semidefinite R matrix ' +
                    #         'subcomponent. Substate %s, signal source %s. ' +
                    #         'R matrix:\n%s'
                    #         %(stateName, key)
                    #     )

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
                                measurementDimensions[key]['length'],
                                measurementDimensions[key]['length'],
                                localRDict[key].shape
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
        try:
            xPlus, PPlus = self.computeUpdatedStateandCovariance(
                xMinus,
                PMinus,
                totaldYMatrix,
                totalHMatrix,
                totalRMatrix
            )
        except:
            raise ValueError('Got NaN state vector')

        return({
            'xPlus': xPlus,
            'PPlus': PPlus,
            'H': totalHMatrix,
            'R': totalRMatrix,
            'dY': totaldYMatrix
            })

    def computeUpdatedStateandCovariance(
            self,
            xMinus,
            PMinus,
            dY,
            H,
            R
    ):
        if PMinus.form == 'covariance':
            # Standard Kalman Filter
            S = H.dot(PMinus.value).dot(H.transpose()) + R

            # Could inversion of S be introducting instability?
            K = PMinus.value.dot(H.transpose()).dot(np.linalg.inv(S))

            IminusKH = np.eye(self.totalDimension) - K.dot(H)

            xPlus = xMinus + K.dot(dY)
            PPlus = (
                IminusKH.dot(PMinus.value).dot(IminusKH.transpose()) +
                K.dot(R).dot(K.transpose())
            )
            PPlus = covarianceContainer(PPlus, 'covariance')
        elif PMinus.form == 'cholesky':
            
            W = PMinus.value
            Z = W.transpose().dot(H.transpose())

            # Compute U.  Instead of using cholesky decomposition however, use
            # LDL (since R + Z^TZ can be semidefinite)
            # U = np.linalg.cholesky(R + Z.transpose().dot(Z))
            
            R_plus_ZTZ = R + Z.transpose().dot(Z)
            myLDL = ldl(R_plus_ZTZ)
            U = myLDL[0].dot(np.sqrt(myLDL[1]))
            # Compute V
            # V = np.linalg.cholesky(R)
            myLDL = ldl(R)
            V = myLDL[0].dot(np.sqrt(myLDL[1]))
            UInv = np.linalg.inv(U)

            WPlus = W.dot(
                np.eye(self.totalDimension) -
                Z.dot(UInv.transpose()).dot(np.linalg.inv(U + V)).dot(Z.transpose())
            )
            PPlus = covarianceContainer(WPlus, PMinus.form)
            xPlus = xMinus + W.dot(Z).dot(UInv.transpose()).dot(UInv).dot(dY)
        if np.any([isnan(stateVal) for stateVal in xPlus]):
            print(xPlus)
            raise ValueError(
                'Computed a NaN updated state vector'
            )
        return (xPlus, PPlus)
    
    @staticmethod
    def covarianceInverse(P):
        cholPInv = np.linalg.inv(np.linalg.cholesky(P))
        return (cholPInv.transpose()).dot(cholPInv)

    def initializeRealTimePlot(
            self,
            plotHandle=None
    ):
        if plotHandle == None:
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
        if self.plotHandle == None:
            self.initializeRealTimePlot()
        for substate in self.subStates:
            self.subStates[substate]['stateObject'].realTimePlot(normalized)
