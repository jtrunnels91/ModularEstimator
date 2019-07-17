from astropy.io import fits
from pint import UnitRegistry
import numpy as np
from scipy.interpolate import interp1d

from .. import utils

class SimulatedSpacecraft():
    def __init__(
            self,
            userData,
            ureg
    ):
        if 'detector' in userData:
            self.detector = SimulatedDetector(
                userData,
                ureg
            )
        if 'dynamicsType' in userData.dynamics and userData.dynamics.dynamicsType.value == 'orbital':
            self.dynamics = SimulatedOrbitalDynamics(
                userData,
                ureg
            )
            if 'simulation' in userData and 'runtime' in userData.simulation:
                tFinal = (
                    userData.simulation.runtime.value * ureg(userData.simulation.runtime.unit)
                ).to(ureg.s).magnitude

                self.dynamics.forwardTimePropagation(tFinal)
            
        else:
            self.dynamics = SimulatedDynamics(
                userData,
                ureg
            )
        self.tStart = 0

        return

    
class SimulatedDetector():
    def __init__(
            self,
            userData,
            ureg
    ):

        # Unpack basic parameters of detector
        self.pixelResolutionX = (
            userData.detector.pixelResolution.value *
            ureg(userData.detector.pixelResolution.unit)
        ).to(ureg('rad')/ ureg('pixel')).magnitude
        self.pixelResolutionY = (
            userData.detector.pixelResolution.value *
            ureg(userData.detector.pixelResolution.unit)
        ).to(ureg('rad')/ ureg('pixel')).magnitude
        
        self.FOV = (
            userData.detector.FOV.value *
            ureg(userData.detector.FOV.unit)
        ).to(ureg('deg')).magnitude
        self.area = (
            userData.detector.area.value *
            ureg(userData.detector.area.unit)
        ).to(ureg.cm ** 2).magnitude

        
        # Determine the resolution and standard deviation of arrival times
        self.timeResolution = (
            userData.detector.timeResolution.value *
            ureg(userData.detector.timeResolution.unit)
        ).to(ureg('s')).magnitude
        
        if userData.detector.TOAstdev.distribution == 'uniform':
            self.TOA_StdDev = self.timeResolution/np.sqrt(12)
        elif userData.detector.TOAstdev.distribution == 'normal':
            self.TOA_StdDev = (
                userData.detector.TOAstdev.value *
                ureg(userData.detector.TOAstdev.unit)
            ).to(ureg('s')).magnitude

        # Use pixel resolution to determine the standard deviation of
        # photon AOA measurements
        if userData.detector.AOAstdev.distribution == 'uniform':
            self.AOA_xStdDev = self.pixelResolutionX/np.sqrt(12)
            self.AOA_yStdDev = self.pixelResolutionY/np.sqrt(12)
        elif userData.detector.AOAstdev.distribution == 'normal':
            self.AOA_xStdDev = (
                userData.detector.AOAstdev.value *
                ureg(userData.detector.AOAstdev.unit)
            ).to(ureg('rad')).magnitude
            
            self.AOA_yStdDev = self.AOA_xStdDev
        self.AOA_StdDev = (self.AOA_xStdDev + self.AOA_yStdDev)/2

        # Store variances for measurements in addition to standard deviations
        self.AOA_xVar = np.square(self.AOA_xStdDev)
        self.AOA_yVar = np.square(self.AOA_yStdDev)
        self.TOA_var = np.square(self.TOA_StdDev)
        
        self.lowerEnergy = (
            userData.detector.energyRange.lower.value *
            ureg(userData.detector.energyRange.lower.unit)
        ).to(ureg.kiloelectron_volt).magnitude
        self.upperEnergy = (
            userData.detector.energyRange.upper.value *
            ureg(userData.detector.energyRange.upper.unit)
        ).to(ureg.kiloelectron_volt).magnitude
        self.energyRange = [self.lowerEnergy, self.upperEnergy]
        self.energyRangeKeV = [self.lowerEnergy, self.upperEnergy]

class SimulatedOrbitalDynamics():
    def __init__(
            self,
            userData,
            ureg
    ):
        if 'MJDREF' in userData.dynamics:
            self.MJDREF = userData.dynamics.MDJREF.value
        else:
            self.MJDREF = utils.spacegeometry.timeObj.now().tt - 2400000.5 

        self.timeStep = 1
        if 'timeStep' in userData.dynamics:
            self.timeStep = (
                userData.dynamics.timeStep.value *
                ureg(userData.dynamics.timeStep.unit)
            ).to(ureg.seconds).magnitude

        self.positionHistory = np.array([[0,0,0]])
        self.velocityHistory = np.array([[0,0,0]])
        self.timeHistory = np.array([0])
        
        if 'position' in userData.dynamics:
            self.positionHistory[0] = (
                userData.dynamics.position.value *
                ureg(userData.dynamics.position.unit)
            ).to(ureg.km).magnitude
        if 'velocity' in userData.dynamics:
            self.velocityHistory[0] = (
                userData.dynamics.velocity.value *
                ureg(userData.dynamics.velocity.unit)
            ).to(ureg.km/ureg.seconds).magnitude

        self.angularVelocity = np.array([0,0,0])
        self.attitudeHistory = np.array([[0,0,0]])
        self.initialAttitude = np.zeros(3)
        
        if 'attitude' in userData.dynamics:
            self.angularVelocity = (
                userData.dynamics.attitude.angularVelocity.value *
                ureg(userData.dynamics.attitude.angularVelocity.unit)
            ).to(ureg.rad/ureg.s).magnitude

            self.initialAttitude = (
                userData.dynamics.attitude.initialAttitude.value *
                ureg(userData.dynamics.attitude.initialAttitude.unit)
            ).to(ureg.rad).magnitude
            
            self.attitudeHistory[0] = self.initialAttitude

        self.secondsToDays = 1/(24.0 * 60.0 * 60.0)
        return

    def getRangeFunction(
            self,
            unitVector,
            tFinal
    ):
        self.forwardTimePropagation(tFinal)

        rangeVector = self.positionHistory.dot(unitVector)
        # rangeVector = np.zeros(len(self.timeHistory))
        return interp1d(
            self.timeHistory,
            rangeVector,
            kind='cubic',
            assume_sorted=True,
            fill_value='extrapolate'
        )

    def forwardTimePropagation(
            self,
            tFinal
    ):
        if tFinal < self.timeHistory[-1]:
            return
        
        currentLength = len(self.timeHistory)
        currentIndex = currentLength
        currentTime = self.timeHistory[-1]

        
        newTimeSteps = np.int(np.ceil((tFinal-currentTime)/self.timeStep))
        
        self.timeHistory = np.append(self.timeHistory, np.zeros(newTimeSteps))

        self.positionHistory = np.append(self.positionHistory, np.zeros([newTimeSteps, 3]), axis=0)
        self.velocityHistory = np.append(self.velocityHistory, np.zeros([newTimeSteps, 3]), axis=0)
        self.attitudeHistory = np.append(self.attitudeHistory, np.zeros([newTimeSteps, 3]), axis=0)

        while currentIndex < len(self.timeHistory):
            self.timeHistory[currentIndex] = (
                self.timeHistory[currentIndex - 1] + self.timeStep
            )

            currentAcceleration = utils.spacegeometry.acceleration(
                self.positionHistory[currentIndex-1],
                self.getTimeScaleObject(self.timeHistory[currentIndex-1])
            )
            self.velocityHistory[currentIndex] = (
                self.velocityHistory[currentIndex-1] +
                currentAcceleration * self.timeStep
            )
            self.positionHistory[currentIndex] = (
                self.positionHistory[currentIndex-1] +
                self.velocityHistory[currentIndex-1] * self.timeStep +
                currentAcceleration * np.square(self.timeStep)/2
            )

            self.attitudeHistory[currentIndex] = (
                self.attitudeHistory[currentIndex - 1] +
                self.angularVelocity * self.timeStep
            )
            
            currentIndex += 1

        self.posX = interp1d(self.timeHistory, self.positionHistory[:,0], kind='cubic', assume_sorted=True, fill_value='extrapolate')
        self.posY = interp1d(self.timeHistory, self.positionHistory[:,1], kind='cubic', assume_sorted=True, fill_value='extrapolate')
        self.posZ = interp1d(self.timeHistory, self.positionHistory[:,2], kind='cubic', assume_sorted=True, fill_value='extrapolate')
        
        self.velX = interp1d(self.timeHistory, self.velocityHistory[:,0], kind='cubic', assume_sorted=True, fill_value='extrapolate')
        self.velY = interp1d(self.timeHistory, self.velocityHistory[:,1], kind='cubic', assume_sorted=True, fill_value='extrapolate')
        self.velZ = interp1d(self.timeHistory, self.velocityHistory[:,2], kind='cubic', assume_sorted=True, fill_value='extrapolate')

        self.attX = interp1d(self.timeHistory, self.attitudeHistory[:,0], kind='cubic', assume_sorted=True, fill_value='extrapolate')
        self.attY = interp1d(self.timeHistory, self.attitudeHistory[:,1], kind='cubic', assume_sorted=True, fill_value='extrapolate')
        self.attZ = interp1d(self.timeHistory, self.attitudeHistory[:,2], kind='cubic', assume_sorted=True, fill_value='extrapolate')
        return

    def position(
            self,
            time
    ):
        self.forwardTimePropagation(time)

        return np.array([self.posX(time)[()], self.posY(time)[()], self.posZ(time)[()]])
        # return np.zeros(3)

    def velocity(
            self,
            time
    ):
        self.forwardTimePropagation(time)

        return np.array([self.velX(time)[()], self.velY(time)[()], self.velZ(time)[()]])
        # return np.zeros(3)

    def acceleration(
            self,
            time
    ):
        currentPosition = self.position(time)
        return utils.spacegeometry.acceleration(currentPosition, self.getTimeScaleObject(time))
        # return np.zeros(3)

    def gradient(self, time):
        currentPosition = self.position(time)
        return utils.spacegeometry.accelerationGradient(currentPosition, self.getTimeScaleObject(time))
    def attitude(
            self,
            t,
            returnQ=True
    ):
        if hasattr(t, '__len__'):
            attitudeArray = []
            for i in range(len(t)):
                attitudeArray.append(self.attitude(t[i],returnQ))
            return attitudeArray
        else:
            self.forwardTimePropagation(t)
            
            eulerAngles = np.array([self.attX(t)[()], self.attY(t)[()], self.attZ(t)[()]])

            if returnQ:
                return utils.euler2quaternion(eulerAngles)
            else:
                return(eulerAngles)

    def initialAttitudeRotationMatrix(
            self
            ):
        if self.__initialAttitudeRotationMatrix__ is None:
            self.__initialAttitudeRotationMatrix__ = self.attitude(0).rotation_matrix
        return self.__initialAttitudeRotationMatrix__
    
    def omega(
            self,
            t
    ):
        return(self.angularVelocity)
        
    def getTimeScaleObject(
            self,
            time
    ):
        return utils.spacegeometry.timeObj.tt_jd(
            2400000.5 +
            self.MJDREF +
            (time*self.secondsToDays)
        )
        
class SimulatedDynamics():
    def __init__(
            self,
            userData,
            ureg
            ):
        self.MJDREF = 58591.50694
        self.__initialAttitudeRotationMatrix__ = None
        # Define a series of functions which describe the dynamics of the spacecraft
        self.angularVelocity = (
            userData.dynamics.attitude.angularVelocity.value *
            ureg(userData.dynamics.attitude.angularVelocity.unit)
        ).to(ureg.rad/ureg.s).magnitude

        if not userData.dynamics.attitude.initialAttitude.value:
            self.initialAttitude = None
        else:
            self.initialAttitude = (
                userData.dynamics.attitude.initialAttitude.value *
                ureg(userData.dynamics.attitude.initialAttitude.unit)
            ).to(ureg.rad).magnitude

        self.orbitAmplitude = (
            userData.dynamics.orbit.amplitude.value *
            ureg(userData.dynamics.orbit.amplitude.unit)
        ).to(ureg.km).magnitude
    
        self.orbitPeriod = (
            userData.dynamics.orbit.period.value *
            ureg(userData.dynamics.orbit.period.unit)
        ).to(ureg.s).magnitude
    
    def attitude(
            self,
            t,
            returnQ=True
    ):
        if hasattr(t, '__len__'):
            attitudeArray = []
            for i in range(len(t)):
                attitudeArray.append(self.attitude(t[i],returnQ))
            return attitudeArray
        else:
            eulerAngles = [
                (t * self.angularVelocity[0]) + self.initialAttitude[0],
                (t * self.angularVelocity[1]) + self.initialAttitude[1],
                (t * self.angularVelocity[2]) + self.initialAttitude[2]
            ]

            if returnQ:
                return utils.euler2quaternion(eulerAngles)
            else:
                return(eulerAngles)

    def initialAttitudeRotationMatrix(
            self
            ):
        if self.__initialAttitudeRotationMatrix__ is None:
            self.__initialAttitudeRotationMatrix__ = self.attitude(0).rotation_matrix
        return self.__initialAttitudeRotationMatrix__
    def omega(
            self,
            t
    ):
        return(self.angularVelocity)

    def position(
            self,
            t
    ):
        return(
            np.array([
                self.orbitAmplitude * np.cos(t/self.orbitPeriod),
                self.orbitAmplitude * np.sin(t/self.orbitPeriod),
                0 * t
            ])
        )

    def velocity(
            self,
            t
    ):
        return(
            (self.orbitAmplitude/self.orbitPeriod) *
            np.array([
                -np.sin(t/self.orbitPeriod),
                np.cos(t/self.orbitPeriod),
                0 * t
                ]
            )
        )

    def acceleration(
            self,
            t
    ):
        return(
            np.power(self.orbitAmplitude/self.orbitPeriod, 2) *
            np.array([
                np.sin(t/self.orbitPeriod),
                -np.cos(t/self.orbitPeriod),
                0 * t
                ]
            )
        )
        
    def getRangeFunction(
            self,
            unitVector,
            tFinal
    ):
        def rangeFunction(t):
            return(
                np.array([
                    self.orbitAmplitude * np.cos(t/self.orbitPeriod),
                    self.orbitAmplitude * np.sin(t/self.orbitPeriod),
                    t * 0 
                ]).dot(unitVector)
            )            
        return rangeFunction
