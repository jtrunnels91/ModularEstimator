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
        self.detector = SimulatedDetector(
            userData,
            ureg
        )
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
        
