from astropy.io import fits
from pint import UnitRegistry
import numpy as np
from scipy.interpolate import interp1d

from .. import utils

planetaryMasses = {
    'Sun': 1.98855e30,
    'Mercury': 3.301e23,
    'Venus': 4.867e24,
    'Earth': 5.972e24,
    'Moon': 7.346e22,
    'Mars': 6.417e23,
    'Jupiter Barycenter': 1.899e27,
    'Saturn Barycenter': 5.685e26,
    'Uranus Barycenter': 8.682e25,
    'Neptune Barycenter': 1.024e26
}

class Chandra():
    def __init__(
            self,
            eventsFile,
            ephemerisFile,
            aspectFile,
            gyroFile,
            userData,
            ureg,
            tStartOffset=-0.1
    ):
        self.detector = ChandraDetector(
            eventsFile,
            userData,
            ureg
        )
        
        aspecthdulist = fits.open(aspectFile)
        self.aspectData = aspecthdulist[1]

        gyrohdulist = fits.open(gyroFile)
        self.gyroData = gyrohdulist[1]

        ephemhdulist = fits.open(ephemerisFile)
        self.ephemData = ephemhdulist[1]

        self.tStart = self.detector.getPhotonMeasurement(0)['t']['value'] + tStartOffset

        self.dynamics = ChandraDynamics(
            ephemerisFile,
            aspectFile,
            gyroFile,
            userData,
            ureg,
            self.tStart
            )
            
            

class ChandraDetector():
    def __init__(
            self,
            eventsFile,
            userData,
            ureg
    ):
        # Import Detector Information
        photonHDUList = fits.open(eventsFile)
        self.photonEvents = photonHDUList[1]
        self.photonEventsHeader = photonHDUList[1].header
        
        self.Name = self.photonEventsHeader['detnam']
        
        self.photonXKey = userData.detector.photonCoords.x.value
        self.photonYKey = userData.detector.photonCoords.y.value

        self.photonEnergyKey = userData.detector.energy.key
        self.binsPerEnergy = (
            userData.detector.energy.binPerEnergy.value *
            ureg(userData.detector.energy.binPerEnergy.unit)
        ).to(ureg('kiloelectron_volt')).magnitude
        self.energyIntercept = (userData.detector.energy.intercept)
        

        self.photonEnergyVar = np.square(self.binsPerEnergy)/12.0
        """
        Stores the expected variance of the photon energy measurement.  
        Since photon energy measurements are truncated to integers we use the uniform distribution
        """

        # Extract and store units from header file
        self.photonXUnits = utils.accessPSC.getHeaderInfo(
            self.photonXKey,
            self.photonEventsHeader
        )['unit']
        """
        Units of photon x angle of arrival measurements
        """
        
        self.photonYUnits = utils.accessPSC.getHeaderInfo(
            self.photonXKey,
            self.photonEventsHeader
        )['unit']
        """
        Units of photon y angle of arrival measurements
        """
        
        self.timeOfArrivalUnits = ureg(utils.accessPSC.getHeaderInfo(
            'time',
            self.photonEvents.header
        )['unit'])
        """
        Photon time-of-arrival units
        """

        self.timeConversionFactor = self.timeOfArrivalUnits.to('s').magnitude
        """
        Factor to multiply arrival times by to get correct values in units
        of seconds
        """
        
        self.timeResolution = (
            userData.detector.timeResolution.value *
            ureg(userData.detector.timeResolution.unit)
        ).to(ureg('s')).magnitude
        """
        Photon time of arrival resolution
        """
        
        if userData.detector.TOAstdev.distribution == 'uniform':
            self.TOA_StdDev = self.timeResolution/np.sqrt(12)
            """
            Standard deviation of arrival time measurements.  Depending on user input, this can be modeled as a uniform distribution (over the time resolution of the detector) or as a normal distribution with standard deviation specified by user.
            """
        elif userData.detector.TOAstdev.distribution == 'normal':
            self.TOA_StdDev = (
                userData.detector.TOAstdev.value *
                ureg(userData.detector.TOAstdev.unit)
            ).to(ureg('s')).magnitude
        

        # Get pixel resolution, convert to specified units
        self.pixelResolutionX = (
            userData.detector.pixelResolution.value *
            ureg(userData.detector.pixelResolution.unit)
        ).to(ureg('rad')/ ureg(self.photonXUnits)).magnitude
        """
        Pixel resolution of the detector's x angle of arrival measurements
        """
        self.pixelResolutionY = (
            userData.detector.pixelResolution.value *
            ureg(userData.detector.pixelResolution.unit)
        ).to(ureg('rad')/ ureg(self.photonYUnits)).magnitude
        """
        Pixel resolution of the detector's y angle of arrival measurements
        """
        
        self.FOV = (
            userData.detector.FOV.value *
            ureg(userData.detector.FOV.unit)
        ).to(ureg('deg')).magnitude
        """
        Detector field of view in degrees (half-cone angle)
        """
        self.area = (
            userData.detector.area.value *
            ureg(userData.detector.area.unit)
        ).to(ureg.cm ** 2).magnitude
        """
        Detector effective area in square centimeters
        """

        
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

        

        # Get the detector offset (i.e. how far off the middle pixel is from (0,0)
        # This is based on user input
        self.detectorOffsetX = (
            userData.detector.offsets.x[self.Name].value *
            ureg(userData.detector.offsets.x[self.Name].unit).to(
                self.photonXUnits
            )
        ).magnitude
        self.detectorOffsetY = (
            userData.detector.offsets.y[self.Name].value *
            ureg(userData.detector.offsets.y[self.Name].unit).to(
                self.photonYUnits
            )
        ).magnitude

        # Store variances for measurements in addition to standard deviations
        self.AOA_xVar = np.square(self.AOA_xStdDev)
        self.AOA_yVar = np.square(self.AOA_yStdDev)
        self.TOA_var = np.square(self.TOA_StdDev)

        self.extractedPhotonEvents = {
            self.photonXKey: self.photonEvents.data[self.photonXKey],
            self.photonYKey: self.photonEvents.data[self.photonYKey],
            self.photonEnergyKey: self.photonEvents.data[self.photonEnergyKey],
            'Time': self.photonEvents.data['Time']
            }

        self.photonEventCount = len(self.extractedPhotonEvents['Time'])
        self.targetObject = self.photonEventsHeader['OBJECT']
        photonHDUList.close()

        return
    
    def getPhotonMeasurement(
            self,
            index
    ):
        myRa = (
            (self.extractedPhotonEvents[self.photonXKey][index] - self.detectorOffsetX)
            * self.pixelResolutionX
        )
        myDec = (
            (self.extractedPhotonEvents[self.photonYKey][index] - self.detectorOffsetY)
            * self.pixelResolutionY
        )
        myEnergy = (
            (self.extractedPhotonEvents[self.photonEnergyKey][index] - self.energyIntercept)
            * self.binsPerEnergy
        )

        photonMeasurementDict = {
            't': {
                'value': self.extractedPhotonEvents['Time'][index],
                'var': self.TOA_var
            },
            'RA': {
                'value': -myRa,
                'var': self.AOA_xVar
            },
            'DEC': {
                'value': myDec,
                'var': self.AOA_yVar
            },
            'energy': {
                'value': myEnergy,
                'var': self.photonEnergyVar
            }
        }
        return photonMeasurementDict
        

class ChandraDynamics():
    def __init__(
            self,
            ephemFile,
            aspectFile,
            gyroFile,
            userData,
            ureg,
            tStart
    ):
        self.planetList = ['Sun',
                           'Mercury',
                           'Venus',
                           'Earth',
                           'Moon',
                           'Mars',
                           'Jupiter Barycenter',
                           'Saturn Barycenter',
                           'Uranus Barycenter',
                           'Neptune Barycenter']
        
        self.tStart=tStart
        # Import data
        aspecthdulist = fits.open(aspectFile)
        aspectData = aspecthdulist[1]

        gyrohdulist = fits.open(gyroFile)
        gyroData = gyrohdulist[1]

        ephemhdulist = fits.open(ephemFile)
        ephemData = ephemhdulist[1]

        # Import units from header files for aspect and gyro data, create conversion factors
        self.recordedRAUnits = ureg(
            utils.accessPSC.getHeaderInfo('ra', aspectData.header)['unit']
        )
        self.recordedDECUnits = ureg(
            utils.accessPSC.getHeaderInfo('dec', aspectData.header)['unit']
        )
        self.recordedRollUnits = ureg(
            utils.accessPSC.getHeaderInfo('roll', aspectData.header)['unit']
        )
        
        self.gyroUnits = ureg(
            utils.accessPSC.getHeaderInfo('scratcor', gyroData.header)['unit']
        )

        self.recordedRaConversionFactor = self.recordedRAUnits.to(ureg('rad')).magnitude
        self.recordedDecConversionFactor = self.recordedDECUnits.to(ureg('rad')).magnitude
        self.recordedRollConversionFactor = self.recordedRollUnits.to(ureg('rad')).magnitude
        self.gyroConversionFactor = self.gyroUnits.to(ureg('rad/s')).magnitude
        

        # Import position units and create conversion factors
        self.posX_Units = ureg(utils.accessPSC.getHeaderInfo('X', ephemData.header)['unit'])
        self.posY_Units = ureg(utils.accessPSC.getHeaderInfo('Y', ephemData.header)['unit'])
        self.posZ_Units = ureg(utils.accessPSC.getHeaderInfo('Z', ephemData.header)['unit'])
        
        self.posXConversionFactor = self.posX_Units.to(ureg('km')).magnitude
        self.posYConversionFactor = self.posY_Units.to(ureg('km')).magnitude
        self.posZConversionFactor = self.posZ_Units.to(ureg('km')).magnitude

        # Import velocity units and create conversion factors
        self.vX_Units = ureg(utils.accessPSC.getHeaderInfo('Vx', ephemData.header)['unit'])
        self.vY_Units = ureg(utils.accessPSC.getHeaderInfo('Vy', ephemData.header)['unit'])
        self.vZ_Units = ureg(utils.accessPSC.getHeaderInfo('Vz', ephemData.header)['unit'])

        self.vXConversionFactor = self.vX_Units.to(ureg('km/s')).magnitude
        self.vYConversionFactor = self.vY_Units.to(ureg('km/s')).magnitude
        self.vZConversionFactor = self.vZ_Units.to(ureg('km/s')).magnitude


        # Get reference MJD and time zero
        self.MJDREF = ephemData.header['MJDREF']
        self.timeZero = ephemData.header['TIMEZERO']
        # self.timeUnits = ureg(utils.accessPSC.getHeaderInfo(
        #     'time',
        #     ephemhdulist[0].header
        # )['unit'])

        self.ephemTimeUnits = ureg(
            ephemhdulist[0].header['TIMEUNIT']
        )
        self.eventTimeConversionFactor = self.ephemTimeUnits.to(ureg('day')).magnitude


        # Define a series of interpolation functions to access position,
        # velocity, angles and angular velocity
        self.chandraX = interp1d(ephemData.data['time'],ephemData.data['X'])
        self.chandraY = interp1d(ephemData.data['time'],ephemData.data['Y'])
        self.chandraZ = interp1d(ephemData.data['time'],ephemData.data['Z'])
        self.chandraVX = interp1d(ephemData.data['time'],ephemData.data['vX'])
        self.chandraVY = interp1d(ephemData.data['time'],ephemData.data['vY'])
        self.chandraVZ = interp1d(ephemData.data['time'],ephemData.data['vZ'])

        self.chandraRoll = interp1d(aspectData.data['time'],aspectData.data['roll'])
        self.chandraDEC = interp1d(aspectData.data['time'],aspectData.data['dec'])
        self.chandraRA = interp1d(aspectData.data['time'],aspectData.data['ra'])
        self.chandraOmegaX = interp1d(gyroData.data['time'],gyroData.data['scratcor'][:,0])
        self.chandraOmegaY = interp1d(gyroData.data['time'],gyroData.data['scratcor'][:,1])
        self.chandraOmegaZ = interp1d(gyroData.data['time'],gyroData.data['scratcor'][:,2])

        self.timeObjType = type(self.chandraTimeToTimeScaleObj(self.tStart))
        return
    
    def chandraTimeToTimeScaleObj(
            self,
            chandraTime
    ):
        """
        See http://cxc.harvard.edu/contrib/arots/time/time_tutorial.html
        for information
        """
        return utils.spacegeometry.timeObj.tt_jd(
            2400000.5 +
            self.MJDREF +
            self.timeZero +
            (chandraTime * self.eventTimeConversionFactor)
        )
    
    def position(
            self,
            t
    ):
        # if not isinstance(t,self.timeObjType):
        tsObj = self.chandraTimeToTimeScaleObj(t)
                      
        earthPosition = utils.spacegeometry.earthObj.at(tsObj
        ).position.km

        chandraPositionX = self.chandraX(t) * self.posXConversionFactor
        chandraPositionY = self.chandraY(t) * self.posYConversionFactor
        chandraPositionZ = self.chandraZ(t) * self.posZConversionFactor
        
        chandraPostionSSB = (
            earthPosition +
            [chandraPositionX, chandraPositionY, chandraPositionZ]
        )
        return chandraPostionSSB

    def velocity(
            self,
            t
    ):
        # if not isinstance(t,self.timeObjType):
        tsObj = self.chandraTimeToTimeScaleObj(t)
        earthVelocity = utils.spacegeometry.earthObj.at(
            tsObj
        ).velocity.km_per_s
        
        chandraVelocityX = self.chandraVX(t) * self.vXConversionFactor
        chandraVelocityY = self.chandraVY(t) * self.vYConversionFactor
        chandraVelocityZ = self.chandraVZ(t) * self.vZConversionFactor
        
        chandraVelocitySSB = (
            earthVelocity +
            [chandraVelocityX, chandraVelocityY, chandraVelocityZ]
        )
        return chandraVelocitySSB
    
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
            roll = self.chandraRoll(t) * self.recordedRollConversionFactor
            dec = -self.chandraDEC(t) * self.recordedDecConversionFactor
            ra = self.chandraRA(t) * self.recordedRaConversionFactor

        eulerAngles = [roll, dec, ra]
        if returnQ:
            return utils.euler2quaternion(eulerAngles)
        else:
            return(eulerAngles)

    def omega(
            self,
            t
    ):
        omegaX = self.chandraOmegaX(t) * self.gyroConversionFactor
        omegaY = self.chandraOmegaY(t) * self.gyroConversionFactor
        omegaZ = self.chandraOmegaZ(t) * self.gyroConversionFactor
        return [omegaX, omegaY, omegaZ]

    def acceleration(
            self,
            t
    ):
        G = 6.67408e-11
        position = self.position(t)
        timeObject = self.chandraTimeToTimeScaleObj(t)
        acceleration = np.zeros(3)
        for planetName in self.planetList:

            # Get the current position of the planet
            planetPosition = (
                utils.spacegeometry.planets[planetName].at(timeObject).position.km
            )

            # Compute the distance between spacecraft and planet, and convert
            # to meters
            relativePosition = (planetPosition - position) * 1000.0
            relativeRange = np.linalg.norm(relativePosition)

            # Compute these ahead of time to save computation
            rangePowerMinus3 = np.power(relativeRange, -3)

            # Compute the acceleration due to gravity contributed by the
            # current planet, and add it to the overall acceleration vector
            acceleration = (
                acceleration +
                (G * planetaryMasses[planetName] *
                 rangePowerMinus3) * relativePosition / 1000.0
            )
        return acceleration

