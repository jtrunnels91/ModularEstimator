from astropy.io import fits
from pint import UnitRegistry
import numpy as np
from .. import utils

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


        # Extract and store units from header file
        self.photonXUnits = utils.accessPSC.getHeaderInfo(
            self.photonXKey,
            self.photonEventsHeader
        )['unit']
        
        self.photonYUnits = utils.accessPSC.getHeaderInfo(
            self.photonXKey,
            self.photonEventsHeader
        )['unit']
        
        self.timeOfArrivalUnits = ureg(utils.accessPSC.getHeaderInfo(
            'time',
            self.photonEvents.header
        )['unit'])

        # Factor to multiply arrival times by to get correct values in units
        # of seconds
        self.timeConversionFactor = self.timeOfArrivalUnits.to('s').magnitude
        
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
        

        # Get pixel resolution, convert to specified units
        self.pixelResolutionX = (
            userData.detector.pixelResolution.value *
            ureg(userData.detector.pixelResolution.unit)
        ).to(ureg('rad')/ ureg(self.photonXUnits)).magnitude
        self.pixelResolutionY = (
            userData.detector.pixelResolution.value *
            ureg(userData.detector.pixelResolution.unit)
        ).to(ureg('rad')/ ureg(self.photonYUnits)).magnitude
        
        self.FOV = (
            userData.detector.FOV.value *
            ureg(userData.detector.FOV.unit)
        ).to(ureg('rad')).magnitude

        
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
            ureg
    ):
    
