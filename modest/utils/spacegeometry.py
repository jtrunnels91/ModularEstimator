import numpy as np
# Import module which will give us information about the motion of Earth
# around the SSB
from skyfield.api import load
import warnings
import sys
if 'sphinx' in sys.modules:
    planets = None
    timeObj = None
    earthObj = None
else:
    planets = load('de421.bsp')
    timeObj = load.timescale()
    earthObj = planets['earth']

planetaryMasses = { # in kg
    'Sun': 1.98855e30,
    'Mercury': 3.301e23,
    'Venus': 4.867e24,
    'Earth': 5.972e24,
    'Moon': 7.346e22,
    'Mars': 6.417e23,
    'Jupiter Barycenter': 1.899e27,
    'Saturn Barycenter': 5.685e26,
    'Uranus Barycenter': 8.682e25,
    'Neptune Barycenter': 1.024e26,
    'Pluto Barycenter': 1.309e22
}

planetaryRadius = { # in KM
    'Sun': 695510,
    'Mercury': 2439,
    'Venus': 6051,
    'Earth': 6378,
    'Moon': 1738,
    'Mars': 3396,
    'Jupiter Barycenter': 71492,
    'Saturn Barycenter': 60268,
    'Uranus Barycenter': 25559,
    'Neptune Barycenter': 24764,
    'Pluto Barycenter': 1195
}


G = 6.67408e-11

    
def phaseError(estDelay, trueDelay, period):
    if hasattr(estDelay, '__len__'):
        error = np.zeros(len(estDelay))
        if hasattr(trueDelay, '__len__'):
            for i in range(len(error)):
                error[i] = phaseError(estDelay[i], trueDelay[i], period)
        else:
            for i in range(len(error)):
                error[i] = phaseError(estDelay[i], trueDelay, period)
    else:
        nIter = 0
        error = estDelay-trueDelay
        if error > period/2:
            while (error > period/2) and nIter < 100:
                nIter += 1
                error = error - period
        elif (error < -period/2):
            while (error < - period/2) and nIter < 100:
                nIter += 1
                error = error + period

        if (
                (error < -period/2) or
                (error > period/2)
        ):
            warnings.warn("Unable to wrap phase")
    return error

    
def deg2rad(deg):
    return deg * np.pi / 180.0


def rad2deg(rad):
    return rad * 180.0 / np.pi


def hms2rad(h=None, m=None, s=None, hms=None):
    if hms is not None:
        if isinstance(hms, str):
            hms = str.split(hms)
            h = float(hms[0])
            m = float(hms[1])
            s = float(hms[2])
        else:
            h = hms[0]
            m = hms[1]
            s = hms[2]
            
    if h > 0:
        hours = h + m / 60.0 + s / 3600.0
    else:
        hours = np.abs(h) + m / 60.0 + s / 3600.0
        hours = - hours
    
    return 2.0 * np.pi * hours / 24.0

def rad2hms(rad):
    """
    This function converts radians to hours, minutes and seconds
    
    Args:
      rad (float): radians to be converted.  Will be made positive if less than zero (by adding 2 pi)

    Returns:
      (dict): A dict containing "hours", "minutes" and "seconds" key-value pairs
    """
    while rad < 0:
        rad = rad + (np.pi*2)
    hours = rad * 12.0/np.pi
    hoursRemainder = hours % 1
    hours = hours - hoursRemainder

    minutes = hoursRemainder * 60.0
    minutesRemainder = minutes % 1
    minutes = minutes - minutesRemainder

    seconds = minutesRemainder * 60.0

    return {'hours': hours, 'minutes': minutes, 'seconds': seconds}

def rad2dms(rad):
    """
    This function converts radians to degrees, minutes and seconds

    Args:
      rad (float): radians to be converted.  

    Returns:
      (dict): A dict containing "degrees", "minutes" and "seconds" key-value pairs
    
    """
    degrees = rad * 180.0/np.pi
    degreesRemainder = np.abs(degrees) % 1
    degrees =  (np.abs(degrees) - degreesRemainder) * np.sign(degrees)

    minutes = degreesRemainder * 60.0
    minutesRemainder = minutes % 1
    minutes = minutes - minutesRemainder

    seconds = minutesRemainder * 60.0

    return {'degrees': degrees, 'minutes': minutes, 'seconds': seconds}

def dms2rad(d=None, m=None, s=None, dms=None):
    if dms is not None:
        if isinstance(dms, str):
            dms = str.split(dms)
            d = float(dms[0])
            m = float(dms[1])
            s = float(dms[2])
        else:
            d = dms[0]
            m = dms[1]
            s = dms[2]
            
    if d > 0:
        degrees = d + (m / 60.0) + (s / 3600.0)
    else:
        degrees = np.abs(d) + (m / 60.0) + (s / 3600.0)
        degrees = - degrees
    return np.pi * degrees / 180.0

def unitVector2RaDec(unitVector):
    D = np.arcsin(unitVector[2])
    cosD = np.cos(D)
    cosRA = unitVector[0]/cosD
    sinRA = unitVector[1]/cosD

    RA = np.arctan2(sinRA, cosRA)

    return(RA, D)

def sidUnitVec(RA, DEC):
    cosD = np.cos(DEC)
    sinD = np.sin(DEC)
    cosRA = np.cos(RA)
    sinRA = np.sin(RA)

    # if isinstance(sinD, yp.uscalar):
    #     return yp.uarray([cosD * cosRA, cosD * sinRA, sinD])
    # else:
    return np.array([cosD * cosRA, cosD * sinRA, sinD])


def getUTC(startTime, ellapsedSeconds, verbose=False):

    if isinstance(startTime, dict):
        mySecond = startTime['second'] + ellapsedSeconds
        myMinute = startTime['minute']
        myHour = startTime['hour']
        myDay = startTime['day']

        if mySecond > 59:
            remainder = np.mod(mySecond, 60)
            myMinute = myMinute + (mySecond - remainder) / 60.0
            mySecond = remainder

        if myMinute > 59:
            remainder = np.mod(myMinute, 60)
            myHour = myHour + (myMinute - remainder) / 60.0
            myMinute = remainder

        if myHour > 23:
            remainder = np.mod(myHour, 24)
            myDay = myDay + (myHour - remainder) / 24.0
            myHour = remainder

        myUTC = timeObj.utc(startTime['year'],
                            startTime['month'],
                            myDay,
                            myHour,
                            myMinute,
                            mySecond)
    else:
        myUTC = timeObj.tt(jd=2400000.5 + startTime)
    if verbose is True:
        print('Days = %i' % myDay)
        print('Hours = %i' % myHour)
        print('Minutes = %i' % myMinute)
        print('Seconds = %i' % mySecond)

    return myUTC


def earthPosition(startTime,
                  ellapsedSeconds,
                  verbose=False):

    myUTC = getUTC(startTime, ellapsedSeconds, verbose)

    return earthObj.at(myUTC).position.km


def earthVelocity(startTime,
                  ellapsedSeconds,
                  verbose=False):

    myUTC = getUTC(startTime, ellapsedSeconds, verbose)
    return earthObj.at(myUTC).velocity.km_per_s


########################################################
#
#  MISC UTILITY FUNCTIONS
#
#  Not nescessarily space geometry functions
#
########################################################

# Simple function to convert Suzaku Time to MJD
def suzakuTime2MJD(suzakuTime, MJDREFI=None, MJDREFF=None, fineClock=None):

    if MJDREFI is None and MJDREFF is None:
        return 51544 + suzakuTime / (24.0 * 60.0 * 60.0)
    else:
        if fineClock is None:
            fineClock = 0
            return (MJDREFI + MJDREFF) + (suzakuTime + fineClock) / 86400.0


# Simple function to convert SWIFT time to MJD
# See: https://swift.gsfc.nasa.gov/analysis/suppl_uguide/time_guide.html
def swiftTime2MJD(swiftTime, MJDREFI=None, MJDREFF=None, UTCF=None):

    if MJDREFI is None:
        MJDREFI = 51910
    if MJDREFF is None:
        MJDREFF = 7.4287037E-4
    if UTCF is None:
        UTCF = 0
    return MJDREFF + MJDREFI + (swiftTime + UTCF) / (24.0 * 60.0 * 60.0)


def swiftTime2JD(swiftTime, MJDREFI=None, MJDREFF=None, UTCF=None):
    return (
        2400000.5 +
        swiftTime2MJD(
            swiftTime,
            MJDREFI=MJDREFI,
            MJDREFF=MJDREFF,
            UTCF=UTCF)
    )

# Function to search through data for large gaps, and return the set
# divided into "slices."  These slices usually correspond to individual
# orbits around the sun.


def sliceData(sortedTimeSeries, threshold=100):
    sliceIndex = np.where(np.diff(sortedTimeSeries) > threshold)

    sliceIndex = sliceIndex[0]
    print(len(sliceIndex))

    if len(sliceIndex > 0):
        slicedData = [sortedTimeSeries[0:sliceIndex[0]]]

    else:
        slicedData = [sortedTimeSeries]

    if (len(sliceIndex) > 1):
        for i in range(len(sliceIndex)):
            if i > 0 and i < len(sliceIndex):
                slicedData.insert(i, sortedTimeSeries[
                                  sliceIndex[i - 1] + 1:sliceIndex[i]])

        slicedData.insert(len(sliceIndex), sortedTimeSeries[
                          sliceIndex[-1] + 1:])

    return(slicedData)


def sigmaDeltaT_Theoretical(period,
                            flux,
                            pulsedFraction,
                            pulseWidth,
                            detectorArea,
                            tObs,
                            backgroundFlux=0):

    dutyCycle = pulseWidth / period

    SNR = ((flux * detectorArea * pulsedFraction * tObs)
           /
           np.sqrt(
               (backgroundFlux + flux * (1 - pulsedFraction)) * (detectorArea * tObs * dutyCycle) +
               flux * detectorArea * pulsedFraction * tObs
    )
    )
    # SNR = ((flux * detectorArea * pulsedFraction * tObs)
    #        /
    #        np.sqrt(
    #            (backgroundFlux + flux * (1 - pulsedFraction)) * (detectorArea * tObs) +
    #            flux * detectorArea * pulsedFraction * tObs * dutyCycle
    # )
    # )

    return pulseWidth / (2 * SNR)

    

def acceleration(position, tCurrent=None):
    if not tCurrent:
        tCurrent = timeObj.now()
    acceleration = np.zeros(3)
    for planetName in planetaryMasses.keys():
        # Get the current position of the planet
        planetPosition = (
            planets[planetName].at(tCurrent).position.km
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


def accelerationGradient(position, tCurrent=None):
    if tCurrent == None:
        tCurrent = timeObj.now()
    aGrad = np.zeros([3,3])
    
    for planetName in planetaryMasses.keys():
        GMCurrent = planetaryMasses[planetName] * G
        # Get the current position of the planet
        planetPosition = (
            planets[planetName].at(tCurrent).position.km
        )

        # Compute the distance between spacecraft and planet, and convert
        # to meters
        relativePosition = (position - planetPosition) * 1000.0
        relativeRange = np.linalg.norm(relativePosition)

        # Compute these ahead of time to save computation
        rangePowerMinus3 = np.power(relativeRange, -3)
        rangePowerMinus5 = np.power(relativeRange, -5)

        aGrad[0, 0] += (
            (3 * np.square(relativePosition[0]) * rangePowerMinus5) - rangePowerMinus3
        ) * GMCurrent
        aGrad[0, 1] += (
            3 * relativePosition[0]*relativePosition[1] * rangePowerMinus5
        ) * GMCurrent
        aGrad[0, 2] += (
            3 * relativePosition[0]*relativePosition[2] * rangePowerMinus5
        ) * GMCurrent

        aGrad[1, 1] += (
            (3 * np.square(relativePosition[1]) * rangePowerMinus5) - rangePowerMinus3
        ) * GMCurrent
        aGrad[1, 2] += (
            (3 * relativePosition[1]*relativePosition[2] * rangePowerMinus5)
        ) * GMCurrent

        aGrad[2, 2] += (
            (3 * np.square(relativePosition[2]) * rangePowerMinus5) - rangePowerMinus3
        ) * GMCurrent

    aGrad[1, 0] = aGrad[0, 1]
    aGrad[2, 0] = aGrad[0, 2]
    aGrad[2, 1] = aGrad[1, 2]

    aGrad = aGrad/1000
    return aGrad
