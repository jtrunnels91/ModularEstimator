import numpy as np
from .. import substates, signals, utils

def buildPulsarModel(
        traj,
        mySpacecraft,
        ureg
):
    # Load detector data needed to build signal models
    detectorArea = mySpacecraft.detector.area
    extentConversionFactor = (
        traj.filesAndDirs.sourceCatalogPixelSize.value *
        ureg(traj.filesAndDirs.sourceCatalogPixelSize.unit)
    )

    # Load pulsar objects, and add the correct one to the filter
    pulsarObjectDict = utils.loadPulsarData(
        detectorArea=detectorArea,
        pulsarDir=traj.filesAndDirs.baseDirectory.value,
        pulsarCatalogFileName=traj.filesAndDirs.pulsarDataFile.value,
        PARDir=traj.filesAndDirs.ParFileDirectory.value,
        profileDir=traj.filesAndDirs.profileDirectory.value,
        observatoryMJDREF=mySpacecraft.dynamics.MJDREF
    )

    try:
        try:
            myPulsarObject = pulsarObjectDict[
                mySpacecraft.detector.targetObject.strip('PSR').strip(' ')
            ]
        except:
            myPulsarObject = pulsarObjectDict[
                traj.filesAndDirs.tarjetObject.value
            ]
    except:
        myPulsarObject = pulsarObjectDict[
            traj.simulation.pulsarName.value
        ]
    
    myPulsarObject.lastTime = mySpacecraft.tStart
    if traj.attitudeFilter.probabilityMeasMat.value == 'unitVec':
        myPulsarObject.useUnitVector = True
    else:
        myPulsarObject.useUnitVector = False
        
    if not np.any(myPulsarObject.extent):
        myPulsarObject.extent = (
            1*ureg('pixel') * extentConversionFactor
        ).to(ureg('rad')).magnitude
        
    return myPulsarObject

def buildStaticSources(
        traj,
        mySpacecraft,
        ureg
):
    startingAttitude = mySpacecraft.dynamics.attitude(mySpacecraft.tStart,returnQ=False)
    startingRA = startingAttitude[2]
    startingDEC = -startingAttitude[1]
    
    # Load other nearby point sources from selected catalog
    myFluxKey = traj.filesAndDirs.pointSourceFluxKey.value
    myExtentKey = traj.filesAndDirs.pointSourceExtentKey.value
    myRaKey = traj.filesAndDirs.raKey.value
    myDecKey = traj.filesAndDirs.decKey.value
    mySrcKey = traj.filesAndDirs.srcNameKey.value
    pointSources = utils.accessPSC.localCatalog_coneSearch(
        RA={'value': startingRA, 'unit': 'rad'},
        DEC={'value': startingDEC, 'unit': 'rad'},
        FOV={'value': mySpacecraft.detector.FOV, 'unit': 'degrees'},
        catalogName=traj.filesAndDirs.pointSourceCatalog.value,
        removeNaNs=False,
        fluxKey=myFluxKey,
        extentKey=myExtentKey,
        raKey=myRaKey,
        decKey=myDecKey,
        srcNameKey=mySrcKey,
    )

    extentConversionFactor = (
        traj.filesAndDirs.sourceCatalogPixelSize.value *
        ureg(traj.filesAndDirs.sourceCatalogPixelSize.unit)
    )
    # Create signal objects for those point sources and add them to the filter
    pointSourceObjectDict = {}
    for signalIndex in range(len(pointSources)):
        myRow = pointSources.iloc[signalIndex]
        myName = myRow[mySrcKey]
        myRa = (
            myRow[myRaKey]['value'] * ureg(myRow[myRaKey]['unit'])
        ).to(ureg.rad).magnitude
        myDec = (
            myRow[myDecKey]['value'] * ureg(myRow[myDecKey]['unit'])
        ).to(ureg.rad).magnitude

        myFlux = (
            myRow[myFluxKey]['value'] * ureg(myRow[myFluxKey]['unit'])
        ).to(ureg('erg/s/cm^2')).magnitude
        myExtent = myRow[myExtentKey]['value']
        print("Extent = %s" %myExtent)
        if myExtent == 0 or np.isnan(myExtent):
            myExtent = 1
        myExtent = myExtent * ureg(myRow[myExtentKey]['unit'].replace('ima_pix','pixel'))
        myExtent = (
            myExtent * extentConversionFactor
        ).to(ureg('rad')).magnitude
            
        # Check to make sure that the flux is in a valid range
        if myFlux > 1e-15 and myFlux < 1e10:  # flux validation
            print('Initializing static point source %s.' %myName)
            if traj.attitudeFilter.probabilityMeasMat.value == 'unitVec':
                useUnitVec = True
            else:
                useUnitVec = False

            pointSourceObjectDict[myName] = (
                signals.StaticXRayPointSource(
                    myRa,
                    myDec,
                    photonEnergyFlux=myFlux,
                    detectorArea=mySpacecraft.detector.area,
                    name=myName,
                    startTime=mySpacecraft.tStart,
                    extent=myExtent,
                    useUnitVector=useUnitVec
                )
            )
    return pointSourceObjectDict
    
