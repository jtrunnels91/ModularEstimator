import pandas as pd
from .. import signals
import numpy as np
import os
import sys


def loadPulsarData(
        detectorArea=1,
        loadPulsarNames=None,
        pulsarDir=None,
        pulsarCatalogFileName='pulsarCatalog.xls',
        PARDir='PAR_files/',
        profileDir='profiles/',
        observatoryMJDREF=None,
        energyRange=None #Should be in KEV for now
):
    if pulsarDir is None:
        pulsarDir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) + '/'
    if energyRange is None:
        electronVoltPerPhoton = 6e3  # Electron-Volt x 10^3
    else:
        electronVoltPerPhoton = 1e3 * (energyRange[1] + energyRange[0])/2
    electronVoltPerErg = 6.242e11
    ergsPerElectronVolt = 1 / electronVoltPerErg
    
    pulsarCatalog = pd.read_excel(pulsarDir + pulsarCatalogFileName)

    pulsarDict = {}

    for pulsarIterator in range(len(pulsarCatalog)):
        pulsarRow = pulsarCatalog.iloc[pulsarIterator]
        pulsarName = pulsarRow['Name']

        if (loadPulsarNames is None) or (pulsarName in loadPulsarNames):
            if not np.isnan(pulsarRow['Flux (erg/cm^2/s)']):
                photonFlux = (
                    pulsarRow['Flux (erg/cm^2/s)'] *
                    electronVoltPerErg / electronVoltPerPhoton
                ) 
            else:
                photonFlux = None
                
            if np.isnan(pulsarRow['useColumn']):
                useColumn=None
            else:
                useColumn = pulsarRow['useColumn']

            if not np.isnan(pulsarRow['Pulsed fraction']):
                pulsedFraction = pulsarRow['Pulsed fraction']/100
            else:
                pulsedFraction = None
            # print("Template string:")
            # print(pulsarRow['Template'])
            if isinstance(pulsarRow['Template'], str) or not np.isnan(pulsarRow['Template']):
                template = pulsarDir + profileDir + pulsarRow['Template']
            else:
                template=None
            
            pulsarDict[pulsarName] = signals.PeriodicXRaySource(
                profile=template,
                PARFile=pulsarDir + PARDir + pulsarRow['PARFile'],
                avgPhotonFlux=photonFlux,
                pulsedFraction=pulsedFraction,
                name=pulsarName,
                useProfileColumn=useColumn,
                observatoryMJDREF=observatoryMJDREF,
                detectorArea=detectorArea
            )
    return pulsarDict
