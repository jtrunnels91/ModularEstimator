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
            
            photonFlux = (
                pulsarRow['Flux (erg/cm^2/s)'] *
                electronVoltPerErg / electronVoltPerPhoton
                ) * detectorArea

            if np.isnan(pulsarRow['useColumn']):
                useColumn=None
            else:
                useColumn = pulsarRow['useColumn']
            
            pulsarDict[pulsarName] = signals.PeriodicXRaySource(
                pulsarDir + profileDir + pulsarRow['Template'],
                PARFile=pulsarDir + PARDir + pulsarRow['PARFile'],
                avgPhotonFlux=photonFlux,
                pulsedFraction=pulsarRow['Pulsed fraction']/100,
                name=pulsarName,
                useProfileColumn=useColumn,
                observatoryMJDREF=observatoryMJDREF
            )
    return pulsarDict
