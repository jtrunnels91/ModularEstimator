from astropy.io import fits
from context import modest as md
import numpy as np
import pandas as pd

myDF = md.utils.accessPSC.localCatalog_coneSearch(
    RA={'value':0,'unit':'rad'},
    DEC={'value':0,'unit':'rad'},
    FOV={'value':10,'unit':'degrees'}
    )
referenceRA = 0
referenceRAUnits = 'rad'

referenceDec = 0
referenceDecUnits = 'rad'

FOV = 8
FOVUnits = 'degrees'

fluxKey = 'FLUX_B8'
catalogName = 'xmmsl2_clean.fits'

dirpath = '/home/joel/Documents/pythonDev/research/pulsarJPDAF/pulsarData/xray_catalogs/'

hdulist = fits.open(dirpath + catalogName)

catalogHeader = hdulist[1].header
catalogData = hdulist[1].data
hdulist.close()

columns = ['UNIQUE_SRCNAME', 'RA', 'DEC', fluxKey]
savedColumns = []
columnIndexDict = {}
catKeys = list(catalogHeader.keys())
for index in range(len(catalogHeader)):
    for column in columns:
        if column == catalogHeader[index]:
            catKey = catKeys[index]
            unitKey = catKey.replace('TYPE', 'UNIT')
            if unitKey in catalogHeader:
                columnUnit = catalogHeader[unitKey]
            else:
                columnUnit = None
            columnIndexDict[column] = {
                'index': index,
                'key': catKey
            }
            if columnUnit:
                columnIndexDict[column]['unit'] = columnUnit

            columns.remove(column)
            savedColumns.append(column)

if columns:
    raise ValueError('Did not find columns %s in local catalog.' %columns)

if columnIndexDict['RA']['unit'] == 'rad':
    raConversionFactor = 1
elif columnIndexDict['RA']['unit'] == 'degrees':
    raConversionFactor = np.pi / 180.0
if columnIndexDict['DEC']['unit'] == 'rad':
    decConversionFactor = 1
elif columnIndexDict['DEC']['unit'] == 'degrees':
    decConversionFactor = np.pi/180.0
    
if referenceRAUnits == 'rad':
    referenceRA = referenceRA
elif referenceRAUnits == 'degrees':
    referenceRA = referenceRA * np.pi / 180.0
else:
    raise ValueError('Unrecougnized RA units %s' % referenceRAUnits)
if referenceDecUnits == 'rad':
    referenceDec = referenceDec
elif referenceDecUnits == 'degrees':
    referenceDec = referenceDec * np.pi / 180.0
else:
    raise ValueError('Unrecougnized Dec units %s' % referenceDecUnits)

if FOVUnits == 'rad':
    FOV = FOV
elif FOVUnits == 'degrees':
    FOV = FOV * np.pi / 180.0
else:
    raise ValueError('Unrecougnized FOV units %s' % FOVUnits)

referenceUnitVector = md.utils.spacegeometry.sidUnitVec(
    referenceRA,
    referenceDec
)
mySourceDF = pd.DataFrame(columns=savedColumns)
for source in catalogData:
    sourceUnitVector = md.utils.spacegeometry.sidUnitVec(
        source['RA'] * raConversionFactor,
        source['DEC'] * decConversionFactor
        )
    angularDiff = np.arccos(referenceUnitVector.dot(sourceUnitVector))

    if angularDiff < (FOV/2):
        mySrcDict = {}
        for key in savedColumns:
            mySrcDict[key] = source[key]
        mySourceDF = mySourceDF.append(mySrcDict, ignore_index=True)
