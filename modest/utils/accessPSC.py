import requests
import pandas as pd
import numpy as np
from tempfile import NamedTemporaryFile
import os
import subprocess
from astropy.io import fits
from . import spacegeometry

def localCatalog_coneSearch(
        RA,
        DEC,
        FOV,
        catalogName='xmmsl2_clean.fits',
        dirpath='/home/joel/Documents/pythonDev/research/pulsarJPDAF/pulsarData/xray_catalogs/',
        removeNaNs=True,
        fluxKey='FLUX_B8'
        ):

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

    if RA['unit'] == 'rad':
        referenceRA = RA['value']
    elif RA['unit'] == 'degrees':
        referenceRA = RA['value'] * np.pi / 180.0
    else:
        raise ValueError('Unrecougnized RA units %s' % RA['unit'])
    
    if DEC['unit'] == 'rad':
        referenceDec = DEC['value']
    elif DEC['unit'] == 'degrees':
        referenceDec = DEC['value'] * np.pi / 180.0
    else:
        raise ValueError('Unrecougnized Dec units %s' % DEC['unit'])

    if FOV['unit'] == 'rad':
        FOVVal = FOV['value']
    elif FOV['unit'] == 'degrees':
        FOVVal = FOV['value'] * np.pi / 180.0
    else:
        raise ValueError('Unrecougnized FOV units %s' % FOV['unit'])

    referenceUnitVector = spacegeometry.sidUnitVec(
        referenceRA,
        referenceDec
    )
    mySourceDF = pd.DataFrame(columns=savedColumns)
    for source in catalogData:
        sourceUnitVector = spacegeometry.sidUnitVec(
            source['RA'] * raConversionFactor,
            source['DEC'] * decConversionFactor
            )
        angularDiff = np.arccos(referenceUnitVector.dot(sourceUnitVector))

        if angularDiff < (FOVVal/2):
            mySrcDict = {}
            skipVal = False
            for columnName, columnInfo in columnIndexDict.items():
                if not skipVal:
                    if 'unit' in columnInfo:
                        mySrcDict[columnName] = {
                            'value': source[columnName],
                            'unit': columnInfo['unit'].replace('cm2', 'cm^2')
                        }
                    else:
                        mySrcDict[columnName] = source[columnName]
                    if removeNaNs:
                        try:
                            skipVal = np.isnan(source[columnName])
                        except:
                            skipVal = False
            if not skipVal:
                mySourceDF = mySourceDF.append(mySrcDict, ignore_index=True)
    return mySourceDF

def xamin_coneSearch(
        RA,
        DEC,
        FOV,
        angleUnits='degrees',
        catalog='xray',
        removeNullFlux=True,
        fluxKey='flux'
        ):
    if angleUnits == 'degrees':
        FOVArcmin = FOV * 60
    elif angleUnits == 'radians':
        FOVArcmin = FOV * 3437.75
    elif angleUnits == 'arc':
        FOVArcmin = FOV
    dirpath = '/home/joel/Documents/pythonDev/modules/ModularFilter/modest/utils'
    fieldCommand = 'fields=name,ra,dec,%s' % fluxKey
    myCommand = ['java',
                 '-jar',
                 dirpath + '/users.jar',
                 'table=%s' %catalog,
                 'position=\'%s, %s\'' % (RA, DEC),
                 'radius=%s' % FOVArcmin,
                 fieldCommand]
    print(myCommand)
    # myQuery += ('table=%s' % catalog)
    # myQuery += ('position=\'%s, %s\'' % (RA, DEC))
    # myQuery += ('radius=%s' % FOV)
    
    # subprocess.call(['java', '-jar', 'users.jar'], env=env)
    # process = subprocess.Popen(['java', '-jar', 'users.jar'], stdout=subprocess.PIPE)
    process = subprocess.Popen(myCommand, stdout=subprocess.PIPE)
    output = process.stdout
    print(output)
    outputDF = pd.read_csv(output, sep='|', comment='#').dropna(how='any')
    outputDF.columns = outputDF.columns.str.strip()
    outputDF = outputDF.rename(columns={str.lower(fluxKey):'flux'})

    print(outputDF)
    for row in range(len(outputDF)):
        try:
            outputDF.set_value(row, 'flux', outputDF.loc[row]['flux'])
        except:
            if removeNullFlux is True:
                outputDF.drop(row, inplace=True)
                # print('Dropping row %i' %(row))

    outputDF.reset_index()
    
    
    return(outputDF)

def chandraPSC_coneSearch(
        RA,
        DEC,
        FOV,
        FOVUnits='degrees',
        minSignificance=0
        ):
    
    if FOVUnits == 'degrees':
        FOVArcmin = FOV * 60
    elif FOVUnits == 'radians':
        FOVArcmin = FOV * 3437.75
    elif FOVUnits == 'arcmins':
        FOVArcmin = FOV
    else:
        raise ValueError('Unrecougnized unit for FOV.  Use either degrees, radians, or arcmins.')
        
    baseQuery=(
        'http://cda.cfa.harvard.edu/csccli/getProperties?query='
        'SELECT m.name, m.ra, m.dec, m.flux_aper_b, m.significance ' +
        'FROM master_source m ' +
        'WHERE (' +
        'dbo.cone_distance(m.ra,m.dec,%s,%s)<=%s'
        %(RA, DEC, FOVArcmin)
    )
    if minSignificance > 0:
        baseQuery = (
            baseQuery +
            'AND m.significance > %s)'
            %minSignificance
            )
    else:
        baseQuery = baseQuery + ')'
    print(baseQuery)
    response=requests.get(baseQuery)
    # t = TemporaryFile()
    # with open('./tmp', 'wb') as f:  
    #     f.write(response.content)

    with NamedTemporaryFile(mode='wb', delete=False) as f:  
        f.write(response.content)
    resultsDF = pd.read_csv(f.name, sep='\t', comment='#')

    f.close()
    os.remove(f.name)
    return(resultsDF)
