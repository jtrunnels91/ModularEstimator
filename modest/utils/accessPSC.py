import requests
import pandas as pd
from tempfile import NamedTemporaryFile
import os
import subprocess
from astropy.io import fits


def localCatalog_coneSearch(
        RA,
        DEC,
        FOV,
        catalogName = 'xmmsl2_clean.fits',
        removeNullFlux=True,
        fluxKey='flux'
        ):
    dirpath = '/home/joel/Documents/pythonDev/modules/ModularFilter/modest/utils/'
    
    hdulist = fits.open(dirpath + catalogName)

    dataHeader = hdulist[1].header
    columns = ['UNIQUE_SRCNAME', 'RA', "DEC"]
    return hdulist

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
