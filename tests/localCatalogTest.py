from astropy.io import fits
import modest as md
import yoctopy as yp
fluxKey = 'FLUX_B8'
catalogName = 'xmmsl2_clean.fits'

dirpath = '/home/joel/Documents/pythonDev/modules/ModularFilter/modest/utils/'
    
hdulist = fits.open(dirpath + catalogName)

catalogHeader = hdulist[1].header
catalogData = hdulist[1].data
hdulist.close()

columns = ['UNIQUE_SRCNAME', 'RA', 'DEC', fluxKey]
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

if columns:
    raise ValueError('Did not find columns %s in local catalog.' %columns)
RaUnit = yp.Unit(columnIndexDict['RA']['unit'])
DecUnit = yp.Unit(columnIndexDict['DEC']['unit'])
for source in catalogData:
    sourceUnitVector = md.utils.spacegeometry.sidUnitVec(
        source['RA'] * RaUnit,
        source['DEC'] * DecUnit
        )
