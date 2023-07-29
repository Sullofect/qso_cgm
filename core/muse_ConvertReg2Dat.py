import glob
import coord
import numpy as np
from astropy.io import ascii
from regions import Regions
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord

# path_reg = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/'
# rel_labels = Regions.read(path_reg, format='ds9')
# x = rel_labels[0].center.ra.degree
# y = rel_labels[0].center.dec.degree
# print(x, y)
#
# #
# c = SkyCoord(ra=x*u.degree, dec=y*u.degree, frame='icrs')
# print(c.to_string('hmsdms'))

def Convert(cubename=None):
    rel_labels = Regions.read('/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/' + cubename + '_ESO-DEEP_ZAP_JL_HST.reg',
                              format='ds9')
    ID_labels = np.loadtxt('/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/' + cubename + '_ESO-DEEP_ZAP_final.reg',
                           usecols=[3], delimiter=',', dtype=object)
    print(ID_labels)

    # Define
    row_array = np.arange(len(rel_labels)) + 1
    ra_array = np.zeros(len(rel_labels))
    dec_array = np.zeros(len(rel_labels))
    name_array = np.zeros(len(rel_labels), dtype=object)
    radius_array = np.zeros(len(rel_labels))

    #
    for j in range(len(rel_labels)):
        ra_array[j] = rel_labels[j].center.ra.degree
        dec_array[j] = rel_labels[j].center.dec.degree
        radius_array[j] = rel_labels[j].radius.value
        ra_string, dec_string, name_string = coord.coordstring(ra_array[j], dec_array[j])
        name_array[j] = name_string[0]

    # Make table
    table = Table()
    table['row'] = row_array
    table['id'] = ID_labels
    table['name'] = name_array
    table['ra'] = np.round(ra_array, 6)
    table['dec'] = np.round(dec_array, 6)
    table['radius'] = radius_array

    table.write('/Users/lzq/Dropbox/MUSEQuBES+CUBS/dats/' + cubename + '_ESO-DEEP_ZAP.dat',
                format='ascii.fixed_width', overwrite=True)

#
filenames = glob.glob('/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/*_JL_HST.reg')
for i, filename_i in enumerate(filenames):
    print(filename_i)
    filename_i = filename_i[44:-24]
    Convert(filename_i)



# filenames = glob.glob('/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/*_JL_HST.reg')
# for i, filename_i in enumerate(filenames[:1]):
#     rel_labels = Regions.read(filename_i, format='ds9')
#
#     row_array = np.arange(len(rel_labels)) + 1
#     ra_array = np.zeros(len(rel_labels))
#     dec_array = np.zeros(len(rel_labels))
#     name_array = np.zeros(len(rel_labels), dtype=object)
#     radius_array = np.zeros(len(rel_labels))
#
#
#
#     for j in range(len(rel_labels)):
#         ra_array[j] = rel_labels[j].center.ra.degree
#         dec_array[j] = rel_labels[j].center.dec.degree
#         radius_array[j] = rel_labels[j].radius.value
#
#         ra_string, dec_string, name_string = coord.coordstring(ra_array[j], dec_array[j])
#         print(name_string)
#
#         c = SkyCoord(ra=ra_array[j] * u.degree, dec=dec_array[j] * u.degree, frame='fk5')
#         name = c.to_string('hmsdms')
#         name = name.replace(' ', '')
#         name = name.replace('h', '')
#         name = name.replace('d', '')
#         name = name.replace('m', '')
#         name = name.replace('s', '')
#         name_array[j] = name
#         print(name)
#         # print(c.dec.dms.d)
#         # ra_hms = "{:.0f}".format(c.ra.hms.h) + "{:.0f}".format(c.ra.hms.m) + "{:.2f}".format(c.ra.hms.s)
#         # dec_dms = "{:+03d}".format(int("{:.0f}".format(c.dec.dms.d))) + "{:.0f}".format(c.dec.dms.m) + "{:.2f}".format(c.dec.dms.s)
#         # name_array[j] = 'J' + ra_hms + dec_dms
#         # print(name_array[j])
#     # dec_dms = c.dec.dms
#     # name = c.to_string('hmsdms')
#     # print(name)
#
#
#     table = Table()
#     table['row'] = row_array
#     table['id'] = row_array
#     table['name'] = name_array
#     table['ra'] = np.round(ra_array, 6)
#     table['dec'] = np.round(dec_array, 6)
#     table['radius'] = radius_array
#     table.pformat(align='<')
#
#     table.write('/Users/lzq/Dropbox/MUSEQuBES+CUBS/dats/' + filename_i[44:-11] + '.dat', format='ascii.fixed_width', overwrite=True)
#     # table.write('/Users/lzq/Dropbox/MUSEQuBES+CUBS/dats/' + filename_i[44:-11] + '.dat', format='fits', overwrite=True)