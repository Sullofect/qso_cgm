import os
from astropy.io import ascii
# import lmfit
# import extinction
import numpy as np
# import astropy.io.fits as fits
# import matplotlib.pyplot as plt
from matplotlib import rc
from PyAstronomy import pyasl
# from muse_gas_spectra_S3S4 import model_all as S3S4_model_all
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe
from astropy.table import Table
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

path_cube = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'raw_data',
                         'ESO_DEEP_offset_zapped.fits_SUBTRACTED.fits')
cube = Cube(path_cube)

#
path_region = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'gas_list_revised.reg')
ra_array_input = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 0]
dec_array_input = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 1]
radius_array_input = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 2]
text_array_input = np.loadtxt(path_region, dtype=str, usecols=[3], delimiter=',')

region = np.array(['S2', 'S4', 'S6'])
region_mask = np.in1d(text_array_input, region)
ra_array, dec_array, radius_array, text_array = ra_array_input[region_mask], dec_array_input[region_mask], \
                                                radius_array_input[region_mask], text_array_input[region_mask]

for i in range(len(ra_array)):
    data_i = cube.aperture((dec_array[i], ra_array[i]), radius_array[i], is_sum=True)
    wave_vac = pyasl.airtovac2(data_i.wave.coord())
    flux_i, flux_err_i = data_i.data * 1e-3, np.sqrt(data_i.var) * 1e-3

    #
    table = Table()
    table['wave'] = wave_vac
    table['flux'] = flux_i
    table['flux_err'] = flux_err_i
    table.write('/Users/lzq/Dropbox/Data/CGM/RegionsSpecData/SpecData_' + region[i] + '.fits', format='fits', overwrite=True)

