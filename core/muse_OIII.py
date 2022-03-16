import os
import numpy as np
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe

# Load the data
path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'ESO_DEEP_offset.fits_SUBTRACTED.fits')
cube = Cube(path)
cube = cube.subcube((-18.8643, 40.1359), 40)
cube = cube.select_lambda(8100, 8200)

continuum = cube.clone(data_init=np.empty, var_init=np.zeros)

for sp, co in zip(iter_spe(cube), iter_spe(continuum)):
    sp.mask_region(8140, 8160)
    co[:] = sp.poly_spec(3, weight=True)
    sp.unmask()

cube.unmask()
continuum.unmask()
cube_OIII = cube - continuum

cube_OIII_line = cube_OIII.select_lambda(8140, 8160)
image_OIII_line = cube_OIII_line.sum(axis=0) * 1.25 * 1e-20 / 0.2 / 0.2  # put into SB units

cube_OIII.write('/Users/lzq/Dropbox/Data/CGM/CUBE_OIII_line_offset.fits')
image_OIII_line.write('/Users/lzq/Dropbox/Data/CGM/image_OIII_line_SB_offset.fits')
