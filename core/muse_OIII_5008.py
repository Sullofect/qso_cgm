import os
import numpy as np
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe

# Load the data
path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'ESO_DEEP_offset_zapped.fits_SUBTRACTED.fits')
cube = Cube(path)
cube = cube.subcube((-18.8643, 40.1359), 40)
cube = cube.select_lambda(8100, 8200)

continuum = cube.clone(data_init=np.empty, var_init=np.zeros)

for sp, co in zip(iter_spe(cube), iter_spe(continuum)):
    sp.mask_region(8140, 8180)
    co[:] = sp.poly_spec(3, weight=True)
    sp.unmask()

cube.unmask()
continuum.unmask()
cube_OIII_5008 = cube - continuum

cube_OIII_5008_line = cube_OIII_5008.select_lambda(8140, 8180)
image_OIII_5008_line = cube_OIII_5008_line.sum(axis=0) * 1.25 * 1e-20 / 0.2 / 0.2  # put into SB units

cube_OIII_5008.write('/Users/lzq/Dropbox/Data/CGM/CUBE_OIII_5008_line_offset_zapped.fits')
image_OIII_5008_line.write('/Users/lzq/Dropbox/Data/CGM/image_OIII_5008_line_SB_offset_zapped.fits')
