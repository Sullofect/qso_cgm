import os
import numpy as np
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe

# Load the data
path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'ESO_DEEP_offset_zapped.fits_SUBTRACTED.fits')
cube = Cube(path)
cube = cube.subcube((-18.8643, 40.1359), 40)
cube = cube.select_lambda(7940, 8040)

continuum = cube.clone(data_init=np.empty, var_init=np.zeros)

for sp, co in zip(iter_spe(cube), iter_spe(continuum)):
    co[:] = sp.poly_spec(3, weight=True)
    sp.unmask()

cube.unmask()
continuum.unmask()
cube_bet_Hbeta_OIII = cube - continuum

cube_bet_Hbeta_OIII_line = cube_bet_Hbeta_OIII
image_bet_Hbeta_OIII_line = cube_bet_Hbeta_OIII_line.sum(axis=0) * 1.25 * 1e-20 / 0.2 / 0.2  # put into SB units

cube_bet_Hbeta_OIII_line.write('/Users/lzq/Dropbox/Data/CGM/CUBE_bet_Hbeta_OIII_line_offset_zapped.fits')
image_bet_Hbeta_OIII_line.write('/Users/lzq/Dropbox/Data/CGM/image_bet_Hbeta_OIII_SB_offset_zapped.fits')
