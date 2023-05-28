import os
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
path_savefig = '/Users/lzq/Dropbox/Data/CGM_plots/'

path_BH = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'raw_data', 'dr7_bh_Nov19_2013.fits')
data, hdr = fits.getdata(path_BH, 1, header=True)
logL_bol = data['LOGLBOL']
logL_bol_qso = np.log10(10.33 * 10 ** 46.21)
print(logL_bol_qso)
mask = np.where(logL_bol > logL_bol_qso)
data = data[mask]
FWHM_broad_HB = data['FWHM_BROAD_HB']
FWHM_narrow_HB = data['FWHM_NARROW_HB']

FWHM_broad_HB_qso, FWHM_narrow_HB_qso = 7709.357052342235, 527.9
plt.figure(figsize=(5, 5))
plt.plot(FWHM_broad_HB, FWHM_narrow_HB, '.')
plt.plot(FWHM_broad_HB_qso, FWHM_narrow_HB_qso, '.k', ms=10)
plt.xlabel(r'$\rm FWHM \, H\beta \, broad(km \, s^{-1})$')
plt.ylabel(r'$\rm FWHM \, H\beta \, narrow(km \, s^{-1})$')
plt.savefig(path_savefig + 'CompareFWHM', bbox_inches='tight')
