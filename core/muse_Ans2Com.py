import os
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
path_savefig = '/Users/lzq/Dropbox/Data/CGM_plots/'

# Comments
path_BH = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'raw_data', 'dr7_bh_Nov19_2013.fits')
data, hdr = fits.getdata(path_BH, 1, header=True)
logL_bol = data['LOGLBOL']
logL_bol_qso = np.log10(10.33 * 10 ** 46.21)
print(logL_bol_qso)
mask = np.where(data['EW_narrow_HB'] != 0)
data = data[mask]
logL_bol = data['LOGLBOL']
FWHM_broad_HB = data['FWHM_BROAD_HB']
FWHM_narrow_HB = data['FWHM_NARROW_HB']
EW_broad_HB = data['EW_BROAD_HB']
EW_narrow_HB = data['EW_NARROW_HB']
EW_OIII = data['EW_OIII_5007']

# QSO information
z_qso = 0.6282144177077355
FWHM_broad_HB_qso, FWHM_narrow_HB_qso = 7709.357052342235, 527.9
EW_broad_HB_qso, EW_narrow_HB_qso = 98.02691867699036, 0.9073753305193385
EW_OIII_qso = 11.319111753029075
#
plt.figure(figsize=(5, 5), dpi=300)
plt.plot(logL_bol, EW_narrow_HB, '.', ms=0.5, label='Shen et al. (2011)')
plt.plot(logL_bol_qso, EW_narrow_HB_qso, '.k', ms=10, label='HE0238-1904')
# plt.plot(FWHM_broad_HB, FWHM_narrow_HB, '.')
# plt.plot(FWHM_broad_HB_qso, FWHM_narrow_HB_qso, '.k', ms=10)
plt.xlim(44, 48)
plt.ylim(-0.5, 10)
# plt.xlabel(r'$\rm FWHM \, H\beta \, broad(km \, s^{-1})$')
plt.xlabel(r'$\rm log(L_{bol}/erg \, s^{-1})$')
plt.ylabel(r'$\rm EW \, H\beta \, narrow(\AA)$')
plt.legend()
# plt.ylabel(r'$\rm FWHM \, H\beta \, narrow(km \, s^{-1})$')
plt.savefig(path_savefig + 'CompareEW_HB', bbox_inches='tight')

plt.figure(figsize=(5, 5), dpi=300)
plt.plot(logL_bol, EW_OIII, '.', ms=0.5, label='Shen et al. (2011)')
plt.plot(logL_bol_qso, EW_OIII_qso, '.k', ms=10, label='HE0238-1904')
# plt.plot(FWHM_broad_HB, FWHM_narrow_HB, '.')
# plt.plot(FWHM_broad_HB_qso, FWHM_narrow_HB_qso, '.k', ms=10)
plt.xlim(44, 48)
plt.ylim(-0.5, 30)
# plt.xlabel(r'$\rm FWHM \, H\beta \, broad(km \, s^{-1})$')
plt.xlabel(r'$\rm log(L_{bol}/erg \, s^{-1})$')
plt.ylabel(r'$\rm EW \, [O \, III] \, narrow(\AA)$')
plt.legend()
# plt.ylabel(r'$\rm FWHM \, H\beta \, narrow(km \, s^{-1})$')
plt.savefig(path_savefig + 'CompareEW_OIII', bbox_inches='tight')
