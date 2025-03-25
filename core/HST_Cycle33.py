import os
import lmfit
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import ascii
from matplotlib import rc
from astropy.wcs import WCS
from regions import PixCoord
from astropy.cosmology import FlatLambdaCDM
from regions import RectangleSkyRegion, RectanglePixelRegion, CirclePixelRegion
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel
from palettable.cmocean.sequential import Dense_20_r
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
wave_OII3727_vac = 3727.092
wave_OII3729_vac = 3729.875
c_kms = 2.998e5

z = 0.78071
def rest2obs(x):
    return x * (1 + z)

def obs2rest(x):
    return x / (1 + z)

# Define the model fit function
def model_OII(wave_vac, z, sigma_kms, flux_OII, r_OII3729_3727, a, b):
    wave_OII3727_obs = wave_OII3727_vac * (1 + z)
    wave_OII3729_obs = wave_OII3729_vac * (1 + z)

    sigma_OII3727_A = (sigma_kms / c_kms * wave_OII3727_obs)
    sigma_OII3729_A = (sigma_kms / c_kms * wave_OII3729_obs)

    flux_OII3727 = flux_OII / (1 + r_OII3729_3727)
    flux_OII3729 = flux_OII / (1 + 1.0 / r_OII3729_3727)

    peak_OII3727 = flux_OII3727 / np.sqrt(2 * sigma_OII3727_A ** 2 * np.pi)
    peak_OII3729 = flux_OII3729 / np.sqrt(2 * sigma_OII3729_A ** 2 * np.pi)

    OII3727_gaussian = peak_OII3727 * np.exp(-(wave_vac - wave_OII3727_obs) ** 2 / 2 / sigma_OII3727_A ** 2)
    OII3729_gaussian = peak_OII3729 * np.exp(-(wave_vac - wave_OII3729_obs) ** 2 / 2 / sigma_OII3729_A ** 2)

    return OII3727_gaussian + OII3729_gaussian + a * wave_vac + b

# Path to the spectra
path_spectra = '../../Proposal/HST+JWST/cycle33/spec-7235-56603-0078.fits'
data = fits.open(path_spectra)
wave, flux, err = 10 ** data[1].data['loglam'], data[1].data['flux'], 1 / np.sqrt(data[1].data['ivar'])
model = data[1].data['model']

# OII
OII_region = np.where((wave > 6600) * (wave < 6675))
wave_OII = wave[OII_region]
flux_OII = flux[OII_region]

redshift_guess = 0.78071
sigma_kms_guess = 150.0
flux_OII_guess = 42
r_OII3729_3727_guess = 100

parameters = lmfit.Parameters()
parameters.add_many(('z', redshift_guess, True, None, None, None),
                    ('sigma_kms', sigma_kms_guess, True, 10.0, 500.0, None),
                    ('flux_OII', flux_OII_guess, True, None, None, None),
                    ('r_OII3729_3727', r_OII3729_3727_guess, True, 0, 3, None),
                    ('a', 0.0, True, None, None, None),
                    ('b', 100, True, None, None, None))
spec_model = lmfit.Model(model_OII, missing='drop')
result = spec_model.fit(flux_OII, wave_vac=wave_OII, params=parameters)
z = result.best_values['z']
print(z)

# Plot the Spectrum
fig, ax = plt.subplots(1, 1, figsize=(16, 4), dpi=300)
ax.plot(wave, flux, lw=1, color='k', drawstyle='steps-mid', label=r'$\mathrm{Data}$')
ax.plot(wave, err, lw=1, color='lightgrey', label=r'$\mathrm{err}$')
ax.set_xlim(4000, 10000)
ax.set_ylim(-10, 95)
ax.minorticks_on()
ax.set_xlabel(r'$\mathrm{Observed \; Wavelength \; [\AA]}$', size=20)
ax.set_ylabel(r'${f}_{\lambda} \; [10^{-17} \; \mathrm{erg \; s^{-1} \; cm^{-2} \AA^{-1}}]$', size=20)
ax.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=20, size=5)
ax.tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
ax.annotate(text=r'$\mathrm{[O \, II]}$', xy=(0.45, 0.8), xycoords='axes fraction', size=20)
ax.annotate(text=r'$\mathrm{Mg \, II}$', xy=(0.2, 0.8), xycoords='axes fraction', size=20)
# ax.annotate(text=r'$\mathrm{[O \, III]}$', xy=(0.63, 0.8),
#             xycoords='axes fraction', size=20)
ax.fill_between([6630, 6650], [-10, -10], [95, 95], color='lightgrey', alpha=0.5)
ax.fill_between([4900, 5100], [-10, -10], [95, 95], color='lightgrey', alpha=0.5)
# ax.fill_between([8050, 8270], [-20, -20], [480, 480], color='lightgrey', alpha=0.7)

# Second axis
secax = ax.secondary_xaxis('top', functions=(obs2rest, rest2obs))
secax.minorticks_on()
secax.set_xlabel(r'$\mathrm{Rest \mbox{-} frame \; Wavelength \; [\AA]}$', size=20)
secax.tick_params(axis='x', which='major', direction='in', top='on', size=5, labelsize=20)
secax.tick_params(axis='x', which='minor', direction='in', top='on', size=3)

# Second plot
axins = ax.inset_axes([0.55, 0.65, 0.15, 0.3])
axins.plot(wave_OII, flux_OII, color='black', drawstyle='steps-mid')
axins.plot(wave_OII, result.best_fit, color='red')
# axins.set_xlim(6050, 6095)
# axins.set_ylim(55, 59)
axins.minorticks_on()
axins.set_ylabel(r'${f}_{\lambda}$', size=15)
axins.text(6655, 32.5, r'$\mathrm{[O\;II]}$', size=13)
axins.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=13, size=5)
axins.tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
# secaxins = axins.secondary_xaxis('top', functions=(obs2rest, rest2obs))
# secaxins.minorticks_on()
# secaxins.tick_params(axis='x', which='major', direction='in', top='on', size=5, labelsize=13)
# secaxins.tick_params(axis='x', which='minor', direction='in', top='on', size=3)

# Third plot
MgII_region = np.where((wave > 4920) * (wave < 5030))
wave_MgII = wave[MgII_region]
flux_MgII = flux[MgII_region]
model_MgII = model[MgII_region]

axins = ax.inset_axes([0.055, 0.1, 0.15, 0.3])
axins.plot(wave_MgII, flux_MgII, lw=1, color='black', drawstyle='steps-mid')
# axins.set_xlim(4800, 5100)
axins.set_ylim(55, 92)
axins.fill_between([4985, 5015], [55, 55], [92, 92], color='red', alpha=0.5)
axins.minorticks_on()
axins.set_ylabel(r'${f}_{\lambda}$', size=15)
axins.text(4930, 80, r'$\mathrm{Mg\, II}$', size=13)
axins.tick_params(axis='both', which='major', direction='in', bottom='on', left='on', right='on', labelsize=13, size=5)
axins.tick_params(axis='both', which='minor', direction='in', bottom='on', left='on', right='on', size=3)
fig.savefig('../../Proposal/HST+JWST/cycle33/Spectra_QSO', bbox_inches='tight')