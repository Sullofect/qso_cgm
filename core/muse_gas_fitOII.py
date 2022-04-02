import os
import aplpy
import lmfit
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from astropy import units as u
from PyAstronomy import pyasl
from astropy.cosmology import FlatLambdaCDM
from matplotlib.colors import ListedColormap
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe, iter_ima
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10

Blues = cm.get_cmap('Blues', 256)
Reds = cm.get_cmap('Reds', 256)
newcolors = Blues(np.linspace(0, 1, 256))
newcolors_red = Reds(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors)


def getSigma_MUSE(wave):
    return (5.866e-8 * wave ** 2 - 9.187e-4 * wave + 6.04) / 2.355


# Define the model fit function
def model(wave_vac, z, sigma_kms, flux_OII, r_OII3729_3727, a, b):
    # Constants
    c_kms = 2.998e5
    wave_OII3727_vac = 3727.092
    wave_OII3729_vac = 3729.875

    wave_OII3727_obs = wave_OII3727_vac * (1 + z)
    wave_OII3729_obs = wave_OII3729_vac * (1 + z)

    sigma_OII3727_A = np.sqrt((sigma_kms / c_kms * wave_OII3727_obs) ** 2 + (getSigma_MUSE(wave_OII3727_obs)) ** 2)
    sigma_OII3729_A = np.sqrt((sigma_kms / c_kms * wave_OII3729_obs) ** 2 + (getSigma_MUSE(wave_OII3729_obs)) ** 2)

    flux_OII3727 = flux_OII / (1 + r_OII3729_3727)
    flux_OII3729 = flux_OII / (1 + 1.0 / r_OII3729_3727)

    peak_OII3727 = flux_OII3727 / np.sqrt(2 * sigma_OII3727_A ** 2 * np.pi)
    peak_OII3729 = flux_OII3729 / np.sqrt(2 * sigma_OII3729_A ** 2 * np.pi)

    OII3727_gaussian = peak_OII3727 * np.exp(-(wave_vac - wave_OII3727_obs) ** 2 / 2 / sigma_OII3727_A ** 2)
    OII3729_gaussian = peak_OII3729 * np.exp(-(wave_vac - wave_OII3729_obs) ** 2 / 2 / sigma_OII3729_A ** 2)

    return OII3727_gaussian + OII3729_gaussian + a * wave_vac + b


# Fitting the narrow band image profile
path_cube_OII = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'CUBE_OII_line_offset.fits')
cube_OII = Cube(path_cube_OII)
# cube_OII = cube_OII.subcube((80, 100), 5, unit_center=None, unit_size=None)
cube_OII[0, :, :].write('/Users/lzq/Dropbox/Data/CGM/image_OII_fitline.fits')

redshift_guess = 0.63
sigma_kms_guess = 150.0
flux_OIII5008_guess = 42
r_OII3729_3727_guess = 100

parameters = lmfit.Parameters()
parameters.add_many(('z', redshift_guess, True, None, None, None),
                    ('sigma_kms', sigma_kms_guess, True, 10.0, 500.0, None),
                    ('flux_OII', flux_OII_guess, True, None, None, None),
                    ('r_OII3729_3727', r_OII3729_3727_guess, True, 0, 3, None),
                    ('a', 0.0, True, None, None, None),
                    ('b', 100, True, None, None, None))

size = np.shape(cube_OIII)[1]
z_fit, dz_fit = np.zeros((size, size)), np.zeros((size, size))
sigma_fit, dsigma_fit = np.zeros((size, size)), np.zeros((size, size))
flux_fit, dflux_fit = np.zeros((size, size)), np.zeros((size, size))
r_fit, dr_fit = np.zeros((size, size)), np.zeros((size, size))
a_fit, b_fit = np.zeros((size, size)), np.zeros((size, size))
da_fit, db_fit = np.zeros((size, size)), np.zeros((size, size))

for i in range(size):
    for j in range(size):
        wave_OII_vac = pyasl.airtovac2(cube_OII.wave.coord())
        flux_OII = cube_OII[:, i, j].data * 1e-3
        spec_model = lmfit.Model(model, missing='drop')
        result = spec_model.fit(flux_OII, wave_vac=wave_OII_vac, params=parameters)

        #
        z, sigma, flux = result.best_values['z'], result.best_values['sigma_kms'], result.best_values['flux_OIII5008']
        r, a, b = result.best_values['r_OII3729_3727'], result.best_values['a'], result.best_values['b']

        #
        dz, dsigma, dflux = result.params['z'].stderr, result.params['sigma_kms'].stderr, \
                            result.params['flux_OIII5008'].stderr
        dr, da, db = result.params['r_OII3729_3727'].stderr, result.params['a'].stderr, result.params['b'].stderr

        # if i == 30:
        #     if (j > 30) and (j < 50):
        #         plt.plot(wave_OIII_vac, flux_OIII, '-')
        #         plt.plot(wave_OIII_vac, model(wave_OIII_vac, z, sigma, flux, a, b))
        #         plt.show()

        #
        z_fit[i, j], dz_fit[i, j] = z, dz
        sigma_fit[i, j], dsigma_fit[i, j] = sigma, dsigma
        flux_fit[i, j], dflux_fit[i, j] = flux, dflux
        r_fit = a_fit[i, j], b_fit[i, j] = r, a, b
        dr_fit, da_fit[i, j], db_fit[i, j] = dr, da, db

z_qso = 0.6282144177077355
v_fit = 3e5 * (z_fit - z_qso) / (1 + z_qso)

info = np.array([z_fit, sigma_fit, flux_fit, a_fit, b_fit])
info_err = np.array([dz_fit, dsigma_fit, dflux_fit, da_fit, db_fit])
fits.writeto('/Users/lzq/Dropbox/Data/CGM/fitOII_info.fits', info, overwrite=True)
fits.writeto('/Users/lzq/Dropbox/Data/CGM/fitOII_info_err.fits', info_err, overwrite=True)


# Convert Fits file into correct form
def ConvertFits(filename=None, table=None):
    path = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', filename + '.fits')
    data, hdr = fits.getdata(path, 1, header=True)
    fits.writeto('/Users/lzq/Dropbox/Data/CGM/' + filename + '_revised.fits', table, overwrite=True)
    data1, hdr1 = fits.getdata('/Users/lzq/Dropbox/Data/CGM/' + filename + '_revised.fits', 0, header=True)
    hdr1['BITPIX'], hdr1['NAXIS'], hdr1['NAXIS1'], hdr1['NAXIS2'] = hdr['BITPIX'], hdr['NAXIS'], hdr['NAXIS1'], hdr['NAXIS2']
    hdr1['CRPIX1'], hdr1['CRPIX2'], hdr1['CTYPE1'], hdr1['CTYPE2'] = hdr['CRPIX1'], hdr['CRPIX2'], hdr['CTYPE1'], hdr['CTYPE2']
    hdr1['CRVAL1'], hdr1['CRVAL2'], hdr1['LONPOLE'], hdr1['LATPOLE'] = hdr['CRVAL1'], hdr['CRVAL2'], hdr['LONPOLE'], hdr['LATPOLE']
    hdr1['CSYER1'], hdr1['CSYER2'], hdr1['MJDREF'], hdr1['RADESYS'] = hdr['CSYER1'], hdr['CSYER2'], hdr['MJDREF'], hdr['RADESYS']
    hdr1['CD1_1'], hdr1['CD1_2'], hdr1['CD2_1'], hdr1['CD2_2'] =  hdr['CD1_1'], hdr['CD1_2'], hdr['CD2_1'], hdr['CD2_2']
    # Rescale the data by 1e17
    fits.writeto('/Users/lzq/Dropbox/Data/CGM/' + filename + '_revised.fits', data1, hdr1, overwrite=True)
ConvertFits(filename='image_OII_fitline', table=v_fit)

# fig = plt.figure(figsize=(8, 8), dpi=300)
# path_dv = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_OIII_fitline_revised.fits')
# gc = aplpy.FITSFigure(path_dv, figure=fig, subplot=(1, 2, 2), north=True)
# gc.set_system_latex(True)
# gc.show_colorscale(vmin=-1000, vmax=1000, cmap='seismic')
# gc.add_colorbar()
# gc.ticks.set_length(30)
# # gc.show_regions('/Users/lzq/Dropbox/Data/CGM/galaxy_list.reg')
# gc.colorbar.set_location('bottom')
# gc.colorbar.set_pad(0.0)
# gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta v \; [km \, s^{-1}]}$')
# gc.colorbar.set_font(size=15)
# gc.colorbar.set_axis_label_font(size=15)
# gc.add_scalebar(length=5 * u.arcsecond)
# gc.scalebar.set_corner('top left')
# gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
# gc.scalebar.set_font_size(15)
# gc.ticks.hide()
# gc.tick_labels.hide()
# gc.axis_labels.hide()
# gc.add_label(0.90, 0.97, r'[OIII 5008]', size=15, relative=True)
# xw, yw = gc.pixel2world(195, 150)
# gc.show_arrows(xw, yw, -0.00005 * yw, 0, color='k')
# gc.show_arrows(xw, yw, 0, -0.00005 * yw, color='k')
# gc.add_label(0.9775, 0.85, r'N', size=15, relative=True)
# gc.add_label(0.88, 0.75, r'E', size=15, relative=True)
# fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/OIII_dv_map.pdf', bbox_inches='tight')