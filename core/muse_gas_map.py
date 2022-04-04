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
from muse_compare_z import compare_z
from astropy.cosmology import FlatLambdaCDM
from matplotlib.colors import ListedColormap
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe, iter_ima
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10

# Cmap
Blues = cm.get_cmap('Blues', 256)
Reds = cm.get_cmap('Reds', 256)
newcolors = Blues(np.linspace(0, 1, 256))
newcolors_red = Reds(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors)


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


def PlotMap(line='OIII', check=False, snr_thr=3, row=None, z=None, ra=None, dec=None):
    # Load OIII
    path_fit_info = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'fit' + line + '_info.fits')
    path_fit_info_err = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'fit' + line + '_info_err.fits')
    fit_info = fits.getdata(path_fit_info, 0, ignore_missing_end=True)
    fit_info_err = fits.getdata(path_fit_info_err, 0, ignore_missing_end=True)

    # Load data
    [z_fit, sigma_fit, flux_fit, a, b] = fit_info
    [dz_fit, dsigma_fit, dflux_fit, da, db] = fit_info_err
    # print(flux_fit / dflux_fit)
    z_qso = 0.6282144177077355
    v_fit = 3e5 * (z_fit - z_qso) / (1 + z_qso)
    v_gal = 3e5 * (z - z_qso) / (1 + z_qso)

    # Check consistency
    if check is True:
        plt.figure(figsize=(8, 8), dpi=300)
        plt.imshow(v_fit, cmap='coolwarm', vmin=-300, vmax=300, origin='lower')

    # Final data
    v_fit = np.where((flux_fit / dflux_fit > snr_thr), v_fit, np.nan)
    ConvertFits(filename='image_' + line + '_fitline', table=v_fit)

    # Plot
    fig = plt.figure(figsize=(8, 8), dpi=300)
    path_dv = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'image_' + line + '_fitline_revised.fits')
    gc = aplpy.FITSFigure(path_dv, figure=fig, north=True)
    gc.set_system_latex(True)
    gc.show_colorscale(vmin=-300, vmax=300, cmap='coolwarm')
    gc.add_colorbar()
    gc.ticks.set_length(30)
    gc.show_markers(40.1359, -18.8643, facecolors='none', marker='*', c='none', edgecolors='k', linewidths=0.5, s=250)
    gc.show_markers(ra, dec, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=100)
    gc.show_markers(ra, dec, marker='o', c=v_gal, linewidths=0.5, s=40,
                    vmin=-300, vmax=300, cmap='coolwarm')
    # gc.show_regions('/Users/lzq/Dropbox/Data/CGM/galaxy_list.reg')
    gc.colorbar.set_location('bottom')
    gc.colorbar.set_pad(0.)
    gc.colorbar.set_axis_label_text(r'$\mathrm{\Delta v \; [km \, s^{-1}]}$')
    gc.colorbar.set_font(size=15)
    gc.colorbar.set_axis_label_font(size=15)
    gc.add_scalebar(length=15 * u.arcsecond)
    gc.scalebar.set_corner('top left')
    gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
    gc.scalebar.set_font_size(15)
    gc.ticks.hide()
    gc.tick_labels.hide()
    gc.axis_labels.hide()

    for i in range(len(row)):
        gc.add_label(ra[i] + 0.00014, dec[i] - 0.00008, "{0:.0f}".format(v_gal[i]), size=10, horizontalalignment='right',
                     verticalalignment='bottom')
    # label
    if line == 'OIII':
        gc.add_label(0.80, 0.97, r'$\Delta v = v_{\mathrm{[O \, III]}} - v_{\mathrm{qso}}$', size=15, relative=True)
    elif line == 'OII':
        gc.add_label(0.80, 0.97, r'$\Delta v = v_{\mathrm{[O \, II]}} - v_{\mathrm{qso}}$', size=15, relative=True)

    xw, yw = gc.pixel2world(195, 150)
    gc.show_arrows(xw, yw, -0.00005 * yw, 0, color='k')
    gc.show_arrows(xw, yw, 0, -0.00005 * yw, color='k')
    gc.add_label(0.9775, 0.85, r'N', size=15, relative=True)
    gc.add_label(0.88, 0.75, r'E', size=15, relative=True)
    fig.savefig('/Users/lzq/Dropbox/Data/CGM_plots/' + line + '_dv_map.png', bbox_inches='tight')


# Load galxies infomation
ggp_info = compare_z(cat_sean='ESO_DEEP_offset_zapped_objects_sean.fits',
                     cat_will='ESO_DEEP_offset_zapped_objects.fits')
row_final = ggp_info[1]
ID_final = ggp_info[2]
z_final = ggp_info[3]
name_final = ggp_info[5]
ra_final = ggp_info[7]
dec_final = ggp_info[8]

col_ID = np.arange(len(row_final))
select_array = np.sort(np.array([6, 7, 181, 182, 80, 81, 82, 179, 4, 5, 64]))
select_gal = np.in1d(row_final, select_array)
row_final = row_final[select_gal]
z_final = z_final[select_gal]
ra_final = ra_final[select_gal]
dec_final = dec_final[select_gal]

# run
PlotMap(line='OII', snr_thr=2.5, row=row_final, z=z_final, ra=ra_final, dec=dec_final)
PlotMap(line='OIII', snr_thr=3, row=row_final, z=z_final, ra=ra_final, dec=dec_final)
