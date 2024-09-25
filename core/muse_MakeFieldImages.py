import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from regions import Regions
from scipy.stats import norm
from astropy.io import ascii
from astropy.table import Table
from astropy import units as u
from scipy.optimize import minimize
from astropy.coordinates import SkyCoord
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.major.size'] = 12
mpl.rcParams['ytick.major.size'] = 12


# Preliminary
def LoadFieldGals(cubename=None, z_qso=None, ):
    path = '../../MUSEQuBES+CUBS/datacubes_gaia' \
           '/{}_ESO-DEEP_ZAP_gaia_spec1D/{}_ESO-DEEP_ZAP_gaia_objects.fits'.format(cubename, cubename)
    data = fits.getdata(path, 1, ignore_missing_end=True)

    # Basic information in catalog
    ra, dec, row, ID, name = data['ra'], data['dec'], data['row'], data['id'], data['name']
    ql, cl, z, ct = data['quality'], data['class'], data['redshift'], data['comment']

    # Array manipulation
    select_gal = np.where(cl == 'galaxy')
    row_gal = row[select_gal]
    ID_gal = ID[select_gal]
    z_gal = z[select_gal]
    name_gal = name[select_gal]
    ql_gal = ql[select_gal]
    ra_gal, dec_gal = ra[select_gal], dec[select_gal]

    select_qua = np.where((ql_gal == 1) | (ql_gal == 2))
    row_qua = row_gal[select_qua]
    ID_qua = ID_gal[select_qua]
    z_qua = z_gal[select_qua]
    v_qua = 3e5 * (z_qua - z_qso) / (1 + z_qso)
    name_qua = name_gal[select_qua]
    ql_qua = ql_gal[select_qua]
    ra_qua, dec_qua = ra_gal[select_qua], dec_gal[select_qua]

    bins_ggp = np.arange(-3000, 3200, 200)
    select_v = np.where((v_qua > bins_ggp[0]) * (v_qua < bins_ggp[-1]))
    row_ggp = row_qua[select_v]
    ID_ggp = ID_qua[select_v]
    z_ggp = z_qua[select_v]
    v_ggp = v_qua[select_v]
    name_ggp = name_qua[select_v]
    ql_ggp = ql_qua[select_v]
    ra_ggp, dec_ggp = ra_qua[select_v], dec_qua[select_v]
    output = np.array([bins_ggp, row_ggp, ID_ggp, z_ggp, v_ggp, name_ggp, ql_ggp, ra_ggp, dec_ggp], dtype=object)

    #
    filename = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
    # if os.path.isfile(filename) is not True:
    t = Table()
    t['row'] = row_ggp
    t['ID'] = ID_ggp
    t['z'] = z_ggp
    t['v'] = v_ggp
    t['name'] = name_ggp
    t['ql'] = ql_ggp
    t['ra'] = ra_ggp
    t['dec'] = dec_ggp
    t.write(filename, format='fits', overwrite=True)
    return output


#
def MakeFieldImage(cubename=None):
    str_zap, UseSeg = '', (1.5, 'gauss', 1.5, 'gauss')
    line_OII, line_OIII = 'OII', 'OIII'
    path_savefig_mini = '../../MUSEQuBES+CUBS/plots/{}_mini_gaia.png'.format(cubename)
    path_savefig = '../../MUSEQuBES+CUBS/plots/{}_gaia.png'.format(cubename)
    path_dat = '../../MUSEQuBES+CUBS/datacubes_gaia/{}_ESO-DEEP_ZAP_gaia.dat'.format(cubename)
    path_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP_subtracted_OII_SB_3DSeg_1.5_gauss_1.5_gauss.fits'.format(cubename)
    path_SB_OII_kin = '../../MUSEQuBES+CUBS/fit_kin/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line_OII, *UseSeg)
    path_SB_OIII_kin = '../../MUSEQuBES+CUBS/fit_kin/{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line_OIII, *UseSeg)

    # Load info
    path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]
    ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]
    print(z_qso)

    # Load the all galaxy catalog
    gal_all_dat = ascii.read(path_dat, format='fixed_width')
    ra_gal_all, dec_gal_all = gal_all_dat['ra'], gal_all_dat['dec']

    #
    bins_gal, row_gal, ID_gal, z_gal, v_gal, name_gal, ql_gal, ra_gal, dec_gal = LoadFieldGals(cubename=cubename,
                                                                                               z_qso=z_qso)
    print(ra_gal)

    if cubename == '3C57':
        path_IMACS = '../../MUSEQuBES+CUBS/gal_info/3C57_gal_IMACS.fits'
        data_IMACS = fits.getdata(path_IMACS, 1)

    # Load the image
    path_hb = '../../MUSEQuBES+CUBS/datacubes_gaia/{}_drc_offset_gaia.fits'.format(cubename)
    # data_hb = fits.getdata(path_hb, 1, ignore_missing_end=True)


    # # Figure
    # fig = plt.figure(figsize=(8, 8), dpi=300)
    # gc = aplpy.FITSFigure(path_hb, figure=fig, north=True, hdu=1)
    # gc.set_xaxis_coord_type('scalar')
    # gc.set_yaxis_coord_type('scalar')
    #
    #
    # #
    # gc.recenter(ra_qso, dec_qso, width=15 / 3600, height=15 / 3600)
    #
    # #
    # gc.set_system_latex(True)
    # # gc.show_colorscale(cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
    # gc.show_colorscale(cmap='Greys', vmin=-0.005, vmax=1, vmid=-0.001, stretch='arcsinh')
    # gc.add_colorbar()
    # # gc.colorbar.set_box([0.15, 0.12, 0.38, 0.02], box_orientation='horizontal')
    # gc.colorbar.set_location('bottom')
    # gc.colorbar.set_pad(0.0)
    # gc.colorbar.set_font(size=30)
    # gc.colorbar.set_axis_label_text(r'$\mathrm{SB \; [10^{-17} \; erg \; cm^{-2} \; '
    #                                 r's^{-1} \; arcsec^{-2}]}$')
    # gc.colorbar.set_axis_label_font(size=30)
    # gc.colorbar.hide()
    #
    #
    # # Hide ticks
    # gc.ticks.set_length(30)
    # gc.ticks.hide()
    # gc.tick_labels.hide()
    # gc.axis_labels.hide()
    #
    # # Markers
    # gc.add_scalebar(length=7 * u.arcsecond)
    # # gc.add_scalebar(length=15 * u.arcsecond)
    # gc.scalebar.set_corner('top left')
    # # gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
    # gc.scalebar.set_label(r"$7'' \approx 50 \mathrm{\; pkpc}$")
    # gc.scalebar.set_font_size(30)
    #
    # gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='*', c='lightgrey',
    #                 edgecolors='k', linewidths=0.5, s=3000)
    # gc.show_markers(ra_gal, dec_gal, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=530)
    #
    # # Draw contours
    # contour_level = 0.20
    # gc.show_contour(path_SB_OII_kin, levels=[contour_level], colors='blue', linewidths=2,
    #                 smooth=5, kernel='box', hdu=1)
    # gc.show_contour(path_SB_OIII_kin, levels=[contour_level], colors='red', linewidths=2,
    #                 smooth=5, kernel='box', hdu=1)
    #
    # # labels
    # gc.add_label(0.58, 0.88, r'$\rm MUSE \, [O\,II]$', color='blue', size=30, relative=True, horizontalalignment='left')
    # gc.add_label(0.58, 0.80, r'$\rm MUSE \, [O\,III]$', color='red', size=30, relative=True, horizontalalignment='left')
    # gc.add_label(0.58, 0.95, r'$\mathrm{ACS\!+\!F814W}$', color='k', size=30, relative=True, horizontalalignment='left')
    # gc.add_label(0.08, 0.08, '(a)', color='k', size=40, relative=True)
    #
    # # NE
    # # xw, yw = gc.pixel2world(140, 140)
    # # gc.show_arrows(xw, yw, -0.000035 * yw, 0, color='k')
    # # gc.show_arrows(xw, yw, 0, -0.000035 * yw, color='k')
    # # gc.add_label(0.9778, 0.30, r'N', size=30, relative=True)
    # # gc.add_label(0.88, 0.10, r'E', size=30, relative=True)
    #
    # # Labels
    # fig.savefig(path_savefig_mini, bbox_inches='tight')
    #
    # raise ValueError('STOP')

    # Figure
    fig = plt.figure(figsize=(8, 8), dpi=300)
    gc1 = aplpy.FITSFigure(path_hb, figure=fig, north=True, hdu=1)
    gc = aplpy.FITSFigure(path_hb, figure=fig, north=True, hdu=1)
    gc.set_xaxis_coord_type('scalar')
    gc.set_yaxis_coord_type('scalar')
    gc1.set_xaxis_coord_type('scalar')
    gc1.set_yaxis_coord_type('scalar')


    #
    gc.recenter(ra_qso, dec_qso, width=90 / 3600, height=90 / 3600)
    gc1.recenter(ra_qso, dec_qso, width=90 / 3600, height=90 / 3600)  # 0.02 / 0.01 40''

    #
    gc.set_system_latex(True)
    gc1.set_system_latex(True)
    gc1.show_colorscale(cmap='coolwarm', vmin=-1000, vmax=1000)
    gc1.hide_colorscale()
    gc1.add_colorbar()
    gc1.colorbar.set_box([0.15, 0.145, 0.38, 0.02], box_orientation='horizontal')
    gc1.colorbar.set_axis_label_text(r'$\mathrm{\Delta} v \mathrm{\; [km \, s^{-1}]}$')
    gc1.colorbar.set_axis_label_font(size=12)
    gc1.colorbar.set_axis_label_pad(-40)
    gc1.colorbar.set_location('bottom')
    # gc1.colorbar.hide()
    # gc.show_colorscale(cmap='Greys', vmin=0, vmax=0.005, stretch='linear', smooth=3, kernel='gauss')
    gc.show_colorscale(cmap='Greys', vmin=-2.353e-2, vmax=4.897e-2)
    # gc.show_colorscale(cmap='Greys')

    gc.add_colorbar()
    gc.colorbar.set_box([0.15, 0.12, 0.38, 0.02], box_orientation='horizontal')
    gc.colorbar.hide()

    # Scale bar
    # gc.add_scalebar(length=15 * u.arcsecond)
    # gc.scalebar.set_corner('top left')
    # gc.scalebar.set_label(r"$15'' \approx 100 \mathrm{\; pkpc}$")
    # gc.scalebar.set_font_size(15)

    # Hide ticks
    gc.ticks.set_length(30)
    gc1.ticks.set_length(30)
    gc.ticks.hide()
    gc.tick_labels.hide()
    gc.axis_labels.hide()
    gc1.ticks.hide()
    gc1.tick_labels.hide()
    gc1.axis_labels.hide()
    norm = mpl.colors.Normalize(vmin=-1000, vmax=1000)

    # Markers
    gc.show_markers(ra_qso, dec_qso, facecolors='none', marker='*', c='lightgrey',
                    edgecolors='k', linewidths=0.5, s=400)
    # gc.add_label(ra_qso - 0.0015, dec_qso, 'QSO', size=10)
    gc.show_markers(ra_gal, dec_gal, marker='o', facecolor='none', c='none', edgecolors=plt.cm.coolwarm(norm(v_gal)),
                    linewidths=1.2, s=80)
    gc.show_markers(ra_gal, dec_gal, facecolor='none', marker='o', c='none', edgecolors='k', linewidths=0.8, s=120)
    # gc.show_markers(ra_gal_all, dec_gal_all, facecolor='none', marker='o', c='none', edgecolors='red',
    #                 linewidths=0.8, s=120)
    gc.show_contour(path_OII, hdu=1, levels=[0.1], color='black', linewidths=1, smooth=5, kernel='box')
    gc.show_markers(data_IMACS['ra'], data_IMACS['dec'], facecolor='none', marker='o', c='none',
                    edgecolors='r', linewidths=0.8, s=150)

    # Labels
    # xw, yw = 40.1231559, -18.8580071
    # gc.show_arrows(xw, yw, -0.0001 * yw, 0, color='k')
    # gc.show_arrows(xw, yw, 0, -0.0001 * yw, color='k')
    # gc.add_label(0.985, 0.85, r'N', size=15, relative=True)
    # gc.add_label(0.89, 0.748, r'E', size=15, relative=True)
    gc.add_label(0.87, 0.97, r'$\mathrm{ACS\!+\!F814W}$', color='k', size=15, relative=True)
    # gc.add_label(0.27, 0.86, r"$\rm MUSE \, 1'\times 1' \, FoV$", size=15, relative=True, rotation=60)
    # gc.add_label(0.47, 0.30, r"$\rm 30'' \times 30''$", size=15, relative=True)
    fig.savefig(path_savefig, bbox_inches='tight')


# MakeFieldImage(cubename='Q0107-0235')
# MakeFieldImage(cubename='PB6291')
# MakeFieldImage(cubename='HE0153-4520')
MakeFieldImage(cubename='3C57')
# MakeFieldImage(cubename='TEX0206-048')
# MakeFieldImage(cubename='HE0226-4110')
# MakeFieldImage(cubename='PKS0232-04')
# MakeFieldImage(cubename='HE0435-5304')
# MakeFieldImage(cubename='HE0439-5254')
# MakeFieldImage(cubename='PKS0552-640')
# MakeFieldImage(cubename='Q1354+048')
# MakeFieldImage(cubename='LBQS1435-0134')
# MakeFieldImage(cubename='PG1522+101')
# MakeFieldImage(cubename='HE1003+0149')
# MakeFieldImage(cubename='PKS0405-123')
# MakeFieldImage(cubename='HE0238-1904')