import os
import pyneb as pn
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from PyAstronomy import pyasl
from scipy import interpolate
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe
from muse_LoadCloudy import format_cloudy_nogrid
from astropy.io import ascii
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)

# [OII] 3726/3729" "[OIII] 4363/5007" "
O2 = pn.Atom('O', 2)
O3 = pn.Atom('O', 3)

# [O II] vs density
fig, axarr = plt.subplots(1, 1, figsize=(5, 5), dpi=300, sharex=True)
fig.subplots_adjust(hspace=0.)
den_array = 10 ** np.linspace(-1, 4, 1000)
tem_array = 10 ** np.array([3.75, 4.00, 4.25, 4.50])
OII3727 = O2.getEmissivity(tem=tem_array, den=den_array, wave=3726)
OII3729 = O2.getEmissivity(tem=tem_array, den=den_array, wave=3729)

for i in range(len(tem_array)):
    axarr.plot(np.log10(den_array), np.log10(OII3729/OII3727)[i, :],
                  '-', label=r'$\mathrm{log} (T/\rm{K})=$ ' + '{:.2f}'.format(np.log10(tem_array[i])))
axarr.set_xlabel(r"$\mathrm{log(Density / cm^{-3})}$", size=25)
axarr.set_ylabel(r'$\mathrm{log( [O \, II] \lambda 3729 / \lambda 3727)}$', size=25)
axarr.minorticks_on()
axarr.legend(prop={'size': 15}, framealpha=0, loc=3, fontsize=15)
axarr.tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                labelsize=20, size=5)
axarr.tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on', size=3)
fig.savefig('../../Papers/Thesis/pyneb_OII.png', bbox_inches='tight')



# [O III] vs temperature
fig, axarr = plt.subplots(1, 1, figsize=(5, 5), dpi=300, sharex=True)
fig.subplots_adjust(hspace=0.)
tem_array = 10 ** np.linspace(3.75, 4.5, 1000)
den_array = np.array([0.1, 1, 10, 100])
OIII4363 = O3.getEmissivity(tem=tem_array, den=den_array, wave=4363)
OIII5007 = O3.getEmissivity(tem=tem_array, den=den_array, wave=5007)
for i in range(len(den_array)):
    axarr.plot(np.log10(tem_array), np.log10(OIII4363/OIII5007)[:, i],
                  '-', label=r'$\mathrm{log}(n_{e}/\rm{cm^{-3}})=$ ' + '{}'.format(np.log10(den_array[i])))
axarr.set_xlabel(r"$\mathrm{log(Temperature / K)}$", size=25)
axarr.set_ylabel(r'$\mathrm{log( [O \, III] \lambda 4363 / \lambda 5007)}$', size=25)
axarr.minorticks_on()
axarr.legend(prop={'size': 15}, framealpha=0, loc=4, fontsize=15)
axarr.tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                labelsize=20, size=5)
axarr.tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on', size=3)
fig.savefig('../../Papers/Thesis/pyneb_OIII.png', bbox_inches='tight')


# Cloudy results

# Load Cloudy result
region = 'S7'
trial = 't1'
c = 3e10
nuL_nu = 46.54
ev_erg = 1.6e-12
Q_H = 10 ** nuL_nu / 13.6 / ev_erg
logd = 22.94  # in cm

#
Hden = np.hstack((np.linspace(-2, 2.6, 24, dtype='f2'), np.linspace(2.8, 4.6, 10, dtype='f2')))[17:24]
U = Q_H / 4 / np.pi / (10 ** logd) ** 2 / 10 ** Hden / c
logU = np.log10(U)
alpha = np.linspace(-1.8, 0, 10, dtype='f2')[:3]
metal = np.linspace(-1.9, -0.3, 9, dtype='f2')

output = format_cloudy_nogrid(filename=[Hden, alpha, metal],
                              path='../../Data/CGM/cloudy/' + region + '_' + trial + '/')[:, :, :, :9]
print(np.log10(U))
print(np.shape(output))
# output = output[:, :, :, ::2]
#
OIIOIII_Halpha = np.log10((10 ** output[3, :, :, :] + 10 ** output[9, :, :, :]) / 3)
OIII_OII = output[9, :, :, :] - output[3, :, :, :]

# Load Grove
data_Grove_z025 = ascii.read('../../Proposal/HST+JWST/photo_Grove/iteramodel_grid1.txt', delimiter=' ')
data_Grove_z050 = ascii.read('../../Proposal/HST+JWST/photo_Grove/iteramodel_grid2.txt', delimiter=' ')
data_Grove_z1 = ascii.read('../../Proposal/HST+JWST/photo_Grove/iteramodel_grid3.txt', delimiter=' ')
data_Grove_z2 = ascii.read('../../Proposal/HST+JWST/photo_Grove/iteramodel_grid4.txt', delimiter=' ')
print(data_Grove_z025)

alpha_025, alpha_050, \
alpha_z1, alpha_z2 = data_Grove_z025['col1'], data_Grove_z050['col1'], data_Grove_z1['col1'], data_Grove_z2['col1']
logu_025, logu_050, \
logu_z1, logu_z2 = data_Grove_z025['col2'], data_Grove_z050['col2'], data_Grove_z1['col2'], data_Grove_z2['col2']
OIII_OII_025, OIII_OII_050, \
OIII_OII_z1, OIII_OII_z2 = data_Grove_z025['col3'] / data_Grove_z025['col4'], data_Grove_z050['col3'] / data_Grove_z050['col4'], \
                             data_Grove_z1['col3'] / data_Grove_z1['col4'], data_Grove_z2['col3'] / data_Grove_z2['col4']
NII_Halpha_025, NII_Halpha_050, \
NII_Halpha_z1, NII_Halpha_z2 = data_Grove_z025['col5'] / data_Grove_z025['col6'], data_Grove_z050['col5'] / data_Grove_z050['col6'], \
                             data_Grove_z1['col5'] / data_Grove_z1['col6'], data_Grove_z2['col5'] / data_Grove_z2['col6']

# Take log and matrixlze the data
logu_025, logu_050, logu_z1, logu_z2 = logu_025.reshape((4, 13)), logu_050.reshape((4, 13)), \
                                       logu_z1.reshape((4, 13)), logu_z2.reshape((4, 13))
OIII_OII_025, OIII_OII_050, OIII_OII_z1, OIII_OII_z2 = OIII_OII_025.reshape((4, 13)), \
                                                       OIII_OII_050.reshape((4, 13)), \
                                                       OIII_OII_z1.reshape((4, 13)), \
                                                       OIII_OII_z2.reshape((4, 13))
NII_Halpha_025, NII_Halpha_050, NII_Halpha_z1, NII_Halpha_z2 = NII_Halpha_025.reshape((4, 13)), \
                                                               NII_Halpha_050.reshape((4, 13)), \
                                                               NII_Halpha_z1.reshape((4, 13)), \
                                                               NII_Halpha_z2.reshape((4, 13))


fig, ax = plt.subplots(2, 1, figsize=(5, 10), dpi=300, sharex=True)
fig.subplots_adjust(hspace=0.)
x_1, x_2, x_3, x_4 = np.log10(OIII_OII_025), np.log10(OIII_OII_050), np.log10(OIII_OII_z1), np.log10(OIII_OII_z2)
y_1, y_2, y_3, y_4 = np.log10(NII_Halpha_025), np.log10(NII_Halpha_050), np.log10(NII_Halpha_z1), np.log10(NII_Halpha_z2)
y_1_sort, y_2_sort, y_3_sort, y_4_sort = np.sort(y_1, axis=1), np.sort(y_2, axis=1), \
                                         np.sort(y_3, axis=1), np.sort(y_4, axis=1)
x_1_sort, x_2_sort, x_3_sort, x_4_sort =  np.take_along_axis(x_1, np.argsort(y_1, axis=1), axis=1), \
                                          np.take_along_axis(x_2, np.argsort(y_2, axis=1), axis=1), \
                                          np.take_along_axis(x_3, np.argsort(y_3, axis=1), axis=1), \
                                          np.take_along_axis(x_4, np.argsort(y_4, axis=1), axis=1)
x_1_sort, x_2_sort, x_3_sort, x_4_sort = x_1_sort[:, :], x_2_sort[:, :], x_3_sort[:, 3:], x_4_sort[:, 5:]
y_1_sort, y_2_sort, y_3_sort, y_4_sort = y_1_sort[:, :], y_2_sort[:, :], y_3_sort[:, 3:], y_4_sort[:, 5:]
ax[0].plot(x_1_sort[0, :], y_1_sort[0, :], color='C1', alpha=0.4)
ax[0].plot(x_1_sort[1, :], y_1_sort[1, :], color='C1', alpha=0.4)
ax[0].plot(x_1_sort[2, :], y_1_sort[2, :], color='C1', alpha=0.4)
ax[0].plot(x_2_sort[0, :], y_2_sort[0, :], color='C3', alpha=0.4)
ax[0].plot(x_2_sort[1, :], y_2_sort[1, :], color='C3', alpha=0.4)
ax[0].plot(x_2_sort[2, :], y_2_sort[2, :], color='C3', alpha=0.4)
ax[0].plot(x_3_sort[0, :], y_3_sort[0, :], color='C5', alpha=0.4)
ax[0].plot(x_3_sort[1, :], y_3_sort[1, :], color='C5', alpha=0.4)
ax[0].plot(x_3_sort[2, :], y_3_sort[2, :], color='C5', alpha=0.4)
ax[0].plot(x_4_sort[0, :], y_4_sort[0, :], color='C7', alpha=0.4)
ax[0].plot(x_4_sort[1, :], y_4_sort[1, :], color='C7', alpha=0.4)
ax[0].plot(x_4_sort[2, :], y_4_sort[2, :], color='C7', alpha=0.4)

for i in range(13):
    alpha_i = 0.5 + 0.2 * i / 13
    ax[0].plot([x_1_sort[0, i], x_1_sort[1, i]], [y_1_sort[0, i], y_1_sort[1, i]], '-', color='C1', alpha=alpha_i)
    ax[0].plot([x_1_sort[1, i], x_1_sort[2, i]], [y_1_sort[1, i], y_1_sort[2, i]], '-', color='C1', alpha=alpha_i)
    ax[0].plot([x_2_sort[0, i], x_2_sort[1, i]], [y_2_sort[0, i], y_2_sort[1, i]], '-', color='C3', alpha=alpha_i)
    ax[0].plot([x_2_sort[1, i], x_2_sort[2, i]], [y_2_sort[1, i], y_2_sort[2, i]], '-', color='C3', alpha=alpha_i)
    if i < 10:
        ax[0].plot([x_3_sort[0, i], x_3_sort[1, i]], [y_3_sort[0, i], y_3_sort[1, i]], '-', color='C5', alpha=alpha_i)
        ax[0].plot([x_3_sort[1, i], x_3_sort[2, i]], [y_3_sort[1, i], y_3_sort[2, i]], '-', color='C5', alpha=alpha_i)
    if i < 8:
        ax[0].plot([x_4_sort[0, i], x_4_sort[1, i]], [y_4_sort[0, i], y_4_sort[1, i]], '-', color='C7', alpha=alpha_i)
        ax[0].plot([x_4_sort[1, i], x_4_sort[2, i]], [y_4_sort[1, i], y_4_sort[2, i]], '-', color='C7', alpha=alpha_i)

for i in range(len(metal[:9]))[::2]:
    color = 'C' + str(i)
    ax[1].fill(np.hstack((OIII_OII[:, 0, i], OIII_OII[::-1, 1, i])),
             np.hstack((OIIOIII_Halpha[:, 0, i], OIIOIII_Halpha[::-1, 1, i])),
             color=color, alpha=0.2)
    ax[1].fill(np.hstack((OIII_OII[:, 1, i], OIII_OII[::-1, 2, i])),
             np.hstack((OIIOIII_Halpha[:, 1, i], OIIOIII_Halpha[::-1, 2, i])),
             color=color, alpha=0.4)
    ax[1].plot(OIII_OII[:, :3, i].T, OIIOIII_Halpha[:, :3, i].T, '-', color=color)

ax[0].minorticks_on()
ax[1].minorticks_on()
ax[0].set_xlim(0, 1.6)
ax[0].set_ylabel(r'$\mathrm{log([N \, II] \lambda 6585 / H\alpha)}$', size=25)
ax[1].set_xlabel(r'$\mathrm{log([O \, III]  / [O \, II])}$', size=25)
ax[1].set_ylabel(r'$\mathrm{log([O \, III] + [O \, II] / H\alpha)}$', size=25)
ax[0].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                labelsize=20, size=5)
ax[0].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on', size=3)
ax[1].tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                labelsize=20, size=5)
ax[1].tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on', size=3)
plt.legend()
# plt.savefig('../../Proposal/HST+JWST/JWST_proposal.png', bbox_inches='tight')



# ============================================================
# Combined figure:
# left-top:    [O III] temperature diagnostic
# left-bottom: [O II] density diagnostic
# right:       Cloudy/Grove diagnostic, keeping its two panels
# ============================================================

fig = plt.figure(figsize=(13, 8), dpi=300)

gs = fig.add_gridspec(
    2, 2,
    width_ratios=[1.0, 1.25],
    height_ratios=[1, 1],
    wspace=0.28,
    hspace=0.18
)

ax_oiii = fig.add_subplot(gs[0, 0])
ax_oii  = fig.add_subplot(gs[1, 0])

gs_right = gs[:, 1].subgridspec(2, 1, hspace=0.0)
ax_grove  = fig.add_subplot(gs_right[0, 0])
ax_cloudy = fig.add_subplot(gs_right[1, 0], sharex=ax_grove)

# ============================================================
# Top-left: [O III] vs temperature
# ============================================================

tem_array = 10 ** np.linspace(3.75, 4.5, 1000)
den_array = np.array([0.1, 1, 10, 100])

OIII4363 = O3.getEmissivity(tem=tem_array, den=den_array, wave=4363)
OIII5007 = O3.getEmissivity(tem=tem_array, den=den_array, wave=5007)

for i in range(len(den_array)):
    ax_oiii.plot(
        np.log10(tem_array),
        np.log10(OIII4363 / OIII5007)[:, i],
        '-',
        label=r'$\mathrm{log}(n_{e}/\rm{cm^{-3}})=$ '
              + '{}'.format(np.log10(den_array[i]))
    )

ax_oiii.set_xlabel(r"$\mathrm{log(Temperature / K)}$", size=20)
ax_oiii.set_ylabel(
    r'$\mathrm{log( [O \, III] \lambda 4363 / \lambda 5007)}$',
    size=20
)
ax_oiii.legend(prop={'size': 11}, framealpha=0, loc=4)
ax_oiii.minorticks_on()

# ============================================================
# Bottom-left: [O II] vs density
# ============================================================

den_array = 10 ** np.linspace(-1, 4, 1000)
tem_array = 10 ** np.array([3.75, 4.00, 4.25, 4.50])

OII3727 = O2.getEmissivity(tem=tem_array, den=den_array, wave=3726)
OII3729 = O2.getEmissivity(tem=tem_array, den=den_array, wave=3729)

for i in range(len(tem_array)):
    ax_oii.plot(
        np.log10(den_array),
        np.log10(OII3729 / OII3727)[i, :],
        '-',
        label=r'$\mathrm{log} (T/\rm{K})=$ '
              + '{:.2f}'.format(np.log10(tem_array[i]))
    )

ax_oii.set_xlabel(r"$\mathrm{log(Density / cm^{-3})}$", size=20)
ax_oii.set_ylabel(
    r'$\mathrm{log( [O \, II] \lambda 3729 / \lambda 3727)}$',
    size=20
)
ax_oii.legend(prop={'size': 11}, framealpha=0, loc=3)
ax_oii.minorticks_on()

# ============================================================
# Right-top: Grove diagnostic
# ============================================================

ax_grove.plot(x_1_sort[0, :], y_1_sort[0, :], color='C1', alpha=0.4)
ax_grove.plot(x_1_sort[1, :], y_1_sort[1, :], color='C1', alpha=0.4)
ax_grove.plot(x_1_sort[2, :], y_1_sort[2, :], color='C1', alpha=0.4)

ax_grove.plot(x_2_sort[0, :], y_2_sort[0, :], color='C3', alpha=0.4)
ax_grove.plot(x_2_sort[1, :], y_2_sort[1, :], color='C3', alpha=0.4)
ax_grove.plot(x_2_sort[2, :], y_2_sort[2, :], color='C3', alpha=0.4)

ax_grove.plot(x_3_sort[0, :], y_3_sort[0, :], color='C5', alpha=0.4)
ax_grove.plot(x_3_sort[1, :], y_3_sort[1, :], color='C5', alpha=0.4)
ax_grove.plot(x_3_sort[2, :], y_3_sort[2, :], color='C5', alpha=0.4)

ax_grove.plot(x_4_sort[0, :], y_4_sort[0, :], color='C7', alpha=0.4)
ax_grove.plot(x_4_sort[1, :], y_4_sort[1, :], color='C7', alpha=0.4)
ax_grove.plot(x_4_sort[2, :], y_4_sort[2, :], color='C7', alpha=0.4)

for i in range(13):
    alpha_i = 0.5 + 0.2 * i / 13

    ax_grove.plot(
        [x_1_sort[0, i], x_1_sort[1, i]],
        [y_1_sort[0, i], y_1_sort[1, i]],
        '-',
        color='C1',
        alpha=alpha_i
    )
    ax_grove.plot(
        [x_1_sort[1, i], x_1_sort[2, i]],
        [y_1_sort[1, i], y_1_sort[2, i]],
        '-',
        color='C1',
        alpha=alpha_i
    )

    ax_grove.plot(
        [x_2_sort[0, i], x_2_sort[1, i]],
        [y_2_sort[0, i], y_2_sort[1, i]],
        '-',
        color='C3',
        alpha=alpha_i
    )
    ax_grove.plot(
        [x_2_sort[1, i], x_2_sort[2, i]],
        [y_2_sort[1, i], y_2_sort[2, i]],
        '-',
        color='C3',
        alpha=alpha_i
    )

    if i < 10:
        ax_grove.plot(
            [x_3_sort[0, i], x_3_sort[1, i]],
            [y_3_sort[0, i], y_3_sort[1, i]],
            '-',
            color='C5',
            alpha=alpha_i
        )
        ax_grove.plot(
            [x_3_sort[1, i], x_3_sort[2, i]],
            [y_3_sort[1, i], y_3_sort[2, i]],
            '-',
            color='C5',
            alpha=alpha_i
        )

    if i < 8:
        ax_grove.plot(
            [x_4_sort[0, i], x_4_sort[1, i]],
            [y_4_sort[0, i], y_4_sort[1, i]],
            '-',
            color='C7',
            alpha=alpha_i
        )
        ax_grove.plot(
            [x_4_sort[1, i], x_4_sort[2, i]],
            [y_4_sort[1, i], y_4_sort[2, i]],
            '-',
            color='C7',
            alpha=alpha_i
        )

ax_grove.set_xlim(0, 1.6)
ax_grove.set_ylabel(
    r'$\mathrm{log([N \, II] \lambda 6585 / H\alpha)}$',
    size=20
)
ax_grove.minorticks_on()
plt.setp(ax_grove.get_xticklabels(), visible=False)

# ============================================================
# Right-bottom: Cloudy diagnostic
# ============================================================

for i in range(len(metal[:9]))[::2]:
    color = 'C' + str(i)

    ax_cloudy.fill(
        np.hstack((OIII_OII[:, 0, i], OIII_OII[::-1, 1, i])),
        np.hstack((OIIOIII_Halpha[:, 0, i], OIIOIII_Halpha[::-1, 1, i])),
        color=color,
        alpha=0.2
    )

    ax_cloudy.fill(
        np.hstack((OIII_OII[:, 1, i], OIII_OII[::-1, 2, i])),
        np.hstack((OIIOIII_Halpha[:, 1, i], OIIOIII_Halpha[::-1, 2, i])),
        color=color,
        alpha=0.4
    )

    ax_cloudy.plot(
        OIII_OII[:, :3, i].T,
        OIIOIII_Halpha[:, :3, i].T,
        '-',
        color=color
    )

ax_cloudy.set_xlim(0, 1.6)
ax_cloudy.set_xlabel(
    r'$\mathrm{log([O \, III]  / [O \, II])}$',
    size=20
)
ax_cloudy.set_ylabel(
    r'$\mathrm{log([O \, III] + [O \, II] / H\alpha)}$',
    size=20
)
ax_cloudy.minorticks_on()

# ============================================================
# Shared formatting
# ============================================================

for ax_i in [ax_oiii, ax_oii, ax_grove, ax_cloudy]:
    ax_i.tick_params(
        axis='both',
        which='major',
        direction='in',
        top=True,
        bottom=True,
        left=True,
        right=True,
        labelsize=15,
        size=5
    )
    ax_i.tick_params(
        axis='both',
        which='minor',
        direction='in',
        top=True,
        bottom=True,
        left=True,
        right=True,
        size=3
    )

ax_oiii.text(0.05, 0.92, r'[O III]', transform=ax_oiii.transAxes, size=18)
ax_oii.text(0.05, 0.92, r'[O II]', transform=ax_oii.transAxes, size=18)
ax_grove.text(0.05, 0.90, r'Grove models', transform=ax_grove.transAxes, size=18)
ax_cloudy.text(0.05, 0.90, r'Cloudy models', transform=ax_cloudy.transAxes, size=18)

fig.savefig(
    '../../Papers/Thesis/combined_pyneb_cloudy.png',
    bbox_inches='tight'
)