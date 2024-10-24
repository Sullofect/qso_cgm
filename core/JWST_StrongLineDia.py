import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from muse_LoadCloudy import format_cloudy_nogrid
from astropy.io import ascii
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick', direction='in')
rc('ytick', direction='in')
rc('xtick.major', size=8)
rc('ytick.major', size=8)


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
# ax[0].fill(np.hstack((x_1_sort[0, :], x_1_sort[1, ::-1])), np.hstack((y_1_sort[0, :], y_1_sort[1, ::-1])),
#               color='C5', alpha=0.2)
# ax[0].fill(np.hstack((x_2_sort[0, :], x_2_sort[1, ::-1])), np.hstack((y_2_sort[0, :], y_2_sort[1, ::-1])),
#               color='C6', alpha=0.2)
# ax[0].fill(np.hstack((x_3_sort[0, :], x_3_sort[1, ::-1])), np.hstack((y_3_sort[0, :], y_3_sort[1, ::-1])),
#               color='C7', alpha=0.2)
# ax[0].fill(np.hstack((x_4_sort[0, :], x_4_sort[1, ::-1])), np.hstack((y_4_sort[0, :], y_4_sort[1, ::-1])),
#               color='C8', alpha=0.2)
ax[0].plot(x_1_sort[0, :], y_1_sort[0, :], color='C5', alpha=0.4)
ax[0].plot(x_1_sort[1, :], y_1_sort[1, :], color='C5', alpha=0.4)
ax[0].plot(x_1_sort[2, :], y_1_sort[2, :], color='C5', alpha=0.4)
ax[0].plot(x_2_sort[0, :], y_2_sort[0, :], color='C6', alpha=0.4)
ax[0].plot(x_2_sort[1, :], y_2_sort[1, :], color='C6', alpha=0.4)
ax[0].plot(x_2_sort[2, :], y_2_sort[2, :], color='C6', alpha=0.4)
ax[0].plot(x_3_sort[0, :], y_3_sort[0, :], color='C7', alpha=0.4)
ax[0].plot(x_3_sort[1, :], y_3_sort[1, :], color='C7', alpha=0.4)
ax[0].plot(x_3_sort[2, :], y_3_sort[2, :], color='C7', alpha=0.4)
ax[0].plot(x_4_sort[0, :], y_4_sort[0, :], color='C8', alpha=0.4)
ax[0].plot(x_4_sort[1, :], y_4_sort[1, :], color='C8', alpha=0.4)
ax[0].plot(x_4_sort[2, :], y_4_sort[2, :], color='C8', alpha=0.4)

# ax[0].fill(np.hstack((x_1_sort[1, :], x_1_sort[2, ::-1])), np.hstack((y_1_sort[1, :], y_1_sort[2, ::-1])),
#               color='C5', alpha=0.4)
# ax[0].fill(np.hstack((x_2_sort[1, :], x_2_sort[2, ::-1])), np.hstack((y_2_sort[1, :], y_2_sort[2, ::-1])),
#               color='C6', alpha=0.4)
# ax[0].fill(np.hstack((x_3_sort[1, :], x_3_sort[2, ::-1])), np.hstack((y_3_sort[1, :], y_3_sort[2, ::-1])),
#               color='C7', alpha=0.4)
# ax[0].fill(np.hstack((x_4_sort[1, :], x_4_sort[2, ::-1])), np.hstack((y_4_sort[1, :], y_4_sort[2, ::-1])),
#               color='C8', alpha=0.4)
# ax[0].fill(np.hstack((x_1_sort[2, :], x_1_sort[3, ::-1])), np.hstack((y_1_sort[2, :], y_1_sort[3, ::-1])),
#               color='C5', alpha=0.2)
# ax[0].fill(np.hstack((x_2_sort[2, :], x_2_sort[3, ::-1])), np.hstack((y_2_sort[2, :], y_2_sort[3, ::-1])),
#               color='C6', alpha=0.2)
# ax[0].fill(np.hstack((x_3_sort[2, :], x_3_sort[3, ::-1])), np.hstack((y_3_sort[2, :], y_3_sort[3, ::-1])),
#               color='C7', alpha=0.2)
# ax[0].fill(np.hstack((x_4_sort[2, :], x_4_sort[3, ::-1])), np.hstack((y_4_sort[2, :], y_4_sort[3, ::-1])),
#               color='C8', alpha=0.2)

for i in range(13):
    alpha_i = 0.5 + 0.2 * i / 13
    ax[0].plot([x_1_sort[0, i], x_1_sort[1, i]], [y_1_sort[0, i], y_1_sort[1, i]], '-', color='C5', alpha=alpha_i)
    ax[0].plot([x_2_sort[0, i], x_2_sort[1, i]], [y_2_sort[0, i], y_2_sort[1, i]], '-', color='C6', alpha=alpha_i)
    ax[0].plot([x_1_sort[1, i], x_1_sort[2, i]], [y_1_sort[1, i], y_1_sort[2, i]], '-', color='C5', alpha=alpha_i)
    ax[0].plot([x_2_sort[1, i], x_2_sort[2, i]], [y_2_sort[1, i], y_2_sort[2, i]], '-', color='C6', alpha=alpha_i)
    if i < 10:
        ax[0].plot([x_3_sort[0, i], x_3_sort[1, i]], [y_3_sort[0, i], y_3_sort[1, i]], '-', color='C7', alpha=alpha_i)
        ax[0].plot([x_3_sort[1, i], x_3_sort[2, i]], [y_3_sort[1, i], y_3_sort[2, i]], '-', color='C7', alpha=alpha_i)
    if i < 8:
        ax[0].plot([x_4_sort[0, i], x_4_sort[1, i]], [y_4_sort[0, i], y_4_sort[1, i]], '-', color='C8', alpha=alpha_i)
        ax[0].plot([x_4_sort[1, i], x_4_sort[2, i]], [y_4_sort[1, i], y_4_sort[2, i]], '-', color='C8', alpha=alpha_i)

for i in range(len(metal[:9]))[::2]:
    print(i)
    color = 'C' + str(i)
    # plt.plot(OIII_OII[:, 0, i], OIIOIII_Halpha[:, 0, i], '-', color=color, alpha=0,
    #          label=r'$\rm log(Z/Z_{\odot})=$' + s tr(metal[i]))
    ax[1].fill(np.hstack((OIII_OII[:, 0, i], OIII_OII[::-1, 1, i])),
             np.hstack((OIIOIII_Halpha[:, 0, i], OIIOIII_Halpha[::-1, 1, i])),
             color=color, alpha=0.2)
    ax[1].fill(np.hstack((OIII_OII[:, 1, i], OIII_OII[::-1, 2, i])),
             np.hstack((OIIOIII_Halpha[:, 1, i], OIIOIII_Halpha[::-1, 2, i])),
             color=color, alpha=0.4)
    # plt.fill(np.hstack((OIII_OII[:, 2, i], OIII_OII[::-1, 3, i])),
    #          np.hstack((OIIOIII_Halpha[:, 2, i], OIIOIII_Halpha[::-1, 3, i])),
    #          color=color, alpha=0.6)
    ax[1].plot(OIII_OII[:, :3, i].T, OIIOIII_Halpha[:, :3, i].T, '-', color=color)
    # plt.annotate(r'$\rm log(Z/Z_{\odot})$=' + str(metal[i]),
    #              xy=(1.5, OIIOIII_Halpha[0, 0, i]), xycoords='data', size=10)

# for i in range(4):
#     plt.annotate(r'$\rm \alpha$=' + str(alpha[i]),
#                  xy=(OIII_OII[len(logU) - 1, i, 0], OIIOIII_Halpha[len(logU) - 1, i, 0]),
#                  xycoords='data', size=10)

# for i in range(len(logU)):
#     plt.annotate(r'$\mathrm{log(U)}$=' + str(np.round(logU[i], 1)),
#                  xy=(OIII_OII[i, 2, 0], OIIOIII_Halpha[i, 2, 0] - 0.05),
#                  xycoords='data', size=10)

ax[0].minorticks_on()
ax[1].minorticks_on()
ax[0].set_xlim(0, 1.6)
#plt.ylim(-0.7, 0.7)
# ax[0].set_xlabel(r'$\mathrm{log([O \, III]  / [O \, II])}$', size=20)
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
plt.savefig('../../Proposal/HST+JWST/JWST_proposal.png', bbox_inches='tight')
