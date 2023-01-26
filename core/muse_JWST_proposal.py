import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from muse_LoadCloudy import format_cloudy_nogrid
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)


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
# metal = np.linspace(-1.5, 0.5, 11, dtype='f2')
metal = np.linspace(-1.9, -0.3, 9, dtype='f2')

output = format_cloudy_nogrid(filename=[Hden, alpha, metal],
                              path='/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial + '/')[:, :, :, :9]
print(np.log10(U))
print(np.shape(output))
# output = output[:, :, :, ::2]
#
OIIOIII_Halpha = np.log10((10 ** output[3, :, :, :] + 10 ** output[9, :, :, :]) / 3)
OIII_OII = output[9, :, :, :] - output[3, :, :, :]

plt.figure(figsize=(8, 8), dpi=300)
for i in range(len(metal[:9]))[::2]:
    print(i)
    color = 'C' + str(i)
    # plt.plot(OIII_OII[:, 0, i], OIIOIII_Halpha[:, 0, i], '-', color=color, alpha=0,
    #          label=r'$\rm log(Z/Z_{\odot})=$' + str(metal[i]))
    plt.fill(np.hstack((OIII_OII[:, 0, i], OIII_OII[::-1, 1, i])),
             np.hstack((OIIOIII_Halpha[:, 0, i], OIIOIII_Halpha[::-1, 1, i])),
             color=color, alpha=0.2)
    plt.fill(np.hstack((OIII_OII[:, 1, i], OIII_OII[::-1, 2, i])),
             np.hstack((OIIOIII_Halpha[:, 1, i], OIIOIII_Halpha[::-1, 2, i])),
             color=color, alpha=0.4)
    # plt.fill(np.hstack((OIII_OII[:, 2, i], OIII_OII[::-1, 3, i])),
    #          np.hstack((OIIOIII_Halpha[:, 2, i], OIIOIII_Halpha[::-1, 3, i])),
    #          color=color, alpha=0.6)
    plt.plot(OIII_OII[:, :3, i].T, OIIOIII_Halpha[:, :3, i].T, '-', color=color)
    plt.annotate(r'$\rm log(Z/Z_{\odot})$=' + str(metal[i]),
                 xy=(1.5, OIIOIII_Halpha[0, 0, i]), xycoords='data', size=10)

# for i in range(4):
#     plt.annotate(r'$\rm \alpha$=' + str(alpha[i]),
#                  xy=(OIII_OII[len(logU) - 1, i, 0], OIIOIII_Halpha[len(logU) - 1, i, 0]),
#                  xycoords='data', size=10)

# for i in range(len(logU)):
#     plt.annotate(r'$\mathrm{log(U)}$=' + str(np.round(logU[i], 1)),
#                  xy=(OIII_OII[i, 2, 0], OIIOIII_Halpha[i, 2, 0] - 0.05),
#                  xycoords='data', size=10)

plt.minorticks_on()
plt.xlim(0, 1.85)
#plt.ylim(-0.7, 0.7)
plt.xlabel(r'$\mathrm{log([O \, III]  / [O \, II])}$', size=20)
plt.ylabel(r'$\mathrm{log([O \, III] + [O \, II] / H\alpha)}$', size=20)
plt.tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                labelsize=20, size=5)
plt.tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on', size=3)
plt.legend()
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/JWST_proposal.png', bbox_inches='tight')
