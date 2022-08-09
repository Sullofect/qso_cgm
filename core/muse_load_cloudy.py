import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)


def load_cloudy(filename='alpha_2.0'):
    # Ionization parameter
    u_10 = np.array([])
    u_40 = np.array([])
    f = open('/Users/lzq/Dropbox/Data/CGM/cloudy/' + filename + '.out', 'r')
    for i, d in enumerate(f.readlines()):
        d = d.split(' ')
        if len(d) > 15:
            if d[11].startswith('U(1.0----)'):
                u_10 = np.hstack((u_10, d[11].split(':')[1]))
                u_40 = np.hstack((u_40, d[14].split(':')[1]))
    # Load alpha
    alpha = np.ones(len(u_10)) * float(filename.split('_')[1])

    # H den Grid parameter
    grid = np.genfromtxt('/Users/lzq/Dropbox/Data/CGM/cloudy/' + filename + '.grd', delimiter=None)
    hden_grid = grid[:, 6]

    # Line profile
    line = np.genfromtxt('/Users/lzq/Dropbox/Data/CGM/cloudy/' + filename + '.lin', delimiter=None)
    OIII5008, OIII4960, OIIIboth = line[:, 2], line[:, 3], line[:, 4]
    OII3727, OII3730, Hbeta = line[:, 5], line[:, 6], line[:, 7]

    # Temperature profile
    temp = np.genfromtxt('/Users/lzq/Dropbox/Data/CGM/cloudy/' + filename + '.avr', delimiter=None)

    data = np.vstack((alpha, hden_grid, u_10, u_40, OIII5008, Hbeta, OIII5008, OII3727 + OII3730, temp))

    return data.astype(float).T


def format_cloudy(filename=None):
    for i in range(len(filename)):
        if i == 0:
            output = load_cloudy(filename[i])
        else:
            c_i = load_cloudy(filename[i])
            output = np.vstack((output, c_i))
    return output

filename = ['alpha_2.0', 'alpha_1.7', 'alpha_1.4', 'alpha_1.2']
output = format_cloudy(filename=filename)

# Load grid, plot line ratio density, ionization parameter !!!
alpha = output[:, 0]
hden = output[:, 1]
logu = np.log10(output[:, 2])
OIII_Hbeta = output[:, 4] / output[:, 5]
OIII_OII = output[:, 6] / output[:, 7]
temp = output[:, 8]
alpha_mat = alpha.reshape((len(filename), 21))
temp_mat = temp.reshape((len(filename), 21))
OIII_Hbeta_mat = OIII_Hbeta.reshape((len(filename), 21))
OIII_OII_mat = OIII_OII.reshape((len(filename), 21))

# print(output[:, 1])
# # print(alpha)
# # print(logu)
# print(OIII_Hbeta)
# print(alpha_mat)

plt.figure(figsize=(8, 5), dpi=300)
for i in range(len(filename)):
    plt.plot(hden[:21], temp_mat[i, :], '-', label=r'$\mathrm{Alpha = }$' + str(alpha_mat[i, 0]))
# plt.minorticks_on()
plt.tick_params(axis='both', which='major', direction='in', bottom='on', top='on', left='on', right='on', size=5,
                  labelsize=15)
# plt.tick_params(axis='both', which='minor', direction='in', bottom='on', top='on', left='on', right='on', size=3)
plt.xlabel(r"$\mathrm{log_{10}[Hydrogen \, density]}$", size=15)
plt.ylabel(r"$\mathrm{Temperature \, [K]}$", size=15)
plt.legend(prop={'size': 15}, framealpha=0, loc=1, fontsize=15)
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/cloudy_temp_grid', bbox_inches='tight')


# Plot line ratio
f, axarr = plt.subplots(1, 1, figsize=(8, 5), dpi=300)
# axarr.plot(np.log10(OIII_Hbeta), np.log10(OIII_OII), '.')
x_1 = np.log10(OIII_Hbeta_mat)
y_1 = np.log10(OIII_OII_mat)
y_1_sort = np.sort(y_1, axis=1)
x_1_sort = np.take_along_axis(x_1, np.argsort(y_1, axis=1), axis=1)
axarr.plot(np.log10(OIII_Hbeta_mat), np.log10(OIII_OII_mat), '-', color='grey', lw=1, alpha=0.3)
axarr.fill(np.hstack((x_1_sort[0, :], x_1_sort[1, ::-1])), np.hstack((y_1_sort[0, :], y_1_sort[1, ::-1])),
           color='grey', alpha=0.2)
axarr.fill(np.hstack((x_1_sort[1, :], x_1_sort[2, ::-1])), np.hstack((y_1_sort[1, :], y_1_sort[2, ::-1])),
           color='grey', alpha=0.4)
axarr.fill(np.hstack((x_1_sort[2, 2:], x_1_sort[3, ::-1][2:])), np.hstack((y_1_sort[2, 2:], y_1_sort[3, ::-1][2:])),
           color='grey', alpha=0.6)
axarr.set_xlabel(r'$\mathrm{log([O \, III]\lambda5008 / H\beta)}$', size=20, y=0.02)
axarr.set_ylabel(r'$\mathrm{log([O \, III]\lambda5008  / [O \, II] \lambda \lambda 3727,29)}$', size=20, x=0.05)
axarr.minorticks_on()
axarr.tick_params(axis='both', which='major', direction='in', top='on', bottom='on', left='on', right='on',
                  labelsize=20, size=5)
axarr.tick_params(axis='both', which='minor', direction='in', top='on', bottom='on', left='on', right='on',
                  size=3)
axarr.set_xlim(-0.6, 1.4)
axarr.set_ylim(-2, 1)
plt.savefig('/Users/lzq/Dropbox/Data/CGM_plots/LineRatio_region_test', bbox_inches='tight')
