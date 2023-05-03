import os
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM


# load the region
path_region = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM', 'regions', 'gas_list_revised.reg')
ra_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 0]
dec_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 1]
radius_array = np.loadtxt(path_region, usecols=[0, 1, 2], delimiter=',')[:, 2]
text_array = np.loadtxt(path_region, dtype=str, usecols=[3], delimiter=',')

# Calculate the distance to a specific region
z = 0.6282144177077355
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
d_l = cosmo.angular_diameter_distance(z=z)
ratio = (1 * u.radian).to(u.arcsec).value
arcsec_15 = (15 * d_l / ratio).to(u.kpc).value
ra_qso_muse, dec_qso_muse = 40.13564948691202, -18.864301804042814
ra_s2, dec_s2 =  40.1364401, -18.8655766


c_qso = SkyCoord(ra_qso_muse, dec_qso_muse, frame='icrs', unit='deg')
c_s2 = SkyCoord(ra_array, dec_array, frame='icrs', unit='deg')
ang_sep = c_s2.separation(c_qso).to(u.arcsec).value
distance = np.log10((ang_sep * d_l / ratio).to(u.cm).value)
# print(distance, text_array)
# = 23.049 = 23.05 for S2
# = 22.753 = 22.75 for S8

#### Define the grid
### Trial 1:
# Luminosity, alpha=1.4, high/low cut (1000ev, 5ev converted to radberg),
# radius (fixed), density -2 to 2.5 delta 0.1 dex, metalicity -1.5 to 0.5 delta 0.1 dex,
# z = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
#               -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])
#
# for i in range(len(z)):
#     lines = np.array(['Table power law spectral index -1.4, low=0.37, high=73.5 ',
#                       'nuL(nu) = 46.54 at 1.0 Ryd',
#                       'hden 4 vary',
#                       'grid -2 2.5 0.1',
#                       'save grid "alpha_1.4_' + str(z[i]) + '.grd"',
#                       'metals ' + str(z[i]) + ' log',
#                       'radius 23.05',
#                       'iterative to convergence',
#                       'save averages, file="alpha_1.4_' + str(z[i]) +  '.avr" last no clobber',
#                       'temperature, hydrogen 1 over volume',
#                       'end of averages',
#                       'save line list "alpha_1.4_' + str(z[i]) + '.lin" from "linelist.dat" last'])
#     np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial1/alpha_1.4_' + str(z[i]) + '.in', lines, fmt="%s")
#
# ### Trial 2:
# # Luminosity, alpha=1.4, high/low cut (1000ev, 5ev converted to radberg),
# # radius (fixed), density -2 to 2.5 delta 0.1 dex, metalicity -1.5 to 0.5 delta 0.1 dex,
# alpha_array = np.array([-1.8, -1.75, -1.7, -1.65, -1.6, -1.55, -1.5, -1.45, -1.4, -1.35, -1.3, -1.25, -1.2])
# z = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
#               -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])
# for i in range(len(z)):
#     for j in range(len(alpha_array)):
#         lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) +', low=0.37, high=73.5 ',
#                           'nuL(nu) = 46.54 at 1.0 Ryd',
#                           'hden 4 vary',
#                           'grid -2 2.5 0.1',
#                           'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '.grd"',
#                           'metals ' + str(z[i]) + ' log',
#                           'radius 23.05',
#                           'iterative to convergence',
#                           'save averages, file="alpha_' + str(alpha_array[j])
#                           + '_' + str(z[i]) + '.avr" last no clobber',
#                           'temperature, hydrogen 1 over volume',
#                           'end of averages',
#                           'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
#                           + '.lin" from "linelist.dat" last'])
#         np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial2/alpha_' + str(alpha_array[j]) + '_'
#                    + str(z[i]) + '.in', lines, fmt="%s")
#
#
# ### Trial 2 Part 2:
# # Luminosity, alpha=1.4, high/low cut (1000ev, 5ev converted to radberg),
# # radius (fixed), density -2 to 2.5 delta 0.1 dex, metalicity -1.5 to 0.5 delta 0.1 dex,
# alpha_array = np.array([-1.2, -1.15, -1.1, -1.05, -1.0, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6])
# z = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
#               -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])
# for i in range(len(z)):
#     for j in range(len(alpha_array)):
#         lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) +', low=0.37, high=73.5 ',
#                           'nuL(nu) = 46.54 at 1.0 Ryd',
#                           'hden 4 vary',
#                           'grid -2 2.5 0.1',
#                           'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '.grd"',
#                           'metals ' + str(z[i]) + ' log',
#                           'radius 23.05',
#                           'iterative to convergence',
#                           'save averages, file="alpha_' + str(alpha_array[j])
#                           + '_' + str(z[i]) + '.avr" last no clobber',
#                           'temperature, hydrogen 1 over volume',
#                           'end of averages',
#                           'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
#                           + '.lin" from "linelist.dat" last'])
#         np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial2_p2/alpha_' + str(alpha_array[j]) + '_'
#                    + str(z[i]) + '.in', lines, fmt="%s")
#
# ### Trial 3
# # S8
# # Luminosity, alpha=1.4, high/low cut (1000ev, 5ev converted to radberg),
# # radius (fixed), density -2 to 2.5 delta 0.1 dex, metalicity -1.5 to 0.5 delta 0.1 dex,
# alpha_array = np.array([-1.2, -1.15, -1.1, -1.05, -1.0, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6])
# z = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
#               -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])
# for i in range(len(z)):
#     for j in range(len(alpha_array)):
#         lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) +', low=0.37, high=73.5 ',
#                           'nuL(nu) = 46.54 at 1.0 Ryd',
#                           'hden 4 vary',
#                           'grid -2 2.5 0.1',
#                           'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '.grd"',
#                           'metals ' + str(z[i]) + ' log',
#                           'radius 22.75',
#                           'iterative to convergence',
#                           'save averages, file="alpha_' + str(alpha_array[j])
#                           + '_' + str(z[i]) + '.avr" last no clobber',
#                           'temperature, hydrogen 1 over volume',
#                           'end of averages',
#                           'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
#                           + '.lin" from "linelist.dat" last'])
#         np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial3/alpha_' + str(alpha_array[j]) + '_'
#                    + str(z[i]) + '.in', lines, fmt="%s")
#
#
# ### Trial 4
# # S8
# # Luminosity, alpha=?, high/low cut (1000ev, 5ev converted to radberg),
# # radius (fixed), density -2 to 2.5 delta 0.1 dex, metalicity -1.5 to 0.5 delta 0.1 dex,
# alpha_array = np.array([-1.2, -1.15, -1.1, -1.05, -1.0, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6])
# z = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
#               -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])
# for i in range(len(z)):
#     for j in range(len(alpha_array)):
#         lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) +', low=0.37, high=73.5 ',
#                           'nuL(nu) = 45.54 at 1.0 Ryd',
#                           'hden 4 vary',
#                           'grid -2 2.5 0.1',
#                           'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '.grd"',
#                           'metals ' + str(z[i]) + ' log',
#                           'radius 22.75',
#                           'iterative to convergence',
#                           'save averages, file="alpha_' + str(alpha_array[j])
#                           + '_' + str(z[i]) + '.avr" last no clobber',
#                           'temperature, hydrogen 1 over volume',
#                           'end of averages',
#                           'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
#                           + '.lin" from "linelist.dat" last'])
#         np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial4/alpha_' + str(alpha_array[j]) + '_'
#                    + str(z[i]) + '.in', lines, fmt="%s")
#
# ### Trial 5
# # S8
# # Luminosity, alpha=?, high/low cut (1000ev, 5ev converted to radberg),
# # radius (fixed), density -2 to 2.5 delta 0.1 dex, metalicity -1.5 to 0.5 delta 0.1 dex,
# # alpha_array = np.array([-1.2, -1.15, -1.1, -1.05, -1.0, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6])
# alpha_array = np.linspace(-1.2, 0, 13, dtype='f2')
# # z = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
# #               -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])
# z = np.linspace(-1.5, 0.5, 11, dtype='f2')
# for i in range(len(z)):
#     for j in range(len(alpha_array)):
#         lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) + ', low=0.37, high=73.5 ',
#                           'nuL(nu) = 45.54 at 1.0 Ryd',
#                           'hden 4 vary',
#                           'grid -2 2.5 0.1',
#                           'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '.grd"',
#                           'metals ' + str(z[i]) + ' log',
#                           'radius 22.75',
#                           'iterative to convergence',
#                           'save averages, file="alpha_' + str(alpha_array[j])
#                           + '_' + str(z[i]) + '.avr" last no clobber',
#                           'temperature, hydrogen 1 over volume',
#                           'end of averages',
#                           'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
#                           + '.lin" from "linelist.dat" last'])
#         np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial5/alpha_' + str(alpha_array[j]) + '_'
#                    + str(z[i]) + '.in', lines, fmt="%s")
#
#
# ### Trial 6
# # S8
# # Luminosity, alpha=?, high/low cut (1000ev, 5ev converted to radberg),
# # radius (fixed), density -2 to 2.5 delta 0.1 dex, metalicity -1.5 to 0.5 delta 0.1 dex,
# # alpha_array = np.array([-1.2, -1.15, -1.1, -1.05, -1.0, -0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6])
# alpha_array = np.linspace(-1.8, 0, 10, dtype='f2')
# # z = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
# #               -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])
# z = np.linspace(-1.5, 0.5, 11, dtype='f2')
# for i in range(len(z)):
#     for j in range(len(alpha_array)):
#         lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) + ', low=0.37, high=73.5 ',
#                           'nuL(nu) = 45.54 at 1.0 Ryd',
#                           'hden 4 vary',
#                           'grid -2 2.5 0.2',
#                           'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '.grd"',
#                           'metals ' + str(z[i]) + ' log',
#                           'radius 22.75',
#                           'iterative to convergence',
#                           'save averages, file="alpha_' + str(alpha_array[j])
#                           + '_' + str(z[i]) + '.avr" last no clobber',
#                           'temperature, hydrogen 1 over volume',
#                           'end of averages',
#                           'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
#                           + '.lin" from "linelist.dat" last'])
#         np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial6/alpha_' + str(alpha_array[j]) + '_'
#                    + str(z[i]) + '.in', lines, fmt="%s")
#
#
# ### Trial 7
# ## S1
# z = np.linspace(-1.5, 0.5, 11, dtype='f2')
# alpha_array = np.linspace(-1.8, 0, 10, dtype='f2')
# den_array =  np.linspace(-2, 2.6, 24, dtype='f2')
# command_array = np.array([])
# for i in range(len(z)):
#     for j in range(len(alpha_array)):
#         for k in range(len(den_array)):
#             lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) + ', low=0.37, high=73.5 ',
#                               'nuL(nu) = 46.54 at 1.0 Ryd',
#                               'hden ' + str(den_array[k]),
#                               'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
#                               + '_' + str(den_array[k]) + '.grd"',
#                               'metals ' + str(z[i]) + ' log',
#                               'radius 23.15',
#                               'iterative to convergence',
#                               'save averages, file="alpha_' + str(alpha_array[j])
#                               + '_' + str(z[i]) + '_' + str(den_array[k]) + '.avr" last no clobber',
#                               'temperature, hydrogen 1 over volume',
#                               'end of averages',
#                               'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '_' + str(den_array[k])
#                               + '.lin" from "linelist.dat" last'])
#             np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial7/alpha_' + str(alpha_array[j]) + '_'
#                        + str(z[i]) + '_' + str(den_array[k]) + '.in', lines, fmt="%s")
#
#             # Command
#             command = np.array(['$cloudy -r ' + 'alpha_' + str(alpha_array[j]) + '_'
#                        + str(z[i]) + '_' + str(den_array[k])])
#
#             command_array = np.hstack((command_array, command))
# np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial7/command.txt', command_array, fmt="%s")
#
# ### S3 Trial 8
# ## S3
# z = np.linspace(-1.5, 0.5, 11, dtype='f2')
# alpha_array = np.linspace(-1.8, 0, 10, dtype='f2')
# den_array =  np.linspace(-2, 2.6, 24, dtype='f2')
# command_array = np.array([])
# for i in range(len(z)):
#     for j in range(len(alpha_array)):
#         for k in range(len(den_array)):
#             lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) + ', low=0.37, high=73.5 ',
#                               'nuL(nu) = 46.54 at 1.0 Ryd',
#                               'hden ' + str(den_array[k]),
#                               'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
#                               + '_' + str(den_array[k]) + '.grd"',
#                               'metals ' + str(z[i]) + ' log',
#                               'radius 22.89',
#                               'iterative to convergence',
#                               'save averages, file="alpha_' + str(alpha_array[j])
#                               + '_' + str(z[i]) + '_' + str(den_array[k]) + '.avr" last no clobber',
#                               'temperature, hydrogen 1 over volume',
#                               'end of averages',
#                               'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '_' + str(den_array[k])
#                               + '.lin" from "linelist.dat" last'])
#             np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/S3_t1/alpha_' + str(alpha_array[j]) + '_'
#                        + str(z[i]) + '_' + str(den_array[k]) + '.in', lines, fmt="%s")
#
#             # Command
#             command = np.array(['$cloudy -r ' + 'alpha_' + str(alpha_array[j]) + '_'
#                        + str(z[i]) + '_' + str(den_array[k])])
#
#             command_array = np.hstack((command_array, command))
# np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/S3_t1/command.txt', command_array, fmt="%s")
#
#
# ### S4 Trial 1
# ## S4
# z = np.linspace(-1.5, 0.5, 11, dtype='f2')
# alpha_array = np.linspace(-1.8, 0, 10, dtype='f2')
# den_array =  np.linspace(-2, 2.6, 24, dtype='f2')
# command_array = np.array([])
# for i in range(len(z)):
#     for j in range(len(alpha_array)):
#         for k in range(len(den_array)):
#             lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) + ', low=0.37, high=73.5 ',
#                               'nuL(nu) = 46.54 at 1.0 Ryd',
#                               'hden ' + str(den_array[k]),
#                               'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
#                               + '_' + str(den_array[k]) + '.grd"',
#                               'metals ' + str(z[i]) + ' log',
#                               'radius 22.72',
#                               'iterative to convergence',
#                               'save averages, file="alpha_' + str(alpha_array[j])
#                               + '_' + str(z[i]) + '_' + str(den_array[k]) + '.avr" last no clobber',
#                               'temperature, hydrogen 1 over volume',
#                               'end of averages',
#                               'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '_' + str(den_array[k])
#                               + '.lin" from "linelist.dat" last'])
#             np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/S4_t1/alpha_' + str(alpha_array[j]) + '_'
#                        + str(z[i]) + '_' + str(den_array[k]) + '.in', lines, fmt="%s")
#
#             # Command
#             command = np.array(['$cloudy -r ' + 'alpha_' + str(alpha_array[j]) + '_'
#                        + str(z[i]) + '_' + str(den_array[k])])
#
#             command_array = np.hstack((command_array, command))
# np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/S4_t1/command.txt', command_array, fmt="%s")
#
#
# ### S5 Trial 1
# ## S5
# z = np.linspace(-1.5, 0.5, 11, dtype='f2')
# alpha_array = np.linspace(-1.8, 0, 10, dtype='f2')
# den_array =  np.linspace(-2, 2.6, 24, dtype='f2')
# command_array = np.array([])
# for i in range(len(z)):
#     for j in range(len(alpha_array)):
#         for k in range(len(den_array)):
#             lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) + ', low=0.37, high=73.5 ',
#                               'nuL(nu) = 46.54 at 1.0 Ryd',
#                               'hden ' + str(den_array[k]),
#                               'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
#                               + '_' + str(den_array[k]) + '.grd"',
#                               'metals ' + str(z[i]) + ' log',
#                               'radius 22.44',
#                               'iterative to convergence',
#                               'save averages, file="alpha_' + str(alpha_array[j])
#                               + '_' + str(z[i]) + '_' + str(den_array[k]) + '.avr" last no clobber',
#                               'temperature, hydrogen 1 over volume',
#                               'end of averages',
#                               'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '_' + str(den_array[k])
#                               + '.lin" from "linelist.dat" last'])
#             np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/S5_t1/alpha_' + str(alpha_array[j]) + '_'
#                        + str(z[i]) + '_' + str(den_array[k]) + '.in', lines, fmt="%s")
#
#             # Command
#             command = np.array(['$cloudy -r ' + 'alpha_' + str(alpha_array[j]) + '_'
#                        + str(z[i]) + '_' + str(den_array[k])])
#
#             command_array = np.hstack((command_array, command))
# np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/S5_t1/command.txt', command_array, fmt="%s")
#
# ### S6 Trial 1
# ## S6
# z = np.linspace(-1.5, 0.5, 11, dtype='f2')
# alpha_array = np.linspace(-1.8, 0, 10, dtype='f2')
# den_array =  np.linspace(-2, 2.6, 24, dtype='f2')
# command_array = np.array([])
# for i in range(len(z)):
#     for j in range(len(alpha_array)):
#         for k in range(len(den_array)):
#             lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) + ', low=0.37, high=73.5 ',
#                               'nuL(nu) = 46.54 at 1.0 Ryd',
#                               'hden ' + str(den_array[k]),
#                               'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
#                               + '_' + str(den_array[k]) + '.grd"',
#                               'metals ' + str(z[i]) + ' log',
#                               'radius 22.80',
#                               'iterative to convergence',
#                               'save averages, file="alpha_' + str(alpha_array[j])
#                               + '_' + str(z[i]) + '_' + str(den_array[k]) + '.avr" last no clobber',
#                               'temperature, hydrogen 1 over volume',
#                               'end of averages',
#                               'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '_' + str(den_array[k])
#                               + '.lin" from "linelist.dat" last'])
#             np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/S6_t1/alpha_' + str(alpha_array[j]) + '_'
#                        + str(z[i]) + '_' + str(den_array[k]) + '.in', lines, fmt="%s")
#
#             # Command
#             command = np.array(['$cloudy -r ' + 'alpha_' + str(alpha_array[j]) + '_'
#                        + str(z[i]) + '_' + str(den_array[k])])
#
#             command_array = np.hstack((command_array, command))
# np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/S6_t1/command.txt', command_array, fmt="%s")
#
# ### S7 Trial 1
# ## S7
# z = np.linspace(-1.5, 0.5, 11, dtype='f2')
# alpha_array = np.linspace(-1.8, 0, 10, dtype='f2')
# den_array =  np.linspace(-2, 2.6, 24, dtype='f2')
# command_array = np.array([])
# for i in range(len(z)):
#     for j in range(len(alpha_array)):
#         for k in range(len(den_array)):
#             lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) + ', low=0.37, high=73.5 ',
#                               'nuL(nu) = 46.54 at 1.0 Ryd',
#                               'hden ' + str(den_array[k]),
#                               'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z[i])
#                               + '_' + str(den_array[k]) + '.grd"',
#                               'metals ' + str(z[i]) + ' log',
#                               'radius 22.94',
#                               'iterative to convergence',
#                               'save averages, file="alpha_' + str(alpha_array[j])
#                               + '_' + str(z[i]) + '_' + str(den_array[k]) + '.avr" last no clobber',
#                               'temperature, hydrogen 1 over volume',
#                               'end of averages',
#                               'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z[i]) + '_' + str(den_array[k])
#                               + '.lin" from "linelist.dat" last'])
#             np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/S7_t1/alpha_' + str(alpha_array[j]) + '_'
#                        + str(z[i]) + '_' + str(den_array[k]) + '.in', lines, fmt="%s")
#
#             # Command
#             command = np.array(['$cloudy -r ' + 'alpha_' + str(alpha_array[j]) + '_'
#                        + str(z[i]) + '_' + str(den_array[k])])
#
#             command_array = np.hstack((command_array, command))
# np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/S7_t1/command.txt', command_array, fmt="%s")

z_array = np.linspace(-1.5, 0.5, 11, dtype='f2')
alpha_array = np.linspace(-1.8, 0, 10, dtype='f2')
den_array = np.linspace(-2, 2.6, 24, dtype='f2')

def CreateGrid(z_array, alpha_array, den_array, L_qso=46.54, region=None, trial=None):
    global text_array
    dis = np.around(distance[text_array == region][0], decimals=2)
    command_array = np.array([])
    os.makedirs('/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial, exist_ok=True)
    os.popen('cp /Users/lzq/Dropbox/Data/CGM/cloudy/S1_t1/linelist.dat'
             ' /Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial + '/linelist.dat')
    for i in range(len(z_array)):
        for j in range(len(alpha_array)):
            for k in range(len(den_array)):
                lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) + ', low=0.37, high=73.5 ',
                                  'nuL(nu) = ' + str(L_qso) + ' at 1.0 Ryd',
                                  'hden ' + str(den_array[k]),
                                  'save grid "alpha_' + str(alpha_array[j]) + '_' + str(z_array[i])
                                  + '_' + str(den_array[k]) + '.grd"',
                                  'metals ' + str(z_array[i]) + ' log',
                                  'radius ' + str(dis),
                                  'iterative to convergence',
                                  'save averages, file="alpha_' + str(alpha_array[j])
                                  + '_' + str(z_array[i]) + '_' + str(den_array[k]) + '.avr" last no clobber',
                                  'temperature, hydrogen 1 over volume',
                                  'end of averages',
                                  'save line list "alpha_' + str(alpha_array[j]) + '_' + str(z_array[i]) + '_' + str(
                                      den_array[k])
                                  + '.lin" from "linelist.dat" last'])
                np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial
                           + '/alpha_' + str(alpha_array[j]) + '_'
                           + str(z_array[i]) + '_' + str(den_array[k]) + '.in', lines, fmt="%s")

                # Command
                command = np.array(['$cloudy -r ' + 'alpha_' + str(alpha_array[j]) + '_'
                                    + str(z_array[i]) + '_' + str(den_array[k])])

                command_array = np.hstack((command_array, command))
    np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial + '/command.txt', command_array, fmt="%s")

def CreateGrid_Emi(z_array, alpha_array, den_array, L_qso=46.54, region=None, trial=None):
    global text_array
    dis = np.around(distance[text_array == region][0], decimals=2)
    command_array = np.array([])
    os.makedirs('/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial, exist_ok=True)
    os.popen('cp /Users/lzq/Dropbox/Data/CGM/cloudy/S1_t1/linelist.dat'
             ' /Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial + '/linelist.dat')
    for i in range(len(z_array)):
        for j in range(len(alpha_array)):
            for k in range(len(den_array)):
                lines = np.array(['Table power law spectral index ' + str(alpha_array[j]) + ', low=0.37, high=73.5 ',
                                  'nuL(nu) = ' + str(L_qso) + ' at 1.0 Ryd',
                                  'hden ' + str(den_array[k]),
                                  'metals ' + str(z_array[i]) + ' log',
                                  'radius ' + str(dis),
                                  'iterative to convergence',
                                  'save line list absolute "alpha_' + str(alpha_array[j]) + '_' + str(z_array[i]) + '_'
                                  + str(den_array[k]) + '.lin" from "linelist.dat" last',
                                  'save lines, emissivity, ' + '"alpha_' + str(alpha_array[j]) + '_' + str(z_array[i])
                                  + '_' + str(den_array[k]) + '.emi" last',
                                  'Ne 5 3345.99A',
                                  'Blnd 3726A',
                                  'Blnd 3729A',
                                  'Ne 3 3868.76A',
                                  'He 1 3888.63A',
                                  'H  1 3970.07A',
                                  'H  1 4101.73A',
                                  'H  1 4340.46A',
                                  'O  3 4363.21A',
                                  'He 2 4685.64A',
                                  'H  1 4861.33A',
                                  'o  3 5006.84A',
                                  'o  3 4958.91A',
                                  'Blnd 5007.00A',
                                  'end of lines'])

                np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial
                           + '/alpha_' + str(alpha_array[j]) + '_'
                           + str(z_array[i]) + '_' + str(den_array[k]) + '.in', lines, fmt="%s")

                # Command
                command = np.array(['$cloudy -r ' + 'alpha_' + str(alpha_array[j]) + '_'
                                    + str(z_array[i]) + '_' + str(den_array[k])])

                command_array = np.hstack((command_array, command))
    np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial + '/command.txt', command_array, fmt="%s")

def CreateGrid_BB(z_array, T_array, den_array, L_qso=46.54, region=None, trial=None):
    global text_array
    dis = np.around(distance[text_array == region][0], decimals=2)
    command_array = np.array([])
    os.makedirs('/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial, exist_ok=True)
    os.popen('cp /Users/lzq/Dropbox/Data/CGM/cloudy/S1_t1/linelist.dat'
             ' /Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial + '/linelist.dat')
    for i in range(len(z_array)):
        for j in range(len(T_array)):
            for k in range(len(den_array)):
                lines = np.array(
                    ['black body, t=' + str(T_array[j]),
                     'nuL(nu) = ' + str(L_qso) + ' at 1.0 Ryd',
                     'hden ' + str(den_array[k]),
                     'metals ' + str(z_array[i]) + ' log',
                     'radius ' + str(dis),
                     'iterative to convergence',
                     'save line list "T_' + str(T_array[j]) + '_Z_' + str(z_array[i])
                     + '_n_' + str(den_array[k]) + '.lin" from "linelist.dat" last'])
                np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial
                           + '/T_' + str(T_array[j]) + '_Z_' + str(z_array[i])
                           + '_n_' + str(den_array[k]) + '.in', lines, fmt="%s")

                # Command
                command = np.array(['$cloudy -r ' + 'T_' + str(T_array[j]) + '_Z_' + str(z_array[i])
                                    + '_n_' + str(den_array[k])])

                command_array = np.hstack((command_array, command))
    np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial + '/command.txt', command_array,
               fmt="%s")

def CreateGrid_AGN(den_array, T_array, z_array, alpha_ox_array, alpha_uv_array, alpha_x_array,
                   L_qso=46.54, region=None, trial=None):
    global text_array
    dis = np.around(distance[text_array == region][0], decimals=2)
    command_array = np.array([])
    os.makedirs('/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial, exist_ok=True)
    os.popen('cp /Users/lzq/Dropbox/Data/CGM/cloudy/S1_t1/linelist.dat'
             ' /Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial + '/linelist.dat')
    for i in range(len(z_array)):
        for j in range(len(alpha_ox_array)):
            for k in range(len(den_array)):
                for ii in range(len(alpha_uv_array)):
                    for jj in range(len(alpha_x_array)):
                        for kk in range(len(T_array)):
                            lines = np.array(['AGN T = ' + str(T_array[kk]) + ', a(ox) = ' + str(alpha_ox_array[j])
                                              + ', a(uv) = ' + str(alpha_uv_array[ii]) + ', a(x) = '
                                              + str(alpha_x_array[jj]),
                                              'nuL(nu) = ' + str(L_qso) + ' at 1.0 Ryd',
                                              'hden ' + str(den_array[k]),
                                              'metals ' + str(z_array[i]) + ' log',
                                              'radius ' + str(dis),
                                              'iterative to convergence',
                                              'save line list "ox_' + str(alpha_ox_array[j])
                                              + 'uv_' + str(alpha_uv_array[ii])
                                              + 'x_' + str(alpha_x_array[jj])
                                              + 'T_' + str(T_array[kk])
                                              + 'Z_' + str(z_array[i])
                                              + 'n_' + str(den_array[k]) + '.lin" from "linelist.dat" last'])
                            np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial
                                       + '/ox_' + str(alpha_ox_array[j])
                                       + 'uv_' + str(alpha_uv_array[ii])
                                       + 'x_' + str(alpha_x_array[jj])
                                       + 'T_' + str(T_array[kk])
                                       + 'Z_' + str(z_array[i])
                                       + 'n_' + str(den_array[k]) + '.in', lines, fmt="%s")

                            # Command
                            command = np.array(['$cloudy -r '
                                                + 'ox_' + str(alpha_ox_array[j])
                                                + 'uv_' + str(alpha_uv_array[ii])
                                                + 'x_' + str(alpha_x_array[jj])
                                                + 'T_' + str(T_array[kk])
                                                + 'Z_' + str(z_array[i])
                                                + 'n_' + str(den_array[k])])

                            command_array = np.hstack((command_array, command))
    np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/' + region + '_' + trial + '/command.txt', command_array, fmt="%s")

# S7_t1 JWST proposal
# CreateGrid(np.linspace(-1.9, -1.7, 2, dtype='f2'), alpha_array[:3], den_array[17:25], region='S7', trial='t1_JWST')

# # S8
# CreateGrid(z_array, alpha_array, den_array, region='S8', trial='t1')
#
# # S9
# CreateGrid(z_array, alpha_array, den_array, region='S9', trial='t1')
#
# # S10
# CreateGrid(z_array, alpha_array, den_array, region='S10', trial='t1')
#
# # B1
# CreateGrid(z_array, alpha_array, den_array, region='B1', trial='t1')
#
# # B2
# CreateGrid(z_array, alpha_array, den_array, region='B2', trial='t1')
# den_array_2 = np.linspace(2.8, 4.6, 10, dtype='f2')
# CreateGrid(z_array, alpha_array, den_array_2, region='B2', trial='t1_2')
#
# B3
# CreateGrid(z_array, alpha_array, den_array, region='B3', trial='t1')
# den_array_2 = np.linspace(2.8, 4.6, 10, dtype='f2')
# CreateGrid(z_array, alpha_array, den_array_2, region='B3', trial='t1_2')
#
# # B4
# CreateGrid(z_array, alpha_array, den_array, region='B4', trial='t1')

# S5 extension
# den_array_2 =  np.linspace(2.6, 3.4, 5, dtype='f2')
# CreateGrid(z_array, alpha_array, den_array_2, region='S5', trial='t1_2')

# S6 extension
# CreateGrid(z_array, alpha_array, den_array_2, region='S6', trial='t1_2')

# S5 extension 2
# den_array_3 =  np.linspace(3.6, 6.6, 16, dtype='f2')
# CreateGrid(z_array, alpha_array, den_array_3, region='S5', trial='t1_3')

# S7 extension
# den_array_S7t1_2 = np.linspace(2.8, 4.6, 10, dtype='f2')
# CreateGrid(z_array, alpha_array, den_array_S7t1_2, region='S7', trial='t1_2')

# S8 extension
# den_array_S8t1_2 = np.linspace(2.8, 4.6, 10, dtype='f2')
# CreateGrid(z_array, alpha_array, den_array_S8t1_2, region='S8', trial='t1_2')

# S9 extension
# den_array_S9t1_2 = np.linspace(2.8, 4.6, 10, dtype='f2')
# CreateGrid(z_array, alpha_array, den_array_S9t1_2, region='S9', trial='t1_2')

# S10 extension
# den_array_S10t1_2 = np.linspace(2.8, 4.6, 10, dtype='f2')
# CreateGrid(z_array, alpha_array, den_array_S10t1_2, region='S10', trial='t1_2')

# dim qso by 10 times
# den_array_S5t2_2 = np.linspace(2.8, 4.6, 10, dtype='f2')
# CreateGrid(z_array, alpha_array, den_array_S5t2_2, L_qso=45.54, region='S5', trial='t2_2')
# CreateGrid(z_array, alpha_array, den_array, L_qso=45.54, region='S6', trial='t2')

#
# den_array_t2 = np.linspace(-2, 4.6, 34, dtype='f2')
# CreateGrid(z_array, alpha_array, den_array_t2, L_qso=45.54, region='S7', trial='t2')
# CreateGrid(z_array, alpha_array, den_array_t2, L_qso=45.54, region='S8', trial='t2')
# CreateGrid(z_array, alpha_array, den_array_t2, L_qso=45.54, region='S9', trial='t2')
# CreateGrid(z_array, alpha_array, den_array_t2, L_qso=45.54, region='S10', trial='t2')

# Check AGN continuum with S1
# z_array_AGN = np.linspace(-0.5, 0.5, 6, dtype='f2')
# den_array_AGN = np.linspace(1.0, 2.4, 8, dtype='f2')
# alpha_ox_array_AGN = np.linspace(-1.2, -0.2, 6, dtype='f2')
# alpha_uv_array_AGN = np.linspace(-1.0, 0, 3, dtype='f2')
# alpha_x_array_AGN = np.linspace(-1.5, -0.5, 3, dtype='f2')
# T_array_AGN = np.linspace(5, 5.5, 3, dtype='f2')
# CreateGrid_AGN(den_array_AGN, T_array_AGN, z_array_AGN, alpha_ox_array_AGN, alpha_uv_array_AGN, alpha_x_array_AGN,
#                region='S1', trial='AGN')


# Check AGN continuum trial 2 with S1
# den_array_AGN = np.linspace(1.0, 2.4, 8, dtype='f2')
# z_array_AGN = np.linspace(-0.5, 0.5, 6, dtype='f2')
# T_array_AGN = np.linspace(5, 5.5, 3, dtype='f2')
# alpha_ox_array_AGN = np.linspace(-1.2, 0, 7, dtype='f2')
# # alpha_uv_array_AGN = np.linspace(-1.5, 0.5, 11, dtype='f2')
# alpha_uv_array_AGN = np.array([-0.5], dtype='f2')
# alpha_x_array_AGN = np.linspace(-1.5, 0.5, 11, dtype='f2')
# # CreateGrid_AGN(den_array_AGN, T_array_AGN, z_array_AGN, alpha_ox_array_AGN, alpha_uv_array_AGN, alpha_x_array_AGN,
# #                region='S1', trial='AGN_2')
#
# # Add more grid point
# T_array_AGN_add = np.array([4.75, 5.75], dtype='f2')
# CreateGrid_AGN(den_array_AGN, T_array_AGN_add, z_array_AGN, alpha_ox_array_AGN, alpha_uv_array_AGN, alpha_x_array_AGN,
#                region='S1', trial='AGN_2_2')

# Check AGN continuum with S6
# den_array_AGN = np.linspace(1.0, 2.4, 8, dtype='f2')
# z_array_AGN = np.linspace(-0.5, 0.5, 6, dtype='f2')
# T_array_AGN = np.linspace(4.75, 5.75, 5, dtype='f2')
# alpha_ox_array_AGN = np.linspace(-1.2, 0, 7, dtype='f2')
# alpha_uv_array_AGN = np.array([-0.5], dtype='f2')
# alpha_x_array_AGN = np.linspace(-1.5, 0.5, 11, dtype='f2')
# CreateGrid_AGN(den_array_AGN, T_array_AGN, z_array_AGN, alpha_ox_array_AGN, alpha_uv_array_AGN, alpha_x_array_AGN,
#                region='S6', trial='AGN')

# Check AGN continuum with S6 extension
# den_array_AGN = np.linspace(2.6, 3.4, 5, dtype='f2')
# z_array_AGN = np.linspace(-0.5, 0.5, 6, dtype='f2')
# T_array_AGN = np.linspace(4.75, 5.75, 5, dtype='f2')
# alpha_ox_array_AGN = np.linspace(-1.2, 0, 7, dtype='f2')
# alpha_uv_array_AGN = np.array([-0.5], dtype='f2')
# alpha_x_array_AGN = np.linspace(-1.5, 0.5, 11, dtype='f2')
# CreateGrid_AGN(den_array_AGN, T_array_AGN, z_array_AGN, alpha_ox_array_AGN, alpha_uv_array_AGN, alpha_x_array_AGN,
#                region='S6', trial='AGN_2')


# Check AGN continuum with S9
# den_array_AGN = np.linspace(1.0, 4.6, 19, dtype='f2')
# z_array_AGN = np.linspace(-0.5, 0.5, 6, dtype='f2')
# T_array_AGN = np.linspace(4.75, 5.75, 5, dtype='f2')
# alpha_ox_array_AGN = np.linspace(-1.2, 0, 7, dtype='f2')
# alpha_uv_array_AGN = np.array([-0.5], dtype='f2')
# alpha_x_array_AGN = np.linspace(-1.5, 0.5, 11, dtype='f2')
# CreateGrid_AGN(den_array_AGN, T_array_AGN, z_array_AGN, alpha_ox_array_AGN, alpha_uv_array_AGN, alpha_x_array_AGN,
#                region='S9', trial='AGN')

# Crete BB for S2
# z_array = np.linspace(-1.5, 0.5, 11, dtype='f2')
# T_array = np.linspace(4, 6.5, 13, dtype='f2')
# den_array = np.linspace(-2, 2.6, 24, dtype='f2')
# CreateGrid_BB(z_array, T_array, den_array, L_qso=46.54, region='S2', trial='BB_t1')

# BB for S2 extension
# den_array_2 = np.linspace(2.8, 4.6, 10, dtype='f2')
# CreateGrid_BB(z_array, T_array, den_array_2, L_qso=46.54, region='S2', trial='BB_t1_2')


# # S1 Emissivity
# CreateGrid_Emi(z_array, alpha_array, den_array, region='S1', trial='t1_Emi')

# B3_new Emissivity
# den_array_B3B4_new = np.linspace(-2, 4.6, 34, dtype='f2')
# CreateGrid_Emi(z_array, alpha_array, den_array_B3B4_new, region='B3_new', trial='t1_Emi')

# B4_new Emissivity
# CreateGrid_Emi(z_array, alpha_array, den_array_B3B4_new, region='B4_new', trial='t1_Emi')

# S6 Emissivity
# den_array_S6_Emi = np.linspace(-2, 4.6, 34, dtype='f2')
# CreateGrid_Emi(z_array, alpha_array, den_array_S6_Emi, region='S6', trial='t1_Emi')

# S5 Emissivity
den_array_S5_Emi = np.linspace(-2, 5.6, 39, dtype='f2')
CreateGrid_Emi(z_array, alpha_array, den_array_S5_Emi, region='S5', trial='t1_Emi')
