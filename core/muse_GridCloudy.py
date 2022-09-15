import os
import emcee
import corner
import lmfit
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from scipy import interpolate
from matplotlib import rc
from PyAstronomy import pyasl
from mpdaf.obj import Cube, WCS, WaveCoord, iter_spe

# Calculate the distance to a specific region
z = 0.6282144177077355
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
d_l = cosmo.angular_diameter_distance(z=z)
ratio = (1 * u.radian).to(u.arcsec).value
arcsec_15 = (15 * d_l / ratio).to(u.kpc).value
ra_qso_muse, dec_qso_muse = 40.13564948691202, -18.864301804042814
ra_s2, dec_s2 =  40.1364401, -18.8655766

c_qso = SkyCoord(ra_qso_muse, dec_qso_muse, frame='icrs', unit='deg')
c_s2 = SkyCoord(ra_s2, dec_s2, frame='icrs', unit='deg')
ang_sep = c_s2.separation(c_qso).to(u.arcsec).value
distance = np.log10((ang_sep * d_l / ratio).to(u.cm).value)
# print(distance) = 23.049 = 23.05 for S2

#### Define the grid
# Luminosity, alpha=1.4, high/low cut (1000ev, 5ev converted to radberg),
# radius (fixed), density -2 to 2.5 delta 0.1 dex, metalicity -1.5 to 0.5 delta 0.1 dex,

z = np.array([-1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3,
              -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5])

for i in range(len(z)):
    lines = np.array(['Table power law spectral index -1.4, low=0.37, high=73.5 ',
                      'nuL(nu) = 46.54 at 1.0 Ryd',
                      'hden 4 vary',
                      'grid -2 2.5 0.1',
                      'save grid "alpha_1.4_' + str(z[i]) + '.grd"',
                      'metals ' + str(z[i]) + ' log',
                      'radius 23.05',
                      'iterative to convergence',
                      'save averages, file="alpha_1.4_' + str(z[i]) +  '.avr" last no clobber',
                      'temperature, hydrogen 1 over volume',
                      'end of averages',
                      'save line list "alpha_1.4_' + str(z[i]) + '.lin" from "linelist.dat" last'])
    np.savetxt('/Users/lzq/Dropbox/Data/CGM/cloudy/trial1/alpha_1.4_' + str(z[i]) + '.in', lines, fmt="%s")

####

# Load the actual measurement
# Load S2 line ratio
path_fit_info_sr = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM',
                                'moreline_profile_selected_region.fits')
data_fit_info_sr = fits.getdata(path_fit_info_sr, 0, ignore_missing_end=True)
r_OII = data_fit_info_sr[:, 11]
dr_OII = data_fit_info_sr[:, 25]
r_OIII = data_fit_info_sr[:, 8] / data_fit_info_sr[:, 13]
dr_OIII = r_OIII * np.sqrt((data_fit_info_sr[:, 22] / data_fit_info_sr[:, 8] ) ** 2
                           + (data_fit_info_sr[:, 27] / data_fit_info_sr[:, 13]) ** 2)
logr_OII = np.log10(r_OII)[0]
logr_OIII = np.log10(r_OIII)[0]
logdr_OII = dr_OII[0] / (r_OII[0] * np.log(10))
logdr_OIII = dr_OIII[0] / (r_OIII[0] * np.log(10))

# Load cloudy result











# Define the log likelihood function and run MCMC
def log_prob(x):
    logtem, logden = x[0], x[1]
    # if logtem < 3.5:
    #     return -np.inf
    # if logtem > 4.5:
    #     return -np.inf
    if logden < 1:
        return -np.inf
    # if logden > 2.0:
    #     return -np.inf
    else:
        return - 0.5 * (((f_OII((logtem, logden)) - logr_OII) / logdr_OII) ** 2
                         + ((f_OIII((logtem, logden)) - logr_OIII) / logdr_OIII) ** 2)