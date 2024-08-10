import os
import re
import glob
import lmfit
import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt

# Constants
c_kms = 2.998e5
wave_Halpha_vac = 6564.614
wave_NII_vac = 6585.27
def getSigma_LDSS3(wave, wave_vac, sigma_res):
    sort = np.argmin(np.abs(wave - wave_vac))
    return sigma_res[sort]

def Gaussian(wave_vac, sigma_res, z, sigma_kms, flux, wave_line_vac, a, b):
    wave_obs = wave_line_vac * (1 + z)
    sigma_A = np.sqrt(sigma_kms **2 + getSigma_LDSS3(wave_obs, wave_vac, sigma_res) ** 2) / c_kms * wave_obs
    peak = flux / np.sqrt(2 * sigma_A ** 2 * np.pi)
    gaussian = peak * np.exp(-(wave_vac - wave_obs) ** 2 / 2 / sigma_A ** 2)
    return gaussian + a * wave_vac + b

table_gals = ascii.read('../../MUSEQuBES+CUBS/MASSIVE/gals.csv')
object, z_gals = table_gals['Object Name'], table_gals['Redshift (z)']
_, object = map(list, zip(*(x.split(' ') for x in object)))
object = np.array(object).astype(np.float)
gals = glob.glob('../../MUSEQuBES+CUBS/MASSIVE/inspec/*.dat')

model = Gaussian
parameters = lmfit.Parameters()
parameters.add_many(('z', 0.005, True, 0, 0.1, None),
                    ('sigma_kms', 70, True, 50, 2000.0, None),
                    ('flux', 100, True, 0.0, None, None),
                    ('wave_line_vac', wave_NII_vac, False, None, None, None),
                    ('a', 0, True, None, None, None),
                    ('b', 1, True, None, None, None))


# Figure
wave_NII_array = np.linspace(wave_NII_vac - 20, wave_NII_vac + 20, 50)
# gals = gals[10:15]  # 5:6 10:15
for i, gal in enumerate(gals):
    gal_num = os.path.basename(gal).split('_')[0]
    gal_num = re.findall(r'\d+', gal_num)
    gal_num = np.array(gal_num).astype(np.float)
    i_sort = np.where(object == gal_num)
    z_i = z_gals[i_sort]
    data_massive = ascii.read(gal)
    parameters['z'].value = float(z_i)

    # Load data and normalize
    wave, flux, flux_err, weight, sigma_res = data_massive['col1'], data_massive['col2'], data_massive['col3'], \
                                        data_massive['col4'], data_massive['col5']
    # wave /= (1 + z_i)
    flux_err /= np.median(flux)
    flux /= np.median(flux)

    # mask
    mask = (wave / (1 + z_i) > wave_NII_vac - 20) * (wave / (1 + z_i) < wave_NII_vac + 20)
    wave_NII, flux_NII, flux_err_NII, sigma_res_NII = wave[mask], flux[mask], flux_err[mask], sigma_res[mask]

    spec_model = lmfit.Model(model, missing='drop', independent_vars=['wave_vac','sigma_res'])
    result = spec_model.fit(flux_NII, wave_vac=wave_NII, sigma_res=sigma_res_NII, params=parameters,
                            weights=1 / flux_err_NII)

    # Access the fitting results
    fit_success = result.success
    z, dz = result.best_values['z'], result.params['z'].stderr
    sigma, dsigma = result.best_values['sigma_kms'], result.params['sigma_kms'].stderr
    flux_fit, dflux_fit = result.best_values['flux'], result.params['flux'].stderr
    a, da = result.best_values['a'], result.params['a'].stderr
    b, db = result.best_values['b'], result.params['b'].stderr
    print(z, sigma, sigma * 2.5)

    # plt.plot(wave_NII, flux_NII)
    plt.figure()
    plt.plot(wave / (1 + z), flux, drawstyle='steps-mid')
    plt.plot(wave / (1 + z), flux_err + 0.5, '--')
    plt.plot(wave_NII / (1 + z), result.best_fit, 'r-', label=r'sigma={}, W80={}'.format(int(sigma), int(sigma * 2.5)))
    # plt.plot(wave_NII_array, Gaussian(wave_NII_array, sigma_res_i, z, sigma, flux_fit, wave_NII_vac, a, b), 'r-',
    #          label=r'sigma={}, W80={}'.format(int(sigma), int(sigma * 2.5)))
    plt.axvline(wave_NII_vac, color='grey', linestyle='--')
    plt.axvline(wave_Halpha_vac, color='grey', linestyle='--')
    plt.xlim(6500, 6700)
    plt.ylim(0.8, 1.5)
    plt.legend()
    plt.savefig(f'../../MUSEQuBES+CUBS/MASSIVE/plots/{int(gal_num[0])}_NII.png')
# plt.show()