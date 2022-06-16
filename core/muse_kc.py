import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from scipy.interpolate import interp1d
from scipy.integrate import simps, trapz
from astropy.cosmology import FlatLambdaCDM



def load_filter(filtername=None):
    filter_info = np.loadtxt('/Users/lzq/Dropbox/Data/CGM/filters_kcor/' + filtername)
    filter = np.zeros(len(filter_info[:, 0]), dtype={'names': ('wave', 'transmission'), 'formats': (float, float)})
    filter['wave'] = filter_info[:, 0]
    filter['transmission'] = filter_info[:, 1]
    return filter


def rebin_filter(filter=None, wave_new=None):
    filter_interp = interp1d(filter['wave'], filter['transmission'], bounds_error=False, fill_value=0.0)
    filter_rebinned = np.zeros(len(wave_new), dtype={'names':('wave', 'transmission'), 'formats':(float, float)})
    filter_rebinned['wave'] = wave_new
    filter_rebinned['transmission'] = filter_interp(wave_new)
    # plt.plot(filter_rebinned['wave'], filter_rebinned['transmission'], '-')
    # plt.xlim(filter['wave'].min(), filter['wave'].max())
    # plt.show()
    return filter_rebinned


def rebin_filter_nu(filter=None, nu_new=None):
    filter_interp = interp1d(filter['nu'], filter['transmission'], bounds_error=False, fill_value=0.0)
    filter_rebinned = np.zeros(len(nu_new), dtype={'names':('wave', 'transmission'), 'formats':(float, float)})
    filter_rebinned['nu'] = nu_new
    filter_rebinned['wave'] = 3e18 / filter_rebinned['nu']
    filter_rebinned['transmission'] = filter_interp(nu_new)
    return filter_rebinned


def load_template(model=None):
    filename = os.environ['PYOBS'] + 'data/kcorrect/templates/' + model + '.npy'
    template = np.load(filename)

    return template


def KC(z=None, model=None, filter_o=None, filter_e=None):
    c_A = 2.9979e18

    # Load the filter and template
    filter_e_info = load_filter(filtername=filter_e)
    filter_o_info = load_filter(filtername=filter_o)
    template_e = load_template(model=model)

    # Redshift properly
    filter_o_z_info = load_filter(filtername=filter_o)
    filter_o_z_info['wave'] = filter_o_z_info['wave'] / (1.0 + z)

    # Rebin according to the template
    filter_e_rebin = rebin_filter(filter=filter_e_info, wave_new=template_e['wave'])
    filter_o_rebin = rebin_filter(filter=filter_o_z_info, wave_new=template_e['wave'])

    # Estimate Luminosity
    L_o = simps(template_e['flambda'] * filter_o_rebin['transmission'] * template_e['wave'], template_e['wave'])
    L_e = simps(template_e['flambda'] * filter_e_rebin['transmission'] * template_e['wave'], template_e['wave'])

    # Estimate zero point
    ZP_e = simps(filter_e_info['transmission'] / filter_e_info['wave'], filter_e_info['wave'])
    ZP_o = simps(filter_o_info['transmission'] / filter_o_info['wave'], filter_o_info['wave'])

    K = -2.5 * np.log10(L_o * ZP_e * (1 + z) / L_e / ZP_o)  # !!!!!!!! be careful of 1 + z right here!!!!!!

    return K


def app2abs(m_app=None, z=None, model=None, filter_o=None, filter_e=None):
    # Get the kcorrection term
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    d_l = cosmo.luminosity_distance(z).to(u.pc).value
    DM = 5 * np.log10(d_l / 10)
    K = KC(z=z, model=model, filter_o=filter_o, filter_e=filter_e)

    # Combine apparent magnitude, distance modulus, and kcorrection to get absolute magnitude
    abs = m_app - DM - K
    return abs


