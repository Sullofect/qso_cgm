import os
import numpy as np
import astropy.io.fits as fits

def load_lineratio(region=None):
    path_fit_info_sr = os.path.join(os.sep, 'Users', 'lzq', 'Dropbox', 'Data', 'CGM',
                                    'moreline_profile_selected_region.fits')
    data_fit_info_sr = fits.getdata(path_fit_info_sr, ignore_missing_end=True)
    data_fit_info_sr = data_fit_info_sr[data_fit_info_sr['region'] == region]

    flux_Hbeta, dflux_Hbeta = data_fit_info_sr['flux_Hbeta'], data_fit_info_sr['dflux_Hbeta']
    flux_OII = data_fit_info_sr['flux_OII'] / flux_Hbeta
    dflux_OII = flux_OII * np.sqrt((data_fit_info_sr['dflux_OII'] / data_fit_info_sr['flux_OII']) ** 2
                                   + (dflux_Hbeta / flux_Hbeta) ** 2)
    r_OII, dr_OII = data_fit_info_sr['r_OII'], data_fit_info_sr['dr_OII']
    flux_NeV3346 = data_fit_info_sr['flux_NeV3346'] / flux_Hbeta
    dflux_NeV3346 = flux_NeV3346 * np.sqrt((data_fit_info_sr['dflux_NeV3346'] / data_fit_info_sr['flux_NeV3346']) ** 2
                                           + (dflux_Hbeta / flux_Hbeta) ** 2)
    flux_NeIII3869 = data_fit_info_sr['flux_NeIII3869'] / flux_Hbeta
    dflux_NeIII3869 = flux_NeIII3869 * np.sqrt(
        (data_fit_info_sr['dflux_NeIII3869'] / data_fit_info_sr['flux_NeIII3869']) ** 2
        + (dflux_Hbeta / flux_Hbeta) ** 2)
    flux_Hdel = data_fit_info_sr['flux_Hdel'] / flux_Hbeta
    dflux_Hdel = flux_Hdel * np.sqrt((data_fit_info_sr['dflux_Hdel'] / data_fit_info_sr['flux_Hdel']) ** 2
                                     + (dflux_Hbeta / flux_Hbeta) ** 2)
    flux_Hgam = data_fit_info_sr['flux_Hgam'] / flux_Hbeta
    dflux_Hgam = flux_Hgam * np.sqrt((data_fit_info_sr['dflux_Hgam'] / data_fit_info_sr['flux_Hgam']) ** 2
                                     + (dflux_Hbeta / flux_Hbeta) ** 2)
    flux_OIII4364 = data_fit_info_sr['flux_OIII4364'] / flux_Hbeta
    dflux_OIII4364 = flux_OIII4364 * np.sqrt(
        (data_fit_info_sr['dflux_OIII4364'] / data_fit_info_sr['flux_OIII4364']) ** 2
        + (dflux_Hbeta / flux_Hbeta) ** 2)
    flux_HeII4687 = data_fit_info_sr['flux_HeII4687'] / flux_Hbeta
    dflux_HeII4687 = flux_HeII4687 * np.sqrt(
        (data_fit_info_sr['dflux_HeII4687'] / data_fit_info_sr['flux_HeII4687']) ** 2
        + (dflux_Hbeta / flux_Hbeta) ** 2)
    flux_OIII5008 = data_fit_info_sr['flux_OIII5008'] / flux_Hbeta
    dflux_OIII5008 = flux_OIII5008 * np.sqrt(
        (data_fit_info_sr['dflux_OIII5008'] / data_fit_info_sr['flux_OIII5008']) ** 2
        + (dflux_Hbeta / flux_Hbeta) ** 2)

    # Take the log
    logflux_Hbeta, dlogflux_Hbeta = np.log10(flux_Hbeta), dflux_Hbeta / (flux_Hbeta * np.log(10))
    logflux_NeV3346, dlogflux_NeV3346 = np.log10(flux_NeV3346), np.sqrt(
        (dflux_NeV3346 / (flux_NeV3346 * np.log(10))) ** 2
        + 0.0 ** 2)
    logflux_OII, dlogflux_OII = np.log10(flux_OII), np.sqrt((dflux_OII / (flux_OII * np.log(10))) ** 2 + 0.0 ** 2)
    logr_OII, dlogr_OII = np.log10(r_OII), np.sqrt((dr_OII / (r_OII * np.log(10))) ** 2 + 0.0 ** 2)
    logflux_NeIII3869, dlogflux_NeIII3869 = np.log10(flux_NeIII3869), np.sqrt((dflux_NeIII3869 /
                                                                               (flux_NeIII3869 * np.log(10))) ** 2
                                                                              + 0.0 ** 2)
    logflux_Hdel, dlogflux_Hdel = np.log10(flux_Hdel), np.sqrt((dflux_Hdel / (flux_Hdel * np.log(10))) ** 2 + 0.0 ** 2)
    logflux_Hgam, dlogflux_Hgam = np.log10(flux_Hgam), np.sqrt((dflux_Hgam / (flux_Hgam * np.log(10))) ** 2 + 0.0 ** 2)
    logflux_OIII4364, dlogflux_OIII4364 = np.log10(flux_OIII4364), np.sqrt((dflux_OIII4364 /
                                                                            (flux_OIII4364 * np.log(
                                                                                10))) ** 2 + 0.0 ** 2)
    logflux_HeII4687, dlogflux_HeII4687 =  np.log10(flux_HeII4687), np.sqrt((dflux_HeII4687 /
                                                                             (flux_HeII4687 * np.log(10))) ** 2 + 0.0 ** 2)
    logflux_OIII5008, dlogflux_OIII5008 = np.log10(flux_OIII5008), np.sqrt((dflux_OIII5008 /
                                                                            (flux_OIII5008 * np.log(
                                                                                10))) ** 2 + 0.0 ** 2)

    return logflux_Hbeta, dlogflux_Hbeta, logflux_NeV3346, dlogflux_NeV3346, logflux_OII, dlogflux_OII, logr_OII, \
           dlogr_OII, logflux_NeIII3869, dlogflux_NeIII3869, logflux_Hdel, dlogflux_Hdel, logflux_Hgam, dlogflux_Hgam, \
           logflux_OIII4364, dlogflux_OIII4364, logflux_HeII4687, dlogflux_HeII4687, logflux_OIII5008, dlogflux_OIII5008

