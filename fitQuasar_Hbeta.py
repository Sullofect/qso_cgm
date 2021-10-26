import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt
import lmfit
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
import sys
import cosmography
from scipy.interpolate import UnivariateSpline
import extinction
from scipy.signal import medfilt

name = 'HE0238-1904'
redshift = 0.62832
Av = 0.087
 
template_FeII = Table.read('fe_optical.txt', format='ascii')

c_kms = 2.998e5
 
wave_rest_Hgamma   = 4341.68
wave_rest_Hbeta    = 4862.68
wave_rest_OIII4960 = 4960.30
wave_rest_OIII5008 = 5008.24
 
 
 
def model_broad(wave,
                velocity_b1, sigma_b1_kms, flux_Hbeta_b1,
                velocity_b2, sigma_b2_kms, flux_Hbeta_b2,
                velocity_b3, sigma_b3_kms, flux_Hbeta_b3):
    
   flux_Hbeta_b1 = 10.0**flux_Hbeta_b1
   flux_Hbeta_b2 = 10.0**flux_Hbeta_b2
   flux_Hbeta_b3 = 10.0**flux_Hbeta_b3
   
    
   # Hbeta 
   redshift_b1 = velocity_b1/c_kms
   wave_observed_Hbeta = (1 + redshift_b1)*wave_rest_Hbeta
   sigma_Hbeta_Ang = sigma_b1_kms/c_kms*wave_observed_Hbeta
   peakHbeta_b = flux_Hbeta_b1/np.sqrt(2*sigma_Hbeta_Ang**2*np.pi)
   gaussian_Hbeta_b1 = peakHbeta_b*np.exp(-(wave - wave_observed_Hbeta)**2/2/sigma_Hbeta_Ang**2)
   
   
   # Hbeta 
   redshift_b2 = velocity_b2/c_kms
   wave_observed_Hbeta = (1 + redshift_b2)*wave_rest_Hbeta
   sigma_Hbeta_Ang = sigma_b2_kms/c_kms*wave_observed_Hbeta
   peakHbeta_b = flux_Hbeta_b2/np.sqrt(2*sigma_Hbeta_Ang**2*np.pi)
   gaussian_Hbeta_b2 = peakHbeta_b*np.exp(-(wave - wave_observed_Hbeta)**2/2/sigma_Hbeta_Ang**2)
   
   # Hbeta 
   redshift_b3 = velocity_b3/c_kms
   wave_observed_Hbeta = (1 + redshift_b3)*wave_rest_Hbeta
   sigma_Hbeta_Ang = sigma_b3_kms/c_kms*wave_observed_Hbeta
   peakHbeta_b = flux_Hbeta_b3/np.sqrt(2*sigma_Hbeta_Ang**2*np.pi)
   gaussian_Hbeta_b3 = peakHbeta_b*np.exp(-(wave - wave_observed_Hbeta)**2/2/sigma_Hbeta_Ang**2)
   
    
   return gaussian_Hbeta_b1 + gaussian_Hbeta_b2 + gaussian_Hbeta_b3
 
 
 
def model_narrow(wave,
                 velocity_n1, sigma_n1_kms, flux_OIII5008_n1, flux_Hbeta_n1,
                 velocity_n2, sigma_n2_kms, flux_OIII5008_n2, flux_Hbeta_n2):
    
   
   flux_OIII5008_n1 = 10.0**flux_OIII5008_n1
   flux_Hbeta_n1 = 10.0**flux_Hbeta_n1
   flux_OIII5008_n2 = 10.0**flux_OIII5008_n2
   flux_Hbeta_n2 = 10.0**flux_Hbeta_n2
    
   # [O III]
   redshift_n1 = velocity_n1/c_kms
   wave_observed_OIII5008 = (1 + redshift_n1)*wave_rest_OIII5008
   sigma_OIII5008_Ang = sigma_n1_kms/c_kms*wave_observed_OIII5008
   peakOIII5008 = flux_OIII5008_n1/np.sqrt(2*sigma_OIII5008_Ang**2*np.pi)
   gaussian_OIII5008_n1 = peakOIII5008*np.exp(-(wave - wave_observed_OIII5008)**2/2/sigma_OIII5008_Ang**2)
    
    
   wave_observed_OIII4960 = (1 + redshift_n1)*wave_rest_OIII4960
   sigma_OIII4960_Ang = sigma_n1_kms/c_kms*wave_observed_OIII4960
   #peakOIII4960 = flux_OIII4960_n/np.sqrt(2*sigma_OIII4960_Ang**2*np.pi)
   gaussian_OIII4960_n1 = peakOIII5008/3.0*np.exp(-(wave - wave_observed_OIII4960)**2/2/sigma_OIII4960_Ang**2)
    
    
    
   # Hbeta
   wave_observed_Hbeta = (1 + redshift_n1)*wave_rest_Hbeta
   sigma_Hbeta_Ang = sigma_n1_kms/c_kms*wave_observed_Hbeta
   peakHbeta_n = flux_Hbeta_n1/np.sqrt(2*sigma_Hbeta_Ang**2*np.pi)
   gaussian_Hbeta_n1 = peakHbeta_n*np.exp(-(wave - wave_observed_Hbeta)**2/2/sigma_Hbeta_Ang**2)
 
    
    
    
   # [O III]
   redshift_n2 = velocity_n2/c_kms
   wave_observed_OIII5008 = (1 + redshift_n2)*wave_rest_OIII5008
   sigma_OIII5008_Ang = sigma_n2_kms/c_kms*wave_observed_OIII5008
   peakOIII5008 = flux_OIII5008_n2/np.sqrt(2*sigma_OIII5008_Ang**2*np.pi)
   gaussian_OIII5008_n2 = peakOIII5008*np.exp(-(wave - wave_observed_OIII5008)**2/2/sigma_OIII5008_Ang**2)
    
    
   wave_observed_OIII4960 = (1 + redshift_n2)*wave_rest_OIII4960
   sigma_OIII4960_Ang = sigma_n2_kms/c_kms*wave_observed_OIII4960
   #peakOIII4960 = flux_OIII4960_n/np.sqrt(2*sigma_OIII4960_Ang**2*np.pi)
   gaussian_OIII4960_n2 = peakOIII5008/3.0*np.exp(-(wave - wave_observed_OIII4960)**2/2/sigma_OIII4960_Ang**2)
    
    
   # Hbeta
   wave_observed_Hbeta = (1 + redshift_n2)*wave_rest_Hbeta
   sigma_Hbeta_Ang = sigma_n2_kms/c_kms*wave_observed_Hbeta
   peakHbeta_n = flux_Hbeta_n2/np.sqrt(2*sigma_Hbeta_Ang**2*np.pi)
   gaussian_Hbeta_n2 = peakHbeta_n*np.exp(-(wave - wave_observed_Hbeta)**2/2/sigma_Hbeta_Ang**2)
    
     
    
    
    
    
   return gaussian_OIII4960_n1 + gaussian_OIII5008_n1 + gaussian_Hbeta_n1 \
          + gaussian_OIII4960_n2 + gaussian_OIII5008_n2 + gaussian_Hbeta_n2
 
 
 
def model_FeII(wave, coeff_FeII, sigma_FeII, shift_FeII):
    
    
   template_FeII_interp = interp1d(10.0**(template_FeII['logwave'] + shift_FeII),
                                   gaussian_filter1d(template_FeII['flux'],
                                                    sigma_FeII, mode='nearest'))
                                                     
    
   FeII_model = coeff_FeII*template_FeII_interp(wave)
    
   return FeII_model
 
def model_powerlaw(wave, flambda5100, alpha):
    
   return flambda5100*(wave/5100.0)**alpha
 
 
 
def model(wave, flambda5100, alpha,
                coeff_FeII, sigma_FeII, shift_FeII,
                velocity_n1, sigma_n1_kms, flux_OIII5008_n1, flux_Hbeta_n1,
                velocity_n2, sigma_n2_kms, flux_OIII5008_n2, flux_Hbeta_n2,
                velocity_b1, sigma_b1_kms, flux_Hbeta_b1,
                velocity_b2, sigma_b2_kms, flux_Hbeta_b2,
                velocity_b3, sigma_b3_kms, flux_Hbeta_b3):
    
   
   

   
    
   FeII_model = model_FeII(wave, coeff_FeII, sigma_FeII, shift_FeII)
    
   narrow_model = model_narrow(wave,
                               velocity_n1, sigma_n1_kms, flux_OIII5008_n1, flux_Hbeta_n1,
                               velocity_n2, sigma_n2_kms, flux_OIII5008_n2, flux_Hbeta_n2)
    
   broad_model = model_broad(wave,
                             velocity_b1, sigma_b1_kms, flux_Hbeta_b1,
                             velocity_b2, sigma_b2_kms, flux_Hbeta_b2,
                             velocity_b3, sigma_b3_kms, flux_Hbeta_b3)
    
   powerlaw_model = model_powerlaw(wave, flambda5100, alpha)
    
   return powerlaw_model + FeII_model + narrow_model + broad_model
 
 
spec = Table.read('../cubes/{}_ESO_deep_quasar_r10_rest.fits'.format(name))
spec = np.array(spec)

ext = 2.512**extinction.fitzpatrick99(spec['wave'], Av)

spec['flux'] = spec['flux']*ext
spec['error'] = spec['error']*ext
spec['flux'] = spec['flux']*1e-20
spec['error'] = spec['error']*1e-20
 
spec = spec[(spec['wave'] > 4500) & (spec['wave'] < 5600)]
 
 
parameters = lmfit.Parameters()
parameters.add_many(('flambda5100',     1.2e-15,    True,   None,    None,   None),
                    ('alpha',             -2.20,    True,   None,    None,   None),
                    ('coeff_FeII',         0.10,    True,   None,    None,   None),
                    ('shift_FeII',      +0.0000,    False,   None,    None,   None),
                    ('sigma_FeII',        10.00,    True,    0.5,    None,   None),
                    ('velocity_n1',        0.00,    True, -500.0,   500.0,   None),
                    ('sigma_n1_kms',      250.0,    True, -500.0,   500.0,   None),
                    ('flux_OIII5008_n1', -14.30,    True,   None,    None,   None),
                    ('flux_Hbeta_n1',    -14.00,    True,   None,    None,   None),
                    ('velocity_n2',     -500.00,    True,   None,    None,   None),
                    ('sigma_n2_kms',     300.00,    True, -500.0,   500.0,   None),
                    ('flux_OIII5008_n2', -14.00,    True,   None,    None,   None),
                    ('flux_Hbeta_n2',    -15.00,    True,   None,    None,   None),
                    ('velocity_b1',        0.00,    True,   None,    None,   None),
                    ('sigma_b1_kms',    2000.00,    True,   None,    None,   None),
                    ('flux_Hbeta_b1',    -13.50,    True,   None,    None,   None),
                    ('velocity_b2',        0.00,    True,   None,    None,   None),
                    ('sigma_b2_kms',    3000.00,    True,   None,    None,   None),
                    ('flux_Hbeta_b2',    -13.00,    True,   None,    None,   None),
                    ('velocity_b3',        0.00,    True,   None,    None,   None),
                    ('sigma_b3_kms',    3000.00,    True,   None,    None,   None),
                    ('flux_Hbeta_b3',    -13.50,    True,   None,    None,   None))
 
powerlaw_model = lmfit.Model(model)
weights = 1/spec['error']
#index = np.where((spec['wave'] > 4700) & (spec['wave'] < 5000))
#weights[index] = 0.0
result = powerlaw_model.fit(spec['flux'], wave=spec['wave'],
                            params=parameters, weights=weights)
 
print(result.fit_report())
 
print(result.best_values)

print(result.message)
 
fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 10), sharex=True)
 
ax1.plot(spec['wave'], spec['flux'], drawstyle='steps-mid', color='black')
ax1.plot(spec['wave'], spec['error'], drawstyle='steps-mid', color='blue')
ax1.plot(spec['wave'], result.best_fit, color='red')
 
ax1.plot(spec['wave'], model_powerlaw(spec['wave'],
                                     result.best_values['flambda5100'],
                                     result.best_values['alpha']), color='grey')
                                      
ax1.plot(spec['wave'], model_powerlaw(spec['wave'],
                                     result.best_values['flambda5100'],
                                     result.best_values['alpha']) \
                     +model_FeII(spec['wave'],
                                 result.best_values['coeff_FeII'],
                                 result.best_values['sigma_FeII'],
                                 result.best_values['shift_FeII']),
                                 color='grey')                                 
ax1.plot(spec['wave'],
        model_narrow(spec['wave'],
                     result.best_values['velocity_n1'],
                     result.best_values['sigma_n1_kms'],
                     result.best_values['flux_OIII5008_n1'],
                     result.best_values['flux_Hbeta_n1'],
                     result.best_values['velocity_n2'],
                     result.best_values['sigma_n2_kms'],
                     result.best_values['flux_OIII5008_n2'],
                     result.best_values['flux_Hbeta_n2']), color='orange')
                     
ax1.plot(spec['wave'],
        model_broad(spec['wave'],
                    result.best_values['velocity_b1'],
                    result.best_values['sigma_b1_kms'],
                    result.best_values['flux_Hbeta_b1'],
                    result.best_values['velocity_b2'],
                    result.best_values['sigma_b2_kms'],
                    result.best_values['flux_Hbeta_b2'],
                    result.best_values['velocity_b3'],
                    result.best_values['sigma_b3_kms'],
                    result.best_values['flux_Hbeta_b3']),
                    color='red')

ax1.set_ylabel(r'$F_\lambda$')
ax1.minorticks_on()
 

continuum =  model_powerlaw(spec['wave'],
                            result.best_values['flambda5100'],
                            result.best_values['alpha']) \
                    + model_FeII(spec['wave'],
                                 result.best_values['coeff_FeII'],
                                 result.best_values['sigma_FeII'],
                                 result.best_values['shift_FeII'])
                                 
narrow = model_narrow(spec['wave'],
                     result.best_values['velocity_n1'],
                     result.best_values['sigma_n1_kms'],
                     result.best_values['flux_OIII5008_n1'],
                     result.best_values['flux_Hbeta_n1'],
                     result.best_values['velocity_n2'],
                     result.best_values['sigma_n2_kms'],
                     result.best_values['flux_OIII5008_n2'],
                     result.best_values['flux_Hbeta_n2'])
                     
broad = model_broad(spec['wave'],
                    result.best_values['velocity_b1'],
                    result.best_values['sigma_b1_kms'],
                    result.best_values['flux_Hbeta_b1'],
                    result.best_values['velocity_b2'],
                    result.best_values['sigma_b2_kms'],
                    result.best_values['flux_Hbeta_b2'],
                    result.best_values['velocity_b3'],
                    result.best_values['sigma_b3_kms'],
                    result.best_values['flux_Hbeta_b3'])
                    
broad = broad/np.max(broad)
indexMax = np.argmax(broad)
waveMax_broad = spec[indexMax]['wave']
spline_broad = UnivariateSpline(spec['wave'], broad - 0.5, s=0)

root1_broad, root2_broad = spline_broad.roots()
print('broad {}: {} - {}'.format(waveMax_broad, root1_broad, root1_broad))
FWHM_broad = (root2_broad - root1_broad)/waveMax_broad*c_kms
print('FWHM_broad = {:0.2f}'.format(FWHM_broad))

spec_broad = medfilt(spec['flux'] - continuum - narrow, 21)
spec_broad = spec_broad/np.max(spec_broad)

indexMax = np.argmax(spec_broad)
waveMax_spec_broad = spec[indexMax]['wave']

spline_spec_broad = UnivariateSpline(spec['wave'], spec_broad - 0.5, s=0)
#
root1_spec_broad, root2_spec_broad = spline_spec_broad.roots()
#
FWHM_spec_broad = (root2_spec_broad - root1_spec_broad)/waveMax_spec_broad*c_kms

ax2.axvline(root1_broad, color='red', linestyle=':')
ax2.axvline(root2_broad, color='red', linestyle=':')


ax2.axvline(root1_spec_broad, color='black', linestyle=':')
ax2.axvline(root2_spec_broad, color='black', linestyle=':')
ax2.axhline(0.5, color='black', linestyle=':')



ax2.plot(spec['wave'], spec_broad,
         drawstyle='steps-mid', color='black')
         
ax2.plot(spec['wave'], broad, color='red')
    
ax2.set_ylabel(r'$F_\lambda$')
ax2.set_xlabel(r'$\rm wavelength\ [\AA]$')
ax2.minorticks_on()
                 
 

 
 
#ax1.set_xlabel(r'$\rm wavelength\ [\AA]$')
fig.tight_layout()
plt.savefig('../plots/{}_Hbeta_fit.pdf'.format(name))


lumdist_Mpc = cosmography.Dl(redshift)
lumdist_cm = lumdist_Mpc*3.086e24
flux_5100 = result.best_values['flambda5100']
L_5100 = flux_5100*4*np.pi*lumdist_cm**2*(1 + redshift)*5100.0
L_bol = L_5100*9.26 # bolometric correction from Richards 2006
logLbol = np.log10(L_bol)

print('L5100 = {:0.2f}'.format(np.log10(L_5100)))
print('logLbol = {:0.2f}'.format(logLbol))

a = 0.910 # VP06
b = 0.50

logMbh = a + b*np.log10(L_5100/1e44) + 2.0*np.log10(FWHM_broad)
print('logMbh = {:0.2f}'.format(logMbh))

logLedd = np.log10(1.26e38*10.0**logMbh)

print('logLedd = {:0.2f}'.format(logLedd))

lambda_Edd = 10.0**(logLbol - logLedd)
print('lambda_Edd = {:0.3f}'.format(lambda_Edd))

