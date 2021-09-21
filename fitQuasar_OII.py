import os
import sys
import numpy as np
import mpdaf
import lmfit
from pydl.goddard.astro import airtovac, vactoair
from astropy.io import fits
from scipy.interpolate import interp1d
from astropy.table import Table
import redshift
from pathlib import Path
from matplotlib import pyplot as plt


spec = Table.read('../CUBES/TEX0206-048_zap_quasar_r10.fits')
spec = spec[(spec['wave'] > 7900) & (spec['wave'] < 8000)]
print(spec)

wave_OII3727_vac   = 3727.092
wave_OII3729_vac   = 3729.875

c_kms = 2.998e5

# specify where the quasar is and the size of the box
ra_qso = 32.378362
dec_qso = -4.6405499
zQSO = 1.1317

vMin = -1000
vMax = +1000
dv_fitting = 800.0

zMin_fitting = redshift.dz(zQSO, -dv_fitting)
zMax_fitting = redshift.dz(zQSO, dv_fitting)


# Define a function with the MUSE resolution
def getFWHM_MUSE(wave):
   
   return (5.866e-8*wave**2 - 9.187e-4*wave + 6.04)
   
# Define a function with the MUSE resolution
def getSigma_MUSE(wave):
   
   return (5.866e-8*wave**2 - 9.187e-4*wave + 6.04)/2.355

def getR_MUSE(wave):
   return wave/getFWHM(wave)
   
   
   
   
# Define the model fit function
def model(wave_vac, redshift, sigma_kms, fluxOII, rOII3729_3727, a, b):

   wave_OII3727_obs = wave_OII3727_vac*(1 + redshift)
   wave_OII3729_obs = wave_OII3729_vac*(1 + redshift)

   sigma_OII3727_A   = np.sqrt((sigma_kms/c_kms*wave_OII3727_obs)**2 + (getSigma_MUSE(wave_OII3727_obs))**2)
   sigma_OII3729_A   = np.sqrt((sigma_kms/c_kms*wave_OII3729_obs)**2 + (getSigma_MUSE(wave_OII3729_obs))**2)
   
   
   fluxOII3727 = fluxOII/(1 + rOII3729_3727)
   fluxOII3729 = fluxOII/(1 + 1.0/rOII3729_3727)
   
   peakOII3727 = fluxOII3727/np.sqrt(2*sigma_OII3727_A**2*np.pi)
   peakOII3729 = fluxOII3729/np.sqrt(2*sigma_OII3729_A**2*np.pi)
   
   
   OII3727_gaussian = peakOII3727*np.exp(-(wave_vac - wave_OII3727_obs)**2/2/sigma_OII3727_A**2)
   OII3729_gaussian = peakOII3729*np.exp(-(wave_vac - wave_OII3729_obs)**2/2/sigma_OII3729_A**2)
   
   
   return OII3727_gaussian + OII3729_gaussian + a*wave_vac + b
   
   
   
redshift_guess = zQSO
sigma_kms_guess = 150.0
flux_OII_guess = 300.0
rOII3729_3727_guess = 1.0



parameters = lmfit.Parameters()
parameters.add_many(('sigma_kms',     sigma_kms_guess,          True,  10.0,  500.0,  None),
                    ('redshift',      redshift_guess,           True,  zMin_fitting,  zMax_fitting,  None),
                    ('fluxOII',       flux_OII_guess,           True,  None,  None,  None),
                    ('rOII3729_3727', rOII3729_3727_guess,      True,  0.35,  1.5,  None),
                    ('a',             0.0,                      True,  None,  None,  None),
                    ('b',             44000,                    True,  None,  None,  None))
spec_model = lmfit.Model(model, missing='drop')    

result = spec_model.fit(spec['flux'], wave_vac=spec['wave'], params=parameters)

print('Success = {}'.format(result.success))
print(result.fit_report())



fig, ax = plt.subplots(1)
ax.plot(spec['wave'], spec['flux'], color='black', drawstyle='steps-mid')
#ax.plot(spec['wave'], spec['error'], color='blue', drawstyle='steps-mid')
ax.plot(spec['wave'], result.best_fit, color='red')

ax.minorticks_on()
fig.tight_layout()
plt.savefig('../plots/QSO_OII.pdf')



result.conf_interval()
print(result.ci_report())