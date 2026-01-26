import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rc
from astropy.io import ascii
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=4, visible=True)
rc('ytick.minor', size=4, visible=True)
rc('xtick', direction='in', labelsize=25, top='on')
rc('ytick', direction='in', labelsize=25, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)


# Load the data
path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')

plt.figure(figsize=(5, 5), dpi=300)
plt.plot(data_qso['logRL'], data_qso['sigma_mean_neb'], '.', ms=10, color='black')
plt.xlabel('logRL', size=15)
plt.ylabel('sigma_mean_neb [km/s]', size=15)
plt.savefig('../../MUSEQuBES+CUBS/plots/CUBS_MUSE_RadioLoudness_vs_NebularSigma.png', bbox_inches='tight')