import numpy as np
from regions import Regions
from astropy import units as u
from astropy.coordinates import SkyCoord


path_reg = '/Users/lzq/Dropbox/MUSEQuBES+CUBS/datacubes/Q0107-0235_ESO-DEEP_ZAP_JL_HST.reg'
rel_labels = Regions.read(path_reg, format='ds9')
x = rel_labels[j].center.ra.degree
y = rel_labels[j].center.dec.degree