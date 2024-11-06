import astroplan
import numpy as np
import astropy.io.fits as fits
from astroplan import Observer
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u

# Select alignment stars
path_LS = fits.open('../../MaskDesign/HE0226_mask/HE0226_LS.fits')
data_LS = path_LS[1].data

ID = data_LS['ls_id']
ra_Gaia, dec_Gaia = data_LS['ra'], data_LS['dec']
c_object = SkyCoord(ra_Gaia * u.deg, dec_Gaia * u.deg, frame='fk5')
coords_string = np.asarray(c_object.to_string(style='hmsdms', sep=':', precision=2))
ra_string = np.asarray(c_object.ra.to_string(unit=u.hour, sep=':', precision=2, pad=True))
dec_string = np.asarray(c_object.dec.to_string(unit=u.degree, sep=':', precision=2, pad=True))

# Select alignment star
# Legacy survey
flux_g = data_LS['flux_g'] / data_LS['mw_transmission_g']
flux_r = data_LS['flux_r'] / data_LS['mw_transmission_r']
flux_i = data_LS['flux_i'] / data_LS['mw_transmission_i']
flux_z = data_LS['flux_z'] / data_LS['mw_transmission_z']

mag_g = 22.5 - 2.5 * np.log10(flux_g)
mag_r = 22.5 - 2.5 * np.log10(flux_r)
mag_i = 22.5 - 2.5 * np.log10(flux_i)
mag_z = 22.5 - 2.5 * np.log10(flux_z)

#
type = data_LS['type']
isstar = np.zeros_like(mag_r)
isstar = np.where(~((type == 'PSF') * (mag_r < 21.5)), isstar, 1)

alignments = np.arange(len(isstar))
select_star = np.where((isstar == 1) * (mag_r >= 17.0) * (mag_r < 21)
                       * (mag_g - mag_r > 0.0) * (mag_g - mag_r < 1.0))
alignments = alignments[select_star]

# * .dat file
path_star_dat = '../../MaskDesign/{}_mask/{}_*.dat'.format('HE0226', 'HE0226')
ID_star = list(map(''.join, zip(np.full_like(alignments, '*', dtype=str), np.asarray(ID[alignments], dtype=str))))
star_cat = np.array([ID_star, coords_string[alignments], np.round(mag_r[alignments], decimals=2)]).T
np.savetxt(path_star_dat, star_cat, fmt="%s")

# * .reg file
path_star_reg = '../../MaskDesign/{}_mask/{}_*.reg'.format('HE0226', 'HE0226')
star_reg = np.array(['# Region file format: DS9 version 4.1',
                     'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 '
                     'highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
                     'fk5'])
star_reg_2 = list(map(''.join, zip(np.full_like(alignments, 'circle(', dtype='<U15'),
                                   ra_string[alignments], np.full_like(alignments, ', ', dtype=str),
                                   dec_string[alignments], np.full_like(alignments, ', ', dtype=str),
                                   np.full_like(alignments, '5") # text={', dtype='<U15'), ID_star,
                                   np.full_like(alignments, '}', dtype='<U20'))))
star_reg = np.hstack((star_reg, star_reg_2))
np.savetxt(path_star_reg, star_reg, fmt="%s")


# Select Flux standard star
path = fits.open('../../MaskDesign/HE0226_mask/HE0226_gaia+DES+LS.fits')
data = path[1].data

# Give ID
ID = data['ls_id']
ra_Gaia, dec_Gaia = data['ra_1'], data['dec_1']
c_object = SkyCoord(ra_Gaia * u.deg, dec_Gaia * u.deg, frame='fk5')
coords_string = np.asarray(c_object.to_string(style='hmsdms', sep=':', precision=2))
ra_string = np.asarray(c_object.ra.to_string(unit=u.hour, sep=':', precision=2, pad=True))
dec_string = np.asarray(c_object.dec.to_string(unit=u.degree, sep=':', precision=2, pad=True))

# Select alignment star
# Legacy survey
flux_g = data['flux_g'] / data['mw_transmission_g']
flux_r = data['flux_r'] / data['mw_transmission_r']
flux_i = data['flux_i'] / data['mw_transmission_i']
flux_z = data['flux_z'] / data['mw_transmission_z']

mag_g = 22.5 - 2.5 * np.log10(flux_g)
mag_r = 22.5 - 2.5 * np.log10(flux_r)
mag_i = 22.5 - 2.5 * np.log10(flux_i)
mag_z = 22.5 - 2.5 * np.log10(flux_z)

# Select Flux standard stars
temp, source_id  = data['teff_gspphot'], data['source_id']
isFG = (temp > 5000) * (temp < 7500)
# print(source_id[isFG])
# raise ValueError('Stop')

# Select
# output_isFG = np.column_stack((ra[isFG], dec[isFG], source_id[isFG]))
# path_isFG = '../../MaskDesign/HE0226_mask/HE0226_isFG.dat'
# np.savetxt(path_isFG, output_isFG, fmt="%s")

# Select flux standard star
# source_id_obj = [4951192261076095104, 4951195593970715776, 4951196590403127424,
#                  4951196865281033472, 4951196860984717056, 4951196895344466048,
#                  4951193085709815424, 4951195933271394432, 4951196379947993600]
# DESI_id = [39626876410401459, 39626876410400348, 39626876410397995,
#            39626876410397305, 39626876410397421, 39626876410397585,
#            39626876410401973, 39626876410399691, 4951196379947993600]
# path_stand_reg = '../../MaskDesign/{}_mask/{}_stand.reg'.format('HE0226', 'HE0226')
# stand_reg = np.array(['# Region file format: DS9 version 4.1',
#                      'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 '
#                      'highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
#                      'fk5'])
# stand_reg_2 = list(map(''.join, zip(np.full_like(ID[select_stand], 'circle(', dtype='<U15'),
#                                    ra_string[select_stand], np.full_like(ID[select_stand], ', ', dtype=str),
#                                    dec_string[select_stand], np.full_like(ID[select_stand], ', ', dtype=str),
#                                    np.full_like(ID[select_stand], '5") # text={', dtype='<U15'),
#                                     np.asarray(source_id_obj, dtype=str),
#                                    np.full_like(ID[select_stand], '}', dtype='<U20'))))
# stand_reg = np.hstack((stand_reg, stand_reg_2))
# np.savetxt(path_stand_reg, stand_reg, fmt="%s")


# Add the slit
source_id_obj = [4951192261076095104, 4951195593970715776, 4951196590403127424,
                 4951196865281033472, 4951196860984717056, 4951196895344466048,
                 4951193085709815424, 4951196379947993600]
DESI_id = [39626876410401459, 39626876410400348, 39626876410397995,
           39626876410397305, 39626876410397421, 39626876410397585,
           39626876410401973, 4951196379947993600]
select_stand = np.in1d(source_id, source_id_obj)
path_l1 = '../../MaskDesign/{}_mask/{}_@.dat'.format('HE0226', 'HE0226l1')
l1 = '{} {} {} {} {} {} {} {} {}'.format('@1', '02:28:15.7746', '-40:57:12.314', '16.5', '0' , '0.75', '2', '15', '15')
ID_stand = list(map(''.join, zip(np.full_like(ID[select_stand], '@', dtype=str), np.asarray(ID[select_stand], dtype=str))))
stand_cat = list(map(''.join, zip(ID_stand, np.full_like(ID_stand, ' ', dtype=str),
                                  coords_string[select_stand], np.full_like(ID_stand, ' ', dtype=str),
                                  np.asarray(np.round(mag_r[select_stand], decimals=2), dtype=str),
                                  np.full_like(ID_stand, ' ', dtype=str), np.full_like(ID_stand, '0', dtype=str),
                                  np.full_like(ID_stand, ' ', dtype=str), np.asarray(np.full_like(ID_stand, 0.75), dtype=str),
                                  np.full_like(ID_stand, ' ', dtype=str), np.full_like(ID_stand, '2', dtype=str),
                                  np.full_like(ID_stand, ' ', dtype=str), np.full_like(ID_stand, "7 ", dtype=str),
                                  np.full_like(ID_stand, ' ', dtype=str), np.full_like(ID_stand, '7', dtype=str))))
cat = np.hstack((l1, stand_cat))
# np.savetxt(path_l1, cat, fmt="%s")

#
source_id_obj = [4951192261076095104, 4951195593970715776, 4951196590403127424,
                 4951196865281033472, 4951196860984717056, 4951196895344466048,
                 4951193085709815424, 4951195933271394432, 4951196379947993600]
DESI_id = [39626876410401459, 39626876410400348, 39626876410397995,
           39626876410397305, 39626876410397421, 39626876410397585,
           39626876410401973, 39626876410399691, 4951196379947993600]
select_stand = np.in1d(source_id, source_id_obj)
path_l2 = '../../MaskDesign/{}_mask/{}_@.dat'.format('HE0226', 'HE0226l2')
l2 = '{} {} {} {} {} {} {} {} {}'.format('@2', '02:28:15.4911', '-40:57:11.363', '16.5', '0' , '0.75', '2', '15', '15')
ID_stand = list(map(''.join, zip(np.full_like(ID[select_stand], '@', dtype=str), np.asarray(ID[select_stand], dtype=str))))
stand_cat = list(map(''.join, zip(ID_stand, np.full_like(ID_stand, ' ', dtype=str),
                                  coords_string[select_stand], np.full_like(ID_stand, ' ', dtype=str),
                                  np.asarray(np.round(mag_r[select_stand], decimals=2), dtype=str),
                                  np.full_like(ID_stand, ' ', dtype=str), np.full_like(ID_stand, '0', dtype=str),
                                  np.full_like(ID_stand, ' ', dtype=str), np.asarray(np.full_like(ID_stand, 0.75), dtype=str),
                                  np.full_like(ID_stand, ' ', dtype=str), np.full_like(ID_stand, '2', dtype=str),
                                  np.full_like(ID_stand, ' ', dtype=str), np.full_like(ID_stand, "7 ", dtype=str),
                                  np.full_like(ID_stand, ' ', dtype=str), np.full_like(ID_stand, '7', dtype=str))))
cat = np.hstack((l2, stand_cat))
np.savetxt(path_l2, cat, fmt="%s")

# # * .reg file
# path_star_reg = '../../MaskDesign/{}_mask/{}_@.reg'.format('HE0226', 'HE0226')
# star_reg = np.array(['# Region file format: DS9 version 4.1',
#                      'global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 '
#                      'highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1',
#                      'fk5'])
# star_reg_2 = list(map(''.join, zip(np.full_like(alignments, 'circle(', dtype='<U15'),
#                                    ra_string[alignments], np.full_like(alignments, ', ', dtype=str),
#                                    dec_string[alignments], np.full_like(alignments, ', ', dtype=str),
#                                    np.full_like(alignments, '5") # text={', dtype='<U15'), ID_star,
#                                    np.full_like(alignments, '}', dtype='<U20'))))
# star_reg = np.hstack((star_reg, star_reg_2))
# np.savetxt(path_star_reg, star_reg, fmt="%s")

# output_LS = np.column_stack((ra[mask], dec[mask], source_id_obj))
# path_LS = '../../MaskDesign/HE0226_mask/HE0226_LS.dat'
# np.savetxt(path_LS, output_LS, fmt="%s")


# Select the target



# # Define OBS
# las = Observer.at_site("Las Campanas Observatory", timezone="UTC")
#
#
# # Calcualte Parallactic angle
# time
# para = las.parallactic_angle(time, target)