import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table


def sigma_clip_1d(v, nsigma=3, max_iter=10):
    v = np.asarray(v, dtype=float)
    mask = np.isfinite(v)
    for _ in range(max_iter):
        vv = v[mask]
        if len(vv) < 3:
            break
        mean = np.mean(vv)
        std = np.std(vv, ddof=0)
        if std == 0 or not np.isfinite(std):
            break
        new_mask = np.isfinite(v) & (np.abs(v - mean) <= nsigma * std)
        if np.all(new_mask == mask):
            break
        mask = new_mask
    return mask, np.mean(v[mask]), np.std(v[mask], ddof=0)


cubename_muse = [
    'HE0435-5304', 'HE0226-4110', 'PKS0405-123',
    'HE0238-1904', '3C57', 'PKS0552-640', 'PB6291',
    'Q0107-0235', 'HE0439-5254', 'HE1003+0149', 'TEX0206-048',
    'Q1354+048', 'LBQS1435-0134', 'PG1522+101', 'PKS0232-04']


for cubename in cubename_muse:
    # Input file
    path_gal = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)

    # Output files
    path_out_fits = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia_3sig_members.fits'.format(cubename)
    path_out_txt = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia_3sig_members.txt'.format(cubename)

    # Load galaxy information
    data_gal = fits.open(path_gal)[1].data
    v_gal = data_gal['v']

    # Add the quasar as v = 0 for the clipping calculation
    v_all = np.concatenate(([0.0], v_gal))

    # Run iterative 3-sigma clipping
    member_mask_all, v_mean, v_sigma = sigma_clip_1d(v_all)

    # Remove the first element, which corresponds to the artificial quasar
    qso_is_member = member_mask_all[0]
    member_mask_gal = member_mask_all[1:]

    # Make output table containing only selected galaxy members
    tab_gal = Table(data_gal)
    tab_mem = tab_gal[member_mask_gal]

    # Optional: add useful columns to the output table
    tab_mem['member_3sigma'] = np.ones(len(tab_mem), dtype=bool)
    tab_mem['group_v_mean'] = np.full(len(tab_mem), v_mean)
    tab_mem['group_v_sigma'] = np.full(len(tab_mem), v_sigma)

    # Save as FITS
    # tab_mem.write(path_out_fits, overwrite=True)

    # Save as TXT
    # tab_mem.write(path_out_txt, format='ascii.fixed_width', overwrite=True)

    N = len(v_gal)
    N_galaxy_members = np.sum(member_mask_gal)
    N_galaxy_members_incl_qso = N_galaxy_members + 1

    print('\n', cubename)
    print('Input file:', path_gal)
    print('QSO included in clipping?', qso_is_member)
    print('N total =', N)
    print('N galaxy members =', N_galaxy_members)
    print('N including QSO =', N_galaxy_members_incl_qso)
    print('mean velocity = {:.2f} km/s'.format(v_mean))
    print('sigma velocity = {:.2f} km/s'.format(v_sigma))
    print('Saved:', path_out_fits)
    print('Saved:', path_out_txt)

    if N == N_galaxy_members_incl_qso:

        bins = np.linspace(-3000, 3000, 31)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), )
        ax[0].hist(v_all, bins=bins, color='gray', alpha=0.5, label='All')
        ax[0].hist(v_all[member_mask_all], bins=bins, color='blue', alpha=0.5, label='3-sigma members')
        ax[0].axvline(v_mean, color='red', linestyle='--', label='Mean')

        ax[1].plot(tab_gal['ra'], tab_gal['dec'], 'o', color='gray', label='All galaxies')
        ax[1].plot(tab_mem['ra'], tab_mem['dec'], 'o', color='blue', label='3-sigma members')
        # ax[1].set_xlim(0, 150)
        # ax[1].set_ylim(0, 150)
        ax[0].set_title(cubename)
        plt.show()

