import os
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.io import ascii
# ----------------------------
# Rest-frame Lyman wavelengths (Angstrom)
# ----------------------------
lyman_lines = {"Lyα": 1215.6701,
               "Lyβ": 1025.7223,
               "Lyγ": 972.5368,
               "Lyδ": 949.7431,
               "Lyε": 937.8035,
}
lyman_limit = 911.753
# ----------------------------
# QSO table
# ----------------------------
path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
data_qso = ascii.read(path_qso, format='fixed_width')


def ShowCUBSAbsorptionSpectra(cube_list):
    fig, axes = plt.subplots(3, 5, figsize=(20, 10), sharey=False, sharex=True)
    axes = axes.flatten()

    for i, cubename in enumerate(cube_list):
        ax = axes[i]

        # ----------------------------
        # Get QSO info
        # ----------------------------
        row = data_qso[data_qso['name'] == cubename]
        if len(row) == 0:
            ax.set_title(f'{cubename}\nnot in table')
            ax.axis('off')
            continue

        z_qso = row['redshift'][0]

        # ----------------------------
        # Load COS
        # ----------------------------
        path_CUBS_COS = f'../../MUSEQuBES+CUBS/CUBS_MUSE_COS_STIS/CUBS_COS/{cubename}_final_abscal.fits'
        has_cos = os.path.exists(path_CUBS_COS)

        # ----------------------------
        # Load STIS
        # ----------------------------
        path_CUBS_STIS = f'../../MUSEQuBES+CUBS/CUBS_MUSE_COS_STIS/CUBS_STIS_QZJ/{cubename}_all_plotabs.fits'
        has_stis = os.path.exists(path_CUBS_STIS)

        try:
            # Plot COS
            if has_cos:
                data_CUBS_COS = fits.open(path_CUBS_COS)[1].data
                wave_CUBS_COS = data_CUBS_COS['wave']
                wave_CUBS_COS = wave_CUBS_COS[::100] / (1 + z_qso)
                flux_CUBS_COS = data_CUBS_COS['flux']
                flux_CUBS_COS = flux_CUBS_COS[::100]
                ax.plot(wave_CUBS_COS, flux_CUBS_COS, color='k', lw=0.8, label='COS')

            # Plot STIS
            if has_stis:
                data_CUBS_STIS = fits.open(path_CUBS_STIS)[1].data
                wave_CUBS_STIS = data_CUBS_STIS['WAVE']
                wave_CUBS_STIS = wave_CUBS_STIS[::50] / (1 + z_qso)
                flux_CUBS_STIS = data_CUBS_STIS['FLUX'] / 1e-17
                flux_CUBS_STIS = flux_CUBS_STIS[::50]
                ax.plot(wave_CUBS_STIS, flux_CUBS_STIS, color='tab:blue', lw=0.8, alpha=0.8, label='STIS')

            # ----------------------------
            # Determine wavelength coverage
            # ----------------------------
            wave_all = []
            flux_all = []

            if has_cos:
                wave_all.append(wave_CUBS_COS)
                flux_all.append(flux_CUBS_COS)
            if has_stis:
                wave_all.append(wave_CUBS_STIS)
                flux_all.append(flux_CUBS_STIS)

            if len(wave_all) == 0:
                ax.set_title(f'{cubename}\nno COS/STIS')
                ax.axis('off')
                continue

            wave_all = np.hstack(wave_all)
            flux_all = np.hstack(flux_all)

            # Sort combined arrays for plotting limits / y-range
            sort = np.argsort(wave_all)
            wave_all = wave_all[sort]
            flux_all = flux_all[sort]

            # ----------------------------
            # Mark Lyman lines
            # ----------------------------
            y_top = np.nanpercentile(flux_all[np.isfinite(flux_all)], 95)

            for label, rest_wave in lyman_lines.items():
                obs_wave = rest_wave
                if wave_all.min() <= obs_wave <= wave_all.max():
                    ax.axvline(obs_wave, color='red', ls='--', alpha=0.5, lw=0.8)
                    ax.text(obs_wave, y_top, label, rotation=90,
                            color='red', ha='right', va='top', fontsize=7)

            # ----------------------------
            # Mark Lyman limit
            # ----------------------------
            obs_limit = lyman_limit
            if wave_all.min() <= obs_limit <= wave_all.max():
                ax.axvline(obs_limit, color='blue', ls='--', alpha=0.6, lw=0.8)
                ax.text(obs_limit, y_top, 'LL', rotation=90,
                        color='blue', ha='left', va='top', fontsize=7)

            # ----------------------------
            # Axis formatting
            # ----------------------------
            ax.set_xlim(800, 1300)

            finite_flux = flux_all[np.isfinite(flux_all)]
            if len(finite_flux) > 10:
                y1, y2 = np.percentile(finite_flux, [5, 95])
                pad = 0.15 * (y2 - y1) if y2 > y1 else 0.2
                ax.set_ylim(y1 - pad, y2 + pad)
            ax.axhline(1.0, color='gray', ls=':', lw=0.6)
            ax.set_title(f'{cubename}\nz={z_qso:.4f}', fontsize=10)

        except Exception as e:
            ax.set_title(f'{cubename}\nload failed')
            ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                    ha='center', va='center', fontsize=8)
            ax.axis('off')

    # Turn off unused panels if < 15 objects
    for j in range(len(cube_list), len(axes)):
        axes[j].axis('off')

    # Labels
    for ax in axes[10:]:
        ax.set_xlabel('Observed Wavelength (Å)')
    for ax in axes[::5]:
        ax.set_ylabel('Flux')

    # Legend only once
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.show()


def ShowMUSEAbsorptionSpectra(cube_list):
    fig, axes = plt.subplots(3, 6, figsize=(20, 10), sharey=False, sharex=True)
    axes = axes.flatten()

    for i, cubename in enumerate(cube_list):
        ax = axes[i]

        # ----------------------------
        # Get QSO info
        # ----------------------------
        row = data_qso[data_qso['name'] == cubename]
        if len(row) == 0:
            ax.set_title(f'{cubename}\nnot in table')
            ax.axis('off')
            continue

        z_qso = row['redshift'][0]

        # ----------------------------
        # Load COS
        # ----------------------------
        path_MUSE_COS = f'../../MUSEQuBES+CUBS/CUBS_MUSE_COS_STIS/COS_spectra_sowgat/spec_{cubename}_LA.fits'
        has_cos = os.path.exists(path_MUSE_COS)

        path_MUSE_COS_1 = f'../../MUSEQuBES+CUBS/CUBS_MUSE_COS_STIS/COS_spectra/{cubename}_COS.fits'
        has_cos_1 = os.path.exists(path_MUSE_COS_1)

        path_MUSE_COS_2 = f'../../MUSEQuBES+CUBS/CUBS_MUSE_COS_STIS/COS_spectra/{cubename}_COS_FUV_wavecal.fits'
        has_cos_2 = os.path.exists(path_MUSE_COS_2)

        path_MUSE_COS_3 = f'../../MUSEQuBES+CUBS/CUBS_MUSE_COS_STIS/COS_spectra/{cubename}_COS_wavecal.fits'
        has_cos_3 = os.path.exists(path_MUSE_COS_3)

        path_MUSE_COS_4 = f'../../MUSEQuBES+CUBS/CUBS_MUSE_COS_STIS/COS_spectra/{cubename}_FUV_wavecal.fits'
        has_cos_4 = os.path.exists(path_MUSE_COS_4)

        path_CUBS_STIS = f'../../MUSEQuBES+CUBS/CUBS_MUSE_COS_STIS/COS_spectra/{cubename}_all_plotabs.fits'
        has_stis = os.path.exists(path_CUBS_STIS)

        try:
            # Plot COS
            if has_cos:
                data_MUSE_COS = fits.open(path_MUSE_COS)[1].data
                wave_MUSE_COS = data_MUSE_COS['WAVELENGTH']
                wave_MUSE_COS = wave_MUSE_COS[::100] / (1 + z_qso)
                flux_MUSE_COS = data_MUSE_COS['FLUX']
                flux_MUSE_COS = flux_MUSE_COS[::100]
                ax.plot(wave_MUSE_COS, flux_MUSE_COS, color='k', lw=0.8, label='COS')

            if has_cos_1:
                data_MUSE_COS_1 = fits.open(path_MUSE_COS_1)[1].data
                wave_MUSE_COS_1 = data_MUSE_COS_1['WAVE']
                wave_MUSE_COS_1 = wave_MUSE_COS_1[::100] / (1 + z_qso)
                flux_MUSE_COS_1 = data_MUSE_COS_1['FLUX']
                flux_MUSE_COS_1 = flux_MUSE_COS_1[::100]
                ax.plot(wave_MUSE_COS_1, flux_MUSE_COS_1, color='k', lw=0.8, label='COS 1')

            if has_cos_2:
                data_MUSE_COS_2 = fits.open(path_MUSE_COS_2)[1].data
                wave_MUSE_COS_2 = data_MUSE_COS_2['WAVE']
                wave_MUSE_COS_2 = wave_MUSE_COS_2[::100] / (1 + z_qso)
                flux_MUSE_COS_2 = data_MUSE_COS_2['FLUX']
                flux_MUSE_COS_2 = flux_MUSE_COS_2[::100]
                ax.plot(wave_MUSE_COS_2, flux_MUSE_COS_2, color='k', lw=0.8, label='COS 2')

            if has_cos_3:
                data_MUSE_COS_3 = fits.open(path_MUSE_COS_3)[1].data
                wave_MUSE_COS_3 = data_MUSE_COS_3['WAVE']
                wave_MUSE_COS_3 = wave_MUSE_COS_3[::100] / (1 + z_qso)
                flux_MUSE_COS_3 = data_MUSE_COS_3['FLUX']
                flux_MUSE_COS_3 = flux_MUSE_COS_3[::100]
                ax.plot(wave_MUSE_COS_3, flux_MUSE_COS_3, color='k', lw=0.8, label='COS 3')

            if has_cos_4:
                data_MUSE_COS_4 = fits.open(path_MUSE_COS_4)[1].data
                wave_MUSE_COS_4 = data_MUSE_COS_4['WAVE']
                wave_MUSE_COS_4 = wave_MUSE_COS_4[::100] / (1 + z_qso)
                flux_MUSE_COS_4 = data_MUSE_COS_4['FLUX']
                flux_MUSE_COS_4 = flux_MUSE_COS_4[::100]
                ax.plot(wave_MUSE_COS_4, flux_MUSE_COS_4, color='k', lw=0.8, label='COS 4')

            # Plot STIS
            if has_stis:
                data_CUBS_STIS = fits.open(path_CUBS_STIS)[1].data
                wave_CUBS_STIS = data_CUBS_STIS['WAVE']
                wave_CUBS_STIS = wave_CUBS_STIS[::50] / (1 + z_qso)
                flux_CUBS_STIS = data_CUBS_STIS['FLUX'] / 1e-17
                flux_CUBS_STIS = flux_CUBS_STIS[::50]
                ax.plot(wave_CUBS_STIS, flux_CUBS_STIS, color='tab:blue', lw=0.8, alpha=0.8, label='STIS')

            # ----------------------------
            # Determine wavelength coverage
            # ----------------------------
            wave_all = []
            flux_all = []

            if has_cos:
                wave_all.append(wave_MUSE_COS)
                flux_all.append(flux_MUSE_COS)

            if has_cos_1:
                wave_all.append(wave_MUSE_COS_1)
                flux_all.append(flux_MUSE_COS_1)

            if has_cos_2:
                wave_all.append(wave_MUSE_COS_2)
                flux_all.append(flux_MUSE_COS_2)

            if has_cos_3:
                wave_all.append(wave_MUSE_COS_3)
                flux_all.append(flux_MUSE_COS_3)

            if has_cos_4:
                wave_all.append(wave_MUSE_COS_4)
                flux_all.append(flux_MUSE_COS_4)

            if has_stis:
                wave_all.append(wave_MUSE_STIS)
                flux_all.append(flux_MUSE_STIS)

            if len(wave_all) == 0:
                ax.set_title(f'{cubename}\nno COS/STIS')
                ax.axis('off')
                continue

            wave_all = np.hstack(wave_all)
            flux_all = np.hstack(flux_all)

            # Sort combined arrays for plotting limits / y-range
            sort = np.argsort(wave_all)
            wave_all = wave_all[sort]
            flux_all = flux_all[sort]

            # ----------------------------
            # Mark Lyman lines
            # ----------------------------
            y_top = np.nanpercentile(flux_all[np.isfinite(flux_all)], 95)

            for label, rest_wave in lyman_lines.items():
                obs_wave = rest_wave
                if wave_all.min() <= obs_wave <= wave_all.max():
                    ax.axvline(obs_wave, color='red', ls='--', alpha=0.5, lw=0.8)
                    ax.text(obs_wave, y_top, label, rotation=90,
                            color='red', ha='right', va='top', fontsize=7)

            # ----------------------------
            # Mark Lyman limit
            # ----------------------------
            obs_limit = lyman_limit
            if wave_all.min() <= obs_limit <= wave_all.max():
                ax.axvline(obs_limit, color='blue', ls='--', alpha=0.6, lw=0.8)
                ax.text(obs_limit, y_top, 'LL', rotation=90,
                        color='blue', ha='left', va='top', fontsize=7)

            # ----------------------------
            # Axis formatting
            # ----------------------------
            ax.set_xlim(500, 1300)

            finite_flux = flux_all[np.isfinite(flux_all)]
            if len(finite_flux) > 10:
                y1, y2 = np.percentile(finite_flux, [5, 95])
                pad = 0.15 * (y2 - y1) if y2 > y1 else 0.2
                ax.set_ylim(y1 - pad, y2 + pad)
            ax.axhline(1.0, color='gray', ls=':', lw=0.6)
            ax.set_title(f'{cubename}\nz={z_qso:.4f}', fontsize=10)

        except Exception as e:
            ax.set_title(f'{cubename}\nload failed')
            ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                    ha='center', va='center', fontsize=8)
            ax.axis('off')

    # Turn off unused panels if < 15 objects
    for j in range(len(cube_list), len(axes)):
        axes[j].axis('off')

    # Labels
    for ax in axes[10:]:
        ax.set_xlabel('Observed Wavelength (Å)')
    for ax in axes[::5]:
        ax.set_ylabel('Flux')

    # Legend only once
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    plt.show()

# Choose 15 objects (example: first 15)
CUBS_list = ['J0110-1648', 'J0454-6116', 'J2135-5316', 'J0119-2010', 'HE0246-4101',
             'J0028-3305', 'HE0419-5657', 'PKS2242-498', 'PKS0355-483', 'HE0112-4145',
             'HE2305-5315', 'HE0331-4112', 'J0111-0316',  'J0154-0712', 'HE2336-5540']

# ShowCUBSAbsorptionSpectra(CUBS_list)

MUSE_list = ['HE0435-5304', 'HE0153-4520', 'HE0226-4110', 'PKS0405-123', 'HE0238-1904',
             '3C57', 'PKS0552-640', 'PB6291', 'Q0107-0235', 'HE0439-5254', 'HE1003+0149',
             'TEX0206-048', 'Q1354+048', 'LBQS1435-0134', 'PG1522+101', 'PKS0232-04']
ShowMUSEAbsorptionSpectra(MUSE_list)
