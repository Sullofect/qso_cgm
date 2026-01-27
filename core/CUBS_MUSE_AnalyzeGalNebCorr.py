import os
import h5py
import aplpy
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import rc
from astropy.wcs import WCS
from astropy.io import ascii
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from CUBS_MUSE_MakeV50W80 import APLpyStyle
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=4, visible=True)
rc('ytick.minor', size=4, visible=True)
rc('xtick', direction='in', labelsize=25, top='on')
rc('ytick', direction='in', labelsize=25, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# def chol_logdet(M):
#     L = np.linalg.cholesky(M)
#     return L, 2.0 * np.sum(np.log(np.diag(L)))  # log|M|
#
# def bhattacharyya_coefficient(spaxel_mus, spaxel_Sigmas, mu_g, Sigma_g, jitter=1e-12):
#     N = spaxel_mus.shape[0]
#     if spaxel_Sigmas.shape != (N, 3, 3):
#         raise ValueError("spaxel_Sigmas is 3D but not (N,3,3).")
#
#     d = spaxel_mus - mu_g[None, :]  # (N,3)
#     out = np.full(N, np.nan)
#     Sg = Sigma_g.copy()
#     # Sg.flat[::4] += jitter
#     _, logdet_Sg = chol_logdet(Sg)
#     for i in range(N):
#         if np.isnan(spaxel_mus[i, 2]):
#             continue
#         Ss = spaxel_Sigmas[i].copy()
#         # Ss.flat[::4] += jitter
#         Sm = 0.5 * (Sg + Ss)
#         # Sm.flat[::4] += jitter
#         _, logdet_Ss = chol_logdet(Ss)
#         Lm, logdet_Sm = chol_logdet(Sm)
#         y = np.linalg.solve(Lm, d[i])
#         m2 = float(y @ y)
#         log_bc = 0.25 * (logdet_Sg + logdet_Ss) - 0.5 * logdet_Sm - 0.125 * m2
#         out[i] = np.exp(log_bc)
#     return out


def bhattacharyya_coefficient(mus_neb=None, sigma_x_neb=1.5, sigma_y_neb=1.5, sigma_v_neb=None,
                              mus_gal=None, sigma_x_gal=None, sigma_y_gal=None, sigma_v_gal=20.0):
    N, M = mus_neb.shape[0], mus_gal.shape[0]

    # mask invalid spaxels
    good = np.isfinite(mus_neb[:, 2])
    out = np.full((N, M), np.nan, dtype=float)

    mus = mus_neb[good]       # (Ng,3)
    svs = sigma_v_neb[good]      # (Ng,)

    # deltas (Ng, G)
    dx = mus[:, 0:1] - mus_gal[None, :, 0]
    dy = mus[:, 1:2] - mus_gal[None, :, 1]
    dv = mus[:, 2:3] - mus_gal[None, :, 2]

    # x
    varsum_x = sigma_x_neb ** 2 + sigma_x_gal ** 2
    pref_x = np.sqrt(2.0 * sigma_x_neb * sigma_x_gal / varsum_x)
    expo_x = -(dx * dx) / (4.0 * varsum_x)

    # y
    varsum_y = sigma_y_neb ** 2 + sigma_y_gal ** 2
    pref_y = np.sqrt(2.0 * sigma_y_neb * sigma_y_gal / varsum_y)
    expo_y = -(dy * dy) / (4.0 * varsum_y)

    # v
    varsum_v = svs ** 2 + sigma_v_gal ** 2            # (Ng,)
    pref_v = np.sqrt(2.0 * svs * sigma_v_gal / varsum_v)  # (Ng,)
    expo_v = -(dv * dv) / (4.0 * varsum_v[:, None])

    # Result
    bc_good = (pref_x * pref_y) * pref_v[:, None] * np.exp(expo_x + expo_y + expo_v)  # (Ng,G)
    out[good, :] = bc_good
    return out

class CalculateGalNebCorr:
    def __init__(self, L=None, S=None, A=None, Ntrial=5000):
        path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
        self.data_qso = ascii.read(path_qso, format='fixed_width')
        self._qso_index = {str(n): i for i, n in enumerate(self.data_qso['name'])}
        self.L, self.S , self.A = L, S, A
        self.allType = np.vstack((L, S, A))
        self.Ntrial = Ntrial
        self.cubename_all = np.hstack((L[:, 0], S[:, 0], A[:, 0]))

    def ComputeCorr(self, cubename=None, nums_seg_OII=None, select_seg_OII=False, nums_seg_OIII=None,
                    select_seg_OIII=False):
        # QSO information
        i = self._qso_index.get(cubename, None)
        ra_qso, dec_qso, z_qso = self.data_qso['ra_GAIA'][i], self.data_qso['dec_GAIA'][i], self.data_qso['redshift'][i]

        # V50, S80
        path_v50_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_V50_plot.fits'.format(cubename)
        path_s80_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_S80_plot.fits'.format(cubename)
        v50 = fits.open(path_v50_plot)[1].data
        s80 = fits.open(path_s80_plot)[1].data
        hdr_v50 = fits.open(path_v50_plot)[1].header
        w = WCS(hdr_v50, naxis=2)

        # Load data
        UseSeg = (1.5, 'gauss', 1.5, 'gauss')
        line_OII, line_OIII = 'OII', 'OIII'

        # OII SBs
        if cubename == 'TEX0206-048':
            str_zap = '_zapped'
        else:
            str_zap = ''

        # Load the segmentation map
        path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line_OII, *UseSeg)
        path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
            format(cubename, str_zap, line_OIII, *UseSeg)
        if cubename == 'PKS0552-640':
            path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}_plot.fits'. \
                format(cubename, str_zap, line_OIII, *UseSeg)
        elif cubename == 'HE0226-4110':
            path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}_plot.fits'. \
                format(cubename, str_zap, line_OII, *UseSeg)

        seg_OII = fits.open(path_3Dseg_OII)[1].data
        if select_seg_OII:
            nums_seg_OII = np.setdiff1d(np.arange(1, np.max(seg_OII) + 1), nums_seg_OII)
        seg_OII_mask = np.where(~np.isin(seg_OII, nums_seg_OII), seg_OII, -1)

        # Fix v50 and s80 according to [O II] seg
        v50 = np.where(seg_OII_mask != -1, v50, np.nan)
        s80 = np.where(seg_OII_mask != -1, s80, np.nan)

        # Fix v50 and s80 according to [O III] seg
        if os.path.exists(path_3Dseg_OIII):
            seg_OIII_3D, seg_OIII = fits.open(path_3Dseg_OIII)[0].data, fits.open(path_3Dseg_OIII)[1].data
            if select_seg_OIII:
                nums_seg_OIII = np.setdiff1d(np.arange(1, np.max(seg_OIII) + 1), nums_seg_OIII)
            seg_OIII_mask = np.where(~np.isin(seg_OIII, nums_seg_OIII), seg_OIII, -1)

            # Fix v50 and s80 accordingly to [O II]
            v50 = np.where(seg_OIII_mask != -1, v50, np.nan)
            s80 = np.where(seg_OIII_mask != -1, s80, np.nan)

        # Load galaxy information
        path_gal = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
        data_gal = fits.open(path_gal)[1].data
        v_gal, ra_gal, dec_gal, type = data_gal['v'], data_gal['ra_HST'], data_gal['dec_HST'], data_gal['type']
        c_gal = w.world_to_pixel(SkyCoord(ra_gal, dec_gal, unit='deg', frame='icrs'))

        # Compute the physical scale at the redshift of the quasar
        d_A_kpc = cosmo.angular_diameter_distance(z_qso).value * 1e3
        sigma_physical = (10 / d_A_kpc) * 206265 / 0.2  # Convert 10 kpc to pixel scale
        print('physical scale (pixel) = ', sigma_physical)

        # Start
        x, y = np.meshgrid(np.arange(v50.shape[1]), np.arange(v50.shape[0]))
        x, y = x.ravel(), y.ravel()
        v50_flat = v50.ravel()
        s80_flat = s80.ravel()

        # Nebular spaxels
        input_neb = np.column_stack([x, y, v50_flat])
        sigma_v_neb = s80_flat

        # Galaxys
        gal_mus_all = np.column_stack([c_gal[0], c_gal[1], v_gal]).astype(float)  # (G,3)
        sigma_x_gal = sigma_physical  # pixel
        sigma_y_gal = sigma_physical  # pixel
        sigma_v_gal = 20.0  # km/s

        # Calculate overlapping
        # batch = 256
        # KAF_flat = np.zeros(input_neb.shape[0], dtype=float)  # (N,)
        # for j0 in range(0, gal_mus_all.shape[0], batch):
        #     j1 = min(j0 + batch, gal_mus_all.shape[0])
        #     gal_mus_batch = gal_mus_all[j0:j1]  # (B,3)
        overlap = bhattacharyya_coefficient(mus_neb=input_neb, sigma_x_neb=1.5,
                                            sigma_y_neb=1.5, sigma_v_neb=sigma_v_neb,
                                            mus_gal=gal_mus_all, sigma_x_gal=sigma_x_gal,
                                            sigma_y_gal=sigma_y_gal, sigma_v_gal=sigma_v_gal)
        KAF_flat = np.sum(overlap, axis=1)
        KAF = KAF_flat.reshape(v50.shape)
        CKAF = np.nansum(KAF)

        # Save KAF as fits
        path_KAF = '../../MUSEQuBES+CUBS/KAF/{}_KAF.fits'.format(cubename)
        hdul_KAF = fits.ImageHDU(KAF, header=hdr_v50)
        hdul_KAF.writeto(path_KAF, overwrite=True)

        # Plot the score matrix
        fig = plt.figure(figsize=(8, 8), dpi=300)
        gc = aplpy.FITSFigure(path_KAF, figure=fig, hdu=1)
        gc.show_colorscale(vmin=1e-3, vmax=0.5, cmap='viridis', stretch='log')
        APLpyStyle(gc, type='GasMap', cubename=cubename, ra_qso=ra_qso, dec_qso=dec_qso, z_qso=z_qso, addName=True)

        # Set colorbar
        if cubename == "J2135-5316" or cubename == "Q0107-0235" or cubename == "PKS2242-498" or \
                cubename == "PG1522+101" or cubename == "PKS0232-04":
            gc.colorbar.set_ticks([1e-2, 1e-1, 0.25])
            gc.colorbar._colorbar.set_ticklabels([1e-2, 1e-1, 0.25])
            tick_labels = gc.colorbar._colorbar.ax.get_xticklabels()
            tick_labels[0].set_ha('right')
            gc.colorbar.set_location('bottom')
            gc.colorbar.set_axis_label_text(r'KAF')
            gc.colorbar._colorbar.minorticks_off()
        plt.savefig('../../MUSEQuBES+CUBS/plots/{}_KAF.png'.format(cubename), bbox_inches='tight')
        plt.close()

        return CKAF

    def ComputeCorrControl(self, cubename=None, nums_seg_OII=None, select_seg_OII=False,
                           nums_seg_OIII=None, select_seg_OIII=False, Ntrial=1000):
        # QSO information
        i = self._qso_index.get(cubename, None)
        ra_qso, dec_qso, z_qso = self.data_qso['ra_GAIA'][i], self.data_qso['dec_GAIA'][i], self.data_qso['redshift'][i]

        # V50, S80
        path_v50_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_V50_plot.fits'.format(cubename)
        path_s80_plot = '../../MUSEQuBES+CUBS/fit_kin/{}_S80_plot.fits'.format(cubename)
        v50 = fits.open(path_v50_plot)[1].data
        s80 = fits.open(path_s80_plot)[1].data
        hdr_v50 = fits.open(path_v50_plot)[1].header
        w = WCS(hdr_v50, naxis=2)

        # Load data
        UseSeg = (1.5, 'gauss', 1.5, 'gauss')
        line_OII, line_OIII = 'OII', 'OIII'

        # OII SBs
        if cubename == 'TEX0206-048':
            str_zap = '_zapped'
        else:
            str_zap = ''

        # Load the segmentation map
        path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'.\
            format(cubename, str_zap, line_OII, *UseSeg)
        path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
            format(cubename, str_zap, line_OIII, *UseSeg)
        if cubename == 'PKS0552-640':
            path_3Dseg_OIII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}_plot.fits'. \
                format(cubename, str_zap, line_OIII, *UseSeg)
        elif cubename == 'HE0226-4110':
            path_3Dseg_OII = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}_plot.fits'. \
                format(cubename, str_zap, line_OII, *UseSeg)

        seg_OII = fits.open(path_3Dseg_OII)[1].data
        if select_seg_OII:
            nums_seg_OII = np.setdiff1d(np.arange(1, np.max(seg_OII) + 1), nums_seg_OII)
        seg_OII_mask = np.where(~np.isin(seg_OII, nums_seg_OII), seg_OII, -1)

        # Fix v50 and s80 according to [O II] seg
        v50 = np.where(seg_OII_mask != -1, v50, np.nan)
        s80 = np.where(seg_OII_mask != -1, s80, np.nan)

        # Fix v50 and s80 according to [O III] seg
        if os.path.exists(path_3Dseg_OIII):
            seg_OIII_3D, seg_OIII = fits.open(path_3Dseg_OIII)[0].data, fits.open(path_3Dseg_OIII)[1].data
            if select_seg_OIII:
                nums_seg_OIII = np.setdiff1d(np.arange(1, np.max(seg_OIII) + 1), nums_seg_OIII)
            seg_OIII_mask = np.where(~np.isin(seg_OIII, nums_seg_OIII), seg_OIII, -1)

            # Fix v50 and s80 accordingly to [O II]
            v50 = np.where(seg_OIII_mask != -1, v50, np.nan)
            s80 = np.where(seg_OIII_mask != -1, s80, np.nan)

        # Compute the physical scale at the redshift of the quasar
        d_A_kpc = cosmo.angular_diameter_distance(z_qso).value * 1e3
        sigma_physical = (10 / d_A_kpc) * 206265 / 0.2  # Convert 10 kpc to pixel scale
        print('physical scale (pixel) = ', sigma_physical)

        # Monte Carlo samples from fitted 4D Gaussian
        path_gal = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
        Ngal = len(fits.open(path_gal)[1].data)

        # Start the process
        mu_mle, Sigma_mle = self.Derive3DDist_xyv()
        rng = np.random.default_rng(0)
        Nsamp = Ntrial * Ngal
        Xs = rng.multivariate_normal(mu_mle, Sigma_mle, size=Nsamp)
        x_kpc_gal, y_kpc_gal, v_gal = Xs[:, 0], Xs[:, 1], Xs[:, 2]

        # Convert back to ra, dec
        dlon = np.arctan2(x_kpc_gal, d_A_kpc)  # east
        dlat = np.arctan2(y_kpc_gal, d_A_kpc)  # north
        center = SkyCoord(ra_qso * u.deg, dec_qso * u.deg, frame="icrs")
        off_frame = SkyOffsetFrame(origin=center)
        gal_off = SkyCoord(lon=dlon * u.rad, lat=dlat * u.rad, frame=off_frame)
        gal_icrs = gal_off.transform_to("icrs")
        c_gal = w.world_to_pixel(SkyCoord(gal_icrs.ra.deg, gal_icrs.dec.deg, unit='deg', frame='icrs'))

        # Check galaxy locations
        # idx = 10
        # idx_start, idx_end = idx * Ngal, (idx + 1) * Ngal
        # plt.figure()
        # plt.scatter(c_gal[0][idx_start:idx_end], c_gal[1][idx_start:idx_end], c=v_gal[idx_start:idx_end],
        #             s=20, vmin=-1000, vmax=1000, cmap='coolwarm')
        # plt.imshow(v50, origin='lower', cmap='coolwarm', vmin=-1000, vmax=1000)
        # plt.xlim(-300, 300)
        # plt.ylim(-300, 300)
        # plt.colorbar(label='v50 (km/s)')
        # plt.title('Check galaxy positions')
        # plt.xlabel('X (pixel)')
        # plt.ylabel('Y (pixel)')
        # plt.show()

        # Start
        x, y = np.meshgrid(np.arange(v50.shape[1]), np.arange(v50.shape[0]))
        x, y = x.ravel(), y.ravel()
        v50_flat = v50.ravel()
        s80_flat = s80.ravel()

        # Nebular spaxels
        input_neb = np.column_stack([x, y, v50_flat])
        sigma_v_neb = s80_flat

        # Galaxys
        gal_mus_all = np.column_stack([c_gal[0], c_gal[1], v_gal]).astype(float)  # (G,3)
        sigma_x_gal = sigma_physical  # pixel
        sigma_y_gal = sigma_physical  # pixel
        sigma_v_gal = 20.0  # km/s

        # Calculate overlapping with batch
        CKAF_array = []
        for i in range(Ntrial):
            j0 = i * Ngal
            j1 = (i + 1) * Ngal
            gal_mus_batch = gal_mus_all[j0:j1]  # (B,3)
            overlap = bhattacharyya_coefficient(mus_neb=input_neb, sigma_x_neb=1.5,
                                                sigma_y_neb=1.5, sigma_v_neb=sigma_v_neb,
                                                mus_gal=gal_mus_batch, sigma_x_gal=sigma_x_gal,
                                                sigma_y_gal=sigma_y_gal, sigma_v_gal=sigma_v_gal)
            KAF_flat = np.sum(overlap, axis=1)
            CKAF_array.append(np.nansum(KAF_flat))

        # plt.figure()
        # plt.hist(CKAF_array, bins='auto', histtype='step', color='black')
        # plt.xlabel('CKAF')
        # plt.ylabel('Number of trials')
        # plt.show()
        # print(np.mean(CKAF_array))
        return CKAF_array

    def Derive3DDist_xyv(self):
        # Stack everything
        x_kpc_array = np.array([])
        y_kpc_array = np.array([])
        v_gal_array = np.array([])
        for cubename in self.cubename_all:
            # QSO information
            i = self._qso_index.get(cubename, None)
            ra_qso, dec_qso, z_qso = self.data_qso['ra_GAIA'][i], self.data_qso['dec_GAIA'][i], \
                                     self.data_qso['redshift'][i]

            # Load galaxy information
            path_gal = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
            data_gal = fits.open(path_gal)[1].data
            v_gal, ra_gal, dec_gal, type = data_gal['v'], data_gal['ra_HST'], data_gal['dec_HST'], data_gal['type']

            # Convert to radial profile
            center = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
            target = SkyCoord(ra_gal, dec_gal, unit='deg', frame='icrs')
            sep = center.separation(target).to(u.rad).value  # in arcsec
            pa = center.position_angle(target).to(u.rad).value  # in rad
            d_A_kpc = cosmo.angular_diameter_distance(z_qso).value * 1e3
            r_kpc = sep * d_A_kpc  # in kpc

            # Cartesian
            x_kpc = r_kpc * np.sin(pa)
            y_kpc = r_kpc * np.cos(pa)

            # Append to array
            x_kpc_array = np.hstack((x_kpc_array, x_kpc))
            y_kpc_array = np.hstack((y_kpc_array, y_kpc))
            v_gal_array = np.hstack((v_gal_array, v_gal))

        # Fit a 3D Gaussian to it
        X = np.column_stack([x_kpc_array, y_kpc_array, v_gal_array])
        mu_mle = X.mean(axis=0)
        Sigma_mle = np.cov(X, rowvar=False, bias=True)  # <-- MLE (divide by N)
        return mu_mle, Sigma_mle

    def SummarizeCorr(self):
        # Compute association for each Type
        CKAF_L, CKAF_S, CKAF_A = np.array([]), np.array([]), np.array([])
        control_mean_L, control_mean_S, control_mean_A = np.array([]), np.array([]), np.array([])
        control_sigma_L, control_sigma_S, control_sigma_A = np.array([]), np.array([]), np.array([])
        for i in range(len(self.L)):
            CKAF = self.ComputeCorr(cubename=self.L[i][0], nums_seg_OII=self.L[i][3], select_seg_OII=self.L[i][4],
                                    nums_seg_OIII=self.L[i][5], select_seg_OIII=self.L[i][6])
            CKAF_L = np.hstack((CKAF_L, CKAF))
            infile = "../../MUSEQuBES+CUBS/KAF/{}_CKAF_results_N={}.h5".format(self.L[i][0], self.Ntrial)
            with h5py.File(infile, 'r') as f:
                CKAF_array = f['CKAF_array'][:]
            control_mean_L = np.hstack((control_mean_L, np.mean(CKAF_array)))
            control_sigma_L = np.hstack((control_sigma_L, np.std(CKAF_array)))


        for i in range(len(self.S)):
            CKAF = self.ComputeCorr(cubename=self.S[i][0], nums_seg_OII=self.S[i][3], select_seg_OII=self.S[i][4],
                                    nums_seg_OIII=self.S[i][5], select_seg_OIII=self.S[i][6])
            CKAF_S = np.hstack((CKAF_S, CKAF))
            infile = "../../MUSEQuBES+CUBS/KAF/{}_CKAF_results_N={}.h5".format(self.S[i][0], self.Ntrial)
            with h5py.File(infile, 'r') as f:
                CKAF_array = f['CKAF_array'][:]
            print(self.S[i][0], CKAF, np.mean(CKAF_array), np.std(CKAF_array))
            control_mean_S = np.hstack((control_mean_S, np.mean(CKAF_array)))
            control_sigma_S = np.hstack((control_sigma_S, np.std(CKAF_array)))

        for i in range(len(self.A)):
            CKAF = self.ComputeCorr(cubename=self.A[i][0], nums_seg_OII=self.A[i][3], select_seg_OII=self.A[i][4],
                                    nums_seg_OIII=self.A[i][5], select_seg_OIII=self.A[i][6])
            CKAF_A = np.hstack((CKAF_A, CKAF))
            infile = "../../MUSEQuBES+CUBS/KAF/{}_CKAF_results_N={}.h5".format(self.A[i][0], self.Ntrial)
            with h5py.File(infile, 'r') as f:
                CKAF_array = f['CKAF_array'][:]
            control_mean_A = np.hstack((control_mean_A, np.mean(CKAF_array)))
            control_sigma_A = np.hstack((control_sigma_A, np.std(CKAF_array)))

        # Scatter plot
        CKAF_array = np.hstack((CKAF_L, CKAF_S, CKAF_A))
        res = stats.pearsonr(self.allType[:, 2], CKAF_array)

        plt.figure(figsize=(5, 5), dpi=100, constrained_layout=True)
        plt.scatter(self.L[:, 2], (CKAF_L - control_mean_L) / control_sigma_L, marker="o", alpha=0.8, s=50, color='k', label=r'Irregular, large-scale')
        plt.scatter(self.S[:, 2], (CKAF_S - control_mean_S) / control_sigma_S, marker="s", alpha=0.8, s=50, color='red', label=r'Host-galaxy-scale')
        plt.scatter(self.A[:, 2], (CKAF_A - control_mean_A) / control_sigma_A, marker="^", alpha=0.8, s=50, color='blue', label=r'Associated')
        plt.xlabel(r'$\rm Size \, [kpc]$', size=25)
        plt.ylabel(r'CKAF', size=25)
        plt.xlim(20, 225)
        # plt.legend(loc='best', fontsize=20)
        # plt.savefig('../../MUSEQuBES+CUBS/plots/CUBS+MUSE_CorrScore_ScaleLength.png', bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(5, 5), dpi=300, constrained_layout=True)
        plt.scatter(self.L[:, 2], CKAF_L, marker="o", alpha=0.8, s=50, color='k', label=r'Irregular, large-scale')
        plt.scatter(self.S[:, 2], CKAF_S, marker="s", alpha=0.8, s=50, color='red', label=r'Host-galaxy-scale')
        plt.scatter(self.A[:, 2], CKAF_A, marker="^", alpha=0.8, s=50, color='blue', label=r'Complex Morphology')
        plt.annotate(f'Pearson $r$ = {res[0]:.3f} \n p-value = {res[1]:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                     fontsize=15, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
        plt.xlabel(r'$\rm Size \, [kpc]$', size=25)
        plt.ylabel(r'CKAF', size=25)
        plt.xlim(20, 225)
        plt.legend(loc='best', fontsize=20)
        plt.savefig('../../MUSEQuBES+CUBS/plots/CUBS+MUSE_CorrScore_ScaleLength.png', bbox_inches='tight')

    def CalculateCorrControl(self):
        # Compute association for each Type
        for i in range(len(self.allType)):
            outfile = "../../MUSEQuBES+CUBS/KAF/{}_CKAF_results_N={}.h5".format(self.allType[i][0], self.Ntrial)

            if os.path.exists(outfile):
                print(f"{outfile} already exists. Skipping computation.")
                continue
            CKAF_array = self.ComputeCorrControl(cubename=self.allType[i][0], nums_seg_OII=self.allType[i][3],
                                                 select_seg_OII=self.allType[i][4], nums_seg_OIII=self.allType[i][5],
                                                 select_seg_OIII=self.allType[i][6], Ntrial=self.Ntrial)
            # Save the result
            with h5py.File(outfile, 'w') as f:
                f.create_dataset('CKAF_array', data=CKAF_array)

            # Plot histogram
            plt.figure(figsize=(5, 5), )
            plt.hist(CKAF_array, bins='auto', histtype='step', color='black')
            mean_CKAF = np.mean(CKAF_array)
            sigma_CKAF = np.std(CKAF_array)
            plt.axvline(mean_CKAF, color='red', linestyle='dashed', linewidth=1,
                        label='Mean = {:.3f}\nSigma = {:.3f}'.format(mean_CKAF, sigma_CKAF))
            plt.legend()
            plt.xlabel('CKAF')
            plt.ylabel('Number of trials')
            plt.savefig('../../MUSEQuBES+CUBS/plots/Control_{}_CKAF_Hist.png'.format(self.allType[i][0]),
                        bbox_inches='tight')


L = np.array([["HE0226-4110",     150,  84, [2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], False,
               [1, 5, 6, 8, 9, 10, 11, 16, 19], False],
              ["PKS0405-123",     106, 129, [5, 7, 10, 11, 13, 16, 17, 20], False, [15], False],
              ["HE0238-1904",     113, 103, [1, 6, 12, 13, 17, 19], True, [1, 2, 4, 9, 13, 15, 17, 20], True],
              ["PKS0552-640",     124, 153, [2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], False,
               [5, 6, 7, 8, 12, 15, 16, 17, 18, 20], False],
              ["J0454-6116",      151, 102, [2, 3, 4, 5, 6, 8, 11, 12, 13, 15, 17, 18], False,
               [2, 7, 9, 10, 18, 19], False],
              ["J0119-2010",      166,  96, [3, 4, 6, 7, 10, 11, 12, 14, 16, 17, 18, 20], False,
               [7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20], False],
              ["HE0246-4101",     129,  74, [1], True, [], False],
              ["PKS0355-483",     142,  50, [2, 3, 4, 8, 9, 10, 11], True, [], False],
              ["HE0439-5254",     246,  47, [], False, [], False],
              ["TEX0206-048",     194, 200, [1, 8, 12, 13, 15, 20, 23, 26, 27, 28, 34, 57, 60, 79, 81,
                                             101, 107, 108, 114, 118, 317, 547, 552], True, [], False],
              ["Q1354+048",       123, 126, [1, 2], False, [], False]], dtype=object)
S = np.array([["HE0435-5304",      87,  55, [1], False, [1], False],
              ["3C57",            151,  71, [2], False, [], False],
              ["J0110-1648",       91,  29, [1], False, [2], False],
              ["HE0112-4145",     164,  38, [], False, [], False],
              ["J0154-0712",      137,  63, [5], False, [], False],
              ["LBQS1435-0134",    261,  63, [1, 3, 7], True, [], False ],
              ["J0028-3305",      133,  42, [2], True, [], False],
              ["HE0419-5657",     154,  35, [2, 4, 5], True, [], False],
              ["PB6291",          116,  28, [2, 6, 7], True, [], False],
              ["HE1003+0149",     208,  53, [], False, [], False],
              ["HE0331-4112",     196,  32, [6], True, [], False]], dtype=object)

A = np.array([["J2135-5316",      107,  83, [2, 3, 4, 6, 10, 12, 13, 14, 16, 17, 18, 19], False,
               [4, 7, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20], False],
              ["Q0107-0235",      136,  90, [1, 4, 5, 6], True, [], False],
              ["PKS2242-498",     147,  71, [1, 2], True, [], False],
              ["PG1522+101",      132,  50, [2, 3, 8, 11], True, [], False],
              ["PKS0232-04",      178, 116, [2, 4, 5, 7], False, [], False]], dtype=object)


# Test
# ComputeCorr(cubename='HE0226-4110', scale_length=84)
# ComputeCorr(cubename='PKS0405-123', scale_length=130)
# ComputeCorr(cubename='HE0238-1904', scale_length=103)
# ComputeCorr(cubename='PKS0552-640', scale_length=153)
# ComputeCorr(cubename='3C57', scale_length=71)
# ComputeCorr(cubename='Q0107-0235', scale_length=90)
# ComputeCorr(cubename='TEX0206-048', scale_length=200)
# ComputeCorr(cubename='PB6291', scale_length=28, savefig=True, nums_seg_OII=[2, 6, 7], select_seg_OII=True)
# SummarizeCorr(L=L, S_BR=S_BR, S=S, A=A)

func = CalculateGalNebCorr(L=L, S=S, A=A)
func.SummarizeCorr()
# func.CalculateCorrControl()
# func.ComputeCorrControl(cubename='PKS0405-123')
# func.ComputeCorrControl(cubename='Q0107-0235')