import os
import aplpy
import numpy as np
import matplotlib as mpl
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import rc
from astropy.wcs import WCS
from astropy.io import ascii
from regions import PixCoord
from astropy import units as u
from astropy.coordinates import SkyCoord
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from astropy.cosmology import FlatLambdaCDM
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

# Part 1
# Charaize the 3D distribution of galaxies
# Fields information
cubename_L = ["HE0226-4110", "PKS0405-123", "HE0238-1904", "PKS0552-640", "J0454-6116", "J0119-2010", "HE0246-4101",
              "PKS0355-483", "HE0439-5254", "TEX0206-048", "Q1354+048"]
cubename_S = ["HE0435-5304", "3C57", "J0110-1648", "HE0112-4145", "J0154-0712", "LBQS1435-0134", "J0028-3305",
              "HE0419-5657", "PB6291", "HE1003+0149", "HE0331-4112"]
cubename_A = ["J2135-5316", "Q0107-0235", "PKS2242-498", "PG1522+101", "PKS0232-04"]
cubename_all = cubename_L + cubename_S + cubename_A
# cubename_all = ['HE0238-1904']  # For testing

def Derive3DDist_polar():
    # Stack everything
    pa_array = np.array([])
    r_kpc_array = np.array([])
    v_gal_array = np.array([])
    for cubename in cubename_all:
        # QSO information
        path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
        data_qso = ascii.read(path_qso, format='fixed_width')
        data_qso = data_qso[data_qso['name'] == cubename]
        ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

        # Load galaxy information
        path_gal = '../../MUSEQuBES+CUBS/gal_info/{}_gal_info_gaia.fits'.format(cubename)
        data_gal = fits.open(path_gal)[1].data
        v_gal, ra_gal, dec_gal, type = data_gal['v'], data_gal['ra_HST'], data_gal['dec_HST'], data_gal['type']

        # Convert to radial profile
        center = SkyCoord(ra_qso, dec_qso, unit='deg', frame='icrs')
        target = SkyCoord(ra_gal, dec_gal, unit='deg', frame='icrs')
        sep = center.separation(target).arcsec  # in arcsec
        pa = center.position_angle(target).to(u.rad).value  # in rad
        d_A_kpc = cosmo.angular_diameter_distance(z_qso).value * 1e3
        r_kpc = sep / 206265 * d_A_kpc # in kpc

        # Append to array
        pa_array = np.hstack((pa_array, pa))
        r_kpc_array = np.hstack((r_kpc_array, r_kpc))
        v_gal_array = np.hstack((v_gal_array, v_gal))

    # Make polar plot
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': 'polar'}, layout='constrained')
    # sc = ax.scatter(pa_array, r_kpc_array, c=v_gal_array, cmap='coolwarm', s=50, vmin=-1500, vmax=1500)
    # plt.colorbar(sc, ax=ax, pad=0.1, label="velocity")
    # ax.set_theta_zero_location('N')  # 0° at North
    # ax.set_theta_direction(1)  # clockwise = East to the left
    # ax.set_rlabel_position(90)
    # ax.grid(True)
    # plt.show()

    # Fit a 3D Gaussian to it
    X = np.column_stack([np.cos(pa_array), np.sin(pa_array), r_kpc_array, v_gal_array])  # (N,4)
    mu_mle = X.mean(axis=0)
    Sigma_mle = np.cov(X, rowvar=False, bias=True)  # <-- MLE (divide by N)

    #
    # pa0 = (np.rad2deg(np.arctan2(mu_mle[1], mu_mle[0]))) % 360
    # R = np.hypot(mu_mle[0], mu_mle[1])  # concentration proxy
    #
    # print("mean PA:", pa0, "deg")
    # print("R:", R)

    # Compare
    rng = np.random.default_rng(0)

    # --- Monte Carlo samples from fitted 4D Gaussian
    Nsamp = 5000
    Xs = rng.multivariate_normal(mu_mle, Sigma_mle, size=Nsamp)
    theta_m = np.arctan2(Xs[:, 1], Xs[:, 0]) % (2 * np.pi)  # from (sin, cos)
    r_m = Xs[:, 2]  # distance component

    # --- Choose common radial limits (so panels compare directly)
    rmin = np.nanmin(r_kpc_array)
    rmax = np.nanmax(r_kpc_array)

    # You can also clamp model samples to the plotting range if you want
    mask = (r_m >= rmin) & (r_m <= rmax)
    theta_m = theta_m[mask]
    r_m = r_m[mask]

    # --- Build model density in (theta, r) for the right panel
    n_theta = 90
    n_r = 80
    theta_edges = np.linspace(0, 2 * np.pi, n_theta + 1)
    r_edges = np.linspace(rmin, rmax, n_r + 1)
    H, _, _ = np.histogram2d(theta_m, r_m, bins=[theta_edges, r_edges], density=True)

    # Centers for pcolormesh
    Theta_grid, R_grid = np.meshgrid(theta_edges, r_edges, indexing="ij")  # edges grids

    # --- Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(11, 5), constrained_layout=True)

    for ax in (ax1, ax2):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)
        ax.set_rlim(rmin, rmax)

    # Left: data scatter
    sc = ax1.scatter(pa_array, r_kpc_array, c=v_gal_array, cmap='coolwarm', s=10, vmin=-1500, vmax=1500)
    fig.colorbar(sc, ax=ax1, pad=0.1, label="velocity")
    ax1.set_title("Data (PA, distance)")

    # Right: model density map
    pcm = ax2.pcolormesh(Theta_grid, R_grid, H, shading="auto")
    cb2 = fig.colorbar(pcm, ax=ax2, pad=0.08)
    cb2.set_label("model density")

    # theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    # r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    # THc, Rc = np.meshgrid(theta_centers, r_centers, indexing="ij")
    # ax2.contour(THc, Rc, H, levels=6, linewidths=1)

    ax2.set_title("Fitted Gaussian (projected to PA–distance)")
    plt.show()


    # Check distribution on velcity
    bins = np.arange(-1500, 1700, 200)
    plt.figure()
    plt.hist(v_gal_array, bins=bins, histtype='step', color='black')
    plt.xlim(-1500, 1500)
    plt.show()

    raise ValueError("Stop here")

    # Simulate galaxy and get a

def Derive3DDist_xyv():
    # Stack everything
    x_kpc_array = np.array([])
    y_kpc_array = np.array([])
    v_gal_array = np.array([])
    for cubename in cubename_all:
        # QSO information
        path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
        data_qso = ascii.read(path_qso, format='fixed_width')
        data_qso = data_qso[data_qso['name'] == cubename]
        ra_qso, dec_qso, z_qso = data_qso['ra_GAIA'][0], data_qso['dec_GAIA'][0], data_qso['redshift'][0]

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
        r_kpc = sep * d_A_kpc # in kpc

        # Cartesian
        x_kpc = r_kpc * np.sin(pa)
        y_kpc = r_kpc * np.cos(pa)

        # Append to array
        x_kpc_array = np.hstack((x_kpc_array, x_kpc))
        y_kpc_array = np.hstack((y_kpc_array, y_kpc))
        v_gal_array = np.hstack((v_gal_array, v_gal))

    # Check
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # sc = ax.scatter(x_kpc_array, y_kpc_array, c=v_gal_array, cmap='coolwarm', s=50, vmin=-1500, vmax=1500)
    # plt.colorbar(sc, ax=ax, pad=0.1, label="velocity")
    # plt.gca().invert_xaxis()
    # ax.grid(True)
    # plt.show()

    # Fit a 3D Gaussian to it
    X = np.column_stack([x_kpc_array, y_kpc_array, v_gal_array])
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    # mu_mle = X.mean(axis=0)
    # Sigma_mle = np.cov(X, rowvar=False, bias=True)  # <-- MLE (divide by N)

    # Try Gaussian mixturem models
    gmm = GaussianMixture(n_components=5, covariance_type="full", reg_covar=1e-4, n_init=10, random_state=1)
    gmm.fit(Xz)
    print(gmm.converged_)

    # --- Monte Carlo samples from fitted 4D Gaussian
    rng = np.random.default_rng(0)
    Nsamp = 5000
    # Xs = rng.multivariate_normal(mu_mle, Sigma_mle, size=Nsamp)
    Zs, labels = gmm.sample(Nsamp * 5)
    Xs = scaler.inverse_transform(Zs)

    # Sanity Check
    pairs = [(0, 1), (0, 2), (1, 2)]
    for a, b in pairs:
        plt.figure()
        plt.scatter(X[:, a], X[:, b], s=10, alpha=1, label="data")
        plt.scatter(Xs[:, a], Xs[:, b], s=5, alpha=0.3, label="gmm samp")
        plt.xlabel(f"dim {a}");
        plt.ylabel(f"dim {b}")
        plt.legend()
        plt.show()

    # Comparison plots
    x_m = Xs[:, 0]
    y_m = Xs[:, 1]
    v_m = Xs[:, 2]
    N_data = len(x_kpc_array)
    N_mc = len(x_m)
    w_mc = N_data / N_mc

    x_lo, x_hi = np.nanpercentile(x_kpc_array, [0.5, 99.5])
    y_lo, y_hi = np.nanpercentile(y_kpc_array, [0.5, 99.5])
    print("x range: ", x_lo, x_hi)
    print("y range: ", y_lo, y_hi)

    bins1 = np.arange(-300, 350, 50)
    bins2 = np.arange(-300, 350, 50)
    bins3 = np.arange(-1500, 1750, 250)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].hist(x_kpc_array, bins=bins1, histtype="step", color="black", linewidth=2, label="Data")
    axes[0].hist(x_m, bins=bins1, weights=np.full_like(x_m, w_mc), histtype="step", color="red", linewidth=2, label="Model (MC)")
    axes[1].hist(y_kpc_array, bins=bins2,  histtype="step", color="black", linewidth=2, label="Data")
    axes[1].hist(y_m, bins=bins2, weights=np.full_like(x_m, w_mc), histtype="step", color="red", linewidth=2, label="Model (MC)")
    axes[2].hist(v_gal_array, bins=bins3, histtype="step", color="black", linewidth=2, label="Data")
    axes[2].hist(v_m, bins=bins3, weights=np.full_like(x_m, w_mc), histtype="step", color="red", linewidth=2, label="Model (MC)")
    # one legend for the whole figure (cleaner)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    axes[0].set_xlabel("x (kpc)")
    axes[0].set_ylabel("density")
    axes[1].set_xlabel("y (kpc)")
    axes[1].set_ylabel("density")
    axes[2].set_xlabel("v")
    axes[2].set_ylabel("density")
    plt.savefig('../../MUSEQuBES+CUBS/plots/CUBS+MUSE_CompareGalNebCorrFit.png'.format(cubename), bbox_inches='tight')


Derive3DDist_xyv()
