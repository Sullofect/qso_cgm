import os
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from matplotlib import rc
from astropy.wcs import WCS
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from photutils.profiles import RadialProfile
from astropy.convolution import convolve, Kernel, Gaussian1DKernel, Gaussian2DKernel, Box2DKernel, Box1DKernel
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)
rc('xtick.minor', size=5, visible=True)
rc('ytick.minor', size=5, visible=True)
rc('xtick', direction='in', labelsize=25, top='on')
rc('ytick', direction='in', labelsize=25, right='on')
rc('xtick.major', size=8)
rc('ytick.major', size=8)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Quasars with jet
L = np.array([["PKS0405-123",     106, 129, [5, 7, 10, 11, 13, 16, 17, 20], False, [15], False]], dtype=object)
S = np.array([["3C57",            151,  71, [2], False, [], False],
              ["J0110-1648",       91,  29, [1], False, [2], False]], dtype=object)
A = np.array([["Q0107-0235",      136,  90, [1, 4, 5, 6], True, [], False],
              ["PKS2242-498",     147,  71, [1, 2], True, [], False],
              ["PKS0232-04",      178, 116, [2, 4, 5, 7], False, [], False]], dtype=object)
allType = np.vstack((L, S, A))

# Radio lobe positions
radio_lobe = {"PKS0405-123": [[61.9506047, -12.1970858], [61.9523443, -12.1882108]],
              "3C57": [[30.4897392, -11.5461023], [30.4882094, -11.5424713]],
              "J0110-1648": [[17.6418076, -16.8121537], [17.6479416, -16.8076752]],
              "Q0107-0235": [[17.5557241, -2.3355796], [17.5548950, -2.3312814]],
              "PKS2242-498": [[341.2550230, -49.5247734], [341.2412548, -49.5403274]],
              "PKS0232-04": [[38.7779804, -4.0346468], [38.7815349, -4.0344447]]}

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def wrap_to_180(angle_deg):
    """Wrap angles to [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0

def angle_difference_deg(a, b):
    """Smallest signed difference a-b in degrees, wrapped to [-180, 180)."""
    return wrap_to_180(a - b)

def get_relative_lobe_angle(qso_xy, lobe0_xy, lobe1_xy):
    # angle of first lobe
    dx0 = lobe0_xy[0] - qso_xy[0]
    dy0 = lobe0_xy[1] - qso_xy[1]
    theta0 = np.degrees(np.arctan2(dy0, dx0))

    # angle of second lobe
    dx1 = lobe1_xy[0] - qso_xy[0]
    dy1 = lobe1_xy[1] - qso_xy[1]
    theta1 = np.degrees(np.arctan2(dy1, dx1))

    # second lobe angle relative to first lobe = 0 deg
    theta1_rel = (theta1 - theta0 + 180) % 360 - 180
    return theta1_rel

def sky_to_pixel(ra_deg, dec_deg, wcs):
    """
    Convert world coordinates to pixel coordinates.
    Returns x, y in numpy image convention.
    """
    x, y = wcs.all_world2pix(ra_deg, dec_deg, 0)
    return float(x), float(y)

def compute_axis_angle_from_lobe(qso_xy, lobe_xy):
    """
    Compute image-plane angle in degrees for the vector from quasar to a lobe.

    Convention here:
    angle = arctan2(dy, dx), in degrees
    so:
      0 deg   -> +x direction
      90 deg  -> +y direction

    This is fine as long as we use the same convention everywhere.
    """
    dx = lobe_xy[0] - qso_xy[0]
    dy = lobe_xy[1] - qso_xy[1]
    return np.degrees(np.arctan2(dy, dx))

def make_azimuthal_profile(image, qso_xy, axis_angle_deg, rmin_pix=0.0, rmax_pix=None, mask=None, nbins=18,
                           statistic="mean"):
    """
    Compute azimuthal profile relative to a chosen axis.

    Parameters
    ----------
    image : 2D ndarray
        Emission / SB map.
    qso_xy : tuple
        (x0, y0) quasar position in pixels.
    axis_angle_deg : float
        0 deg will point along this axis.
    rmin_pix, rmax_pix : float
        Radial selection in pixels.
    mask : 2D bool ndarray or None
        True means keep pixel, False means exclude pixel.
    nbins : int
        Number of angular bins over 360 deg.
    statistic : {'mean', 'median', 'sum'}
        Statistic per angular bin.

    Returns
    -------
    results : dict
        Contains bin centers, values, errors, counts, etc.
    """
    ny, nx = image.shape
    y, x = np.indices((ny, nx))

    dx = x - qso_xy[0]
    dy = y - qso_xy[1]
    r = np.sqrt(dx**2 + dy**2)

    # Pixel angle in image plane
    theta = np.degrees(np.arctan2(dy, dx))

    # Re-center so 0 deg points toward chosen radio lobe
    theta_rel = wrap_to_180(theta - axis_angle_deg)

    valid = np.isfinite(image)

    if mask is not None:
        valid &= mask.astype(bool)

    valid &= (r >= rmin_pix)
    if rmax_pix is not None:
        valid &= (r < rmax_pix)

    theta_use = theta_rel[valid]
    image_use = image[valid]

    # Angular bins over [-180, 180)
    edges = np.linspace(-180.0, 180.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    values = np.full(nbins, np.nan)
    errs = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        in_bin = (theta_use >= edges[i]) & (theta_use < edges[i + 1])
        vals = image_use[in_bin]
        vals = vals[np.isfinite(vals)]
        counts[i] = len(vals)

        if len(vals) == 0:
            continue

        if statistic == "mean":
            values[i] = np.mean(vals)
            errs[i] = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        elif statistic == "median":
            values[i] = np.median(vals)
            errs[i] = 1.253 * np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        elif statistic == "sum":
            values[i] = np.sum(vals)
            errs[i] = np.sqrt(np.sum(vals**2)) / len(vals) if len(vals) > 0 else np.nan
        else:
            raise ValueError("statistic must be 'mean', 'median', or 'sum'.")

    return {
        "bin_edges": edges,
        "bin_centers": centers,
        "bin_width_deg": width,
        "values": values,
        "errors": errs,
        "counts": counts,
        "theta_rel": theta_use,
        "image_use": image_use,
    }

def make_folded_profile(image, qso_xy, axis_angle_deg, rmin_pix=0.0, rmax_pix=None, mask=None, nbins=9,
                        statistic="mean"):
    """
    Fold the azimuthal profile about the radio axis.

    Returns angle from axis in [0, 90] deg:
      0 deg  = along axis (either lobe direction)
      90 deg = perpendicular to axis
    """
    ny, nx = image.shape
    y, x = np.indices((ny, nx))

    dx = x - qso_xy[0]
    dy = y - qso_xy[1]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.degrees(np.arctan2(dy, dx))
    theta_rel = wrap_to_180(theta - axis_angle_deg)

    # Fold onto [0, 90]
    # First take absolute angle from axis in [0, 180]
    theta_abs = np.abs(theta_rel)
    # Fold 180 -> 0 symmetry
    theta_fold = np.minimum(theta_abs, 180.0 - theta_abs)

    valid = np.isfinite(image)

    if mask is not None:
        valid &= mask.astype(bool)

    valid &= (r >= rmin_pix)
    if rmax_pix is not None:
        valid &= (r < rmax_pix)

    theta_use = theta_fold[valid]
    image_use = image[valid]

    edges = np.linspace(0.0, 90.0, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    values = np.full(nbins, np.nan)
    errs = np.full(nbins, np.nan)
    counts = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        in_bin = (theta_use >= edges[i]) & (theta_use < edges[i + 1])
        vals = image_use[in_bin]
        vals = vals[np.isfinite(vals)]
        counts[i] = len(vals)

        if len(vals) == 0:
            continue

        if statistic == "mean":
            values[i] = np.mean(vals)
            errs[i] = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        elif statistic == "median":
            values[i] = np.median(vals)
            errs[i] = 1.253 * np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        elif statistic == "sum":
            values[i] = np.sum(vals)
            errs[i] = np.sqrt(np.sum(vals**2)) / len(vals) if len(vals) > 0 else np.nan
        else:
            raise ValueError("statistic must be 'mean', 'median', or 'sum'.")

    return {
        "bin_edges": edges,
        "bin_centers": centers,
        "bin_width_deg": width,
        "values": values,
        "errors": errs,
        "counts": counts,
    }


def plot_azimuthal_profiles(az_result, fold_result=None, axis_angle_deg_1=None, cubename=None, show_counts=False,):
    """
    Quick-look plotting.
    """
    if fold_result is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Full 360 profile
    ax = axes[0]
    ax.errorbar(az_result["bin_centers"], az_result["values"], yerr=az_result["errors"], fmt="o-", lw=1.8, capsize=3)
    ax.axvline(0, ls="--", color="0.5", lw=1.2)
    ax.axvline(axis_angle_deg_1, ls="--", color="0.5", lw=1.2)
    ax.axvline(180, ls="--", color="0.5", lw=1.2)
    ax.axvline(-180, ls="--", color="0.5", lw=1.2)
    ax.set_xlim(-180, 180)
    ax.set_xlabel("Angle relative to radio axis [deg]")
    ax.set_ylabel("Mean emission intensity")
    ax.set_title(f"{cubename}: azimuthal profile" if cubename else "Azimuthal profile")

    if show_counts:
        for x, y, n in zip(az_result["bin_centers"], az_result["values"], az_result["counts"]):
            if np.isfinite(y):
                ax.text(x, y, str(n), fontsize=8, ha="center", va="bottom")

    # Folded profile
    if fold_result is not None:
        ax = axes[1]
        ax.errorbar(
            fold_result["bin_centers"],
            fold_result["values"],
            yerr=fold_result["errors"],
            fmt="o-",
            lw=1.8,
            capsize=3,
        )
        ax.set_xlim(0, 90)
        ax.set_xlabel("Angle from radio axis [deg]")
        ax.set_ylabel("Mean emission intensity")
        ax.set_title(f"{cubename}: folded profile" if cubename else "Folded profile")

        if show_counts:
            for x, y, n in zip(fold_result["bin_centers"], fold_result["values"], fold_result["counts"]):
                if np.isfinite(y):
                    ax.text(x, y, str(n), fontsize=8, ha="center", va="bottom")

    fig.tight_layout()
    return fig, axes

def EmissionRadio(cubename=None,
                  str_zap="",
                  line="OIII",
                  rmin_arcsec=0.0,
                  rmax_arcsec=8.0,
                  nbins_az=18,
                  nbins_fold=9,
                  statistic="mean",
                  sb_threshold=None,
                  use_manual_mask=None,
                  show_plot=True,
                  ):
    """
    Compute emission intensity vs angle relative to the radio axis.

    Parameters
    ----------
    cubename : str
        QSO / cube name, must exist in radio_lobe dict.
    str_zap : str
        Your filename suffix piece.
    line : str
        'OII', 'OIII', or whatever map you want to use.
    lobe_index_for_zero : int
        Which lobe defines 0 deg. Use 0 or 1.
    rmin_arcsec, rmax_arcsec : float
        Radial annulus for the profile.
    nbins_az : int
        Number of bins for full 360 profile.
    nbins_fold : int
        Number of bins for folded profile.
    statistic : str
        'mean', 'median', or 'sum'.
    sb_threshold : float or None
        If provided, only pixels with image >= threshold are kept.
    use_manual_mask : 2D bool ndarray or None
        Optional extra mask; True means keep pixel.
    show_plot : bool
        Whether to make plots.

    Returns
    -------
    out : dict
        Results dictionary.
    """
    if cubename not in radio_lobe:
        raise ValueError(f"{cubename} not found in radio_lobe dictionary.")

    # --------------------------------------------------------
    # QSO information
    # --------------------------------------------------------
    path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]

    if len(data_qso) == 0:
        raise ValueError(f"Could not find {cubename} in quasars.dat")

    ra_qso = data_qso['ra_GAIA'][0]
    dec_qso = data_qso['dec_GAIA'][0]
    z_qso = data_qso['redshift'][0]

    # --------------------------------------------------------
    # Load 2D emission map
    # Adjust this path pattern to your actual file naming.
    # --------------------------------------------------------
    UseSeg = (1.5, 'gauss', 1.5, 'gauss')
    path_map = ('../../MUSEQuBES+CUBS/fit_kin/''{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits')\
        .format(cubename, str_zap, line, *UseSeg)

    if not os.path.exists(path_map):
        raise FileNotFoundError(f"Could not find map:\n{path_map}")

    image = fits.open(path_map)[1].data
    header =  fits.open(path_map)[1].header
    wcs = WCS(header).celestial

    # --------------------------------------------------------
    # Convert QSO + radio lobes to pixel coordinates
    # --------------------------------------------------------
    qso_xy = sky_to_pixel(ra_qso, dec_qso, wcs)
    lobe0_xy = sky_to_pixel(radio_lobe[cubename][0][0], radio_lobe[cubename][0][1], wcs)
    lobe1_xy = sky_to_pixel(radio_lobe[cubename][1][0], radio_lobe[cubename][1][1], wcs)
    axis_angle_deg = compute_axis_angle_from_lobe(qso_xy, lobe0_xy)
    axis_angle_deg_1 = get_relative_lobe_angle(qso_xy, lobe0_xy, lobe1_xy)

    # --------------------------------------------------------
    # Radius selection
    # --------------------------------------------------------
    pixscale = 0.2
    rmin_pix = rmin_arcsec / pixscale
    rmax_pix = rmax_arcsec / pixscale if rmax_arcsec is not None else None

    # --------------------------------------------------------
    # Build mask
    # True = keep
    # --------------------------------------------------------
    mask = np.isfinite(image)

    if sb_threshold is not None:
        mask &= (image >= sb_threshold)

    if use_manual_mask is not None:
        mask &= use_manual_mask.astype(bool)

    # --------------------------------------------------------
    # Compute profiles
    # --------------------------------------------------------
    az_result = make_azimuthal_profile(
        image=image,
        qso_xy=qso_xy,
        axis_angle_deg=axis_angle_deg,
        rmin_pix=rmin_pix,
        rmax_pix=rmax_pix,
        mask=mask,
        nbins=nbins_az,
        statistic=statistic,
    )

    # fold_result = make_folded_profile(
    #     image=image,
    #     qso_xy=qso_xy,
    #     axis_angle_deg=axis_angle_deg,
    #     rmin_pix=rmin_pix,
    #     rmax_pix=rmax_pix,
    #     mask=mask,
    #     nbins=nbins_fold,
    #     statistic=statistic,
    # )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    if show_plot:
        plot_azimuthal_profiles(az_result=az_result,
                                axis_angle_deg_1=axis_angle_deg_1,
                                fold_result=None,
                                cubename=cubename
                                )
        plt.show()

    return {"cubename": cubename,
            "z_qso": z_qso,
            "path_map": path_map,
            "image": image,
            "header": header,
            "wcs": wcs,
            "qso_xy": qso_xy,
            "lobe0_xy": lobe0_xy,
            "lobe1_xy": lobe1_xy,
            "axis_angle_deg": axis_angle_deg,
            "pixscale_arcsec": pixscale,
            "azimuthal_profile": az_result,
            }


# out = EmissionRadio(
#         cubename="PKS0405-123",
#         str_zap="",
#         line="OII",
#         rmin_arcsec=1.0,         # exclude central PSF-dominated region
#         rmax_arcsec=None,
#         nbins_az=18,             # 20-degree bins
#         nbins_fold=9,            # 10-degree bins in folded profile
#         statistic="mean",
#         sb_threshold=0.0,
#         show_plot=True,
# )

# out = EmissionRadio(
#         cubename="3C57",
#         str_zap="",
#         line="OIII",
#         rmin_arcsec=1.0,         # exclude central PSF-dominated region
#         rmax_arcsec=None,
#         nbins_az=18,             # 20-degree bins
#         nbins_fold=9,            # 10-degree bins in folded profile
#         statistic="mean",
#         sb_threshold=0.0,
#         show_plot=True,
# )

# out = EmissionRadio(
#         cubename="J0110-1648",
#         str_zap="",
#         line="OII",
#         rmin_arcsec=1.0,         # exclude central PSF-dominated region
#         rmax_arcsec=None,
#         nbins_az=18,             # 20-degree bins
#         nbins_fold=9,            # 10-degree bins in folded profile
#         statistic="mean",
#         sb_threshold=0.0,
#         show_plot=True,
# )

out = EmissionRadio(
    cubename="Q0107-0235",
    str_zap="",
    line="OII",
    rmin_arcsec=1.0,         # exclude central PSF-dominated region
    rmax_arcsec=None,
    nbins_az=18,             # 20-degree bins
    nbins_fold=9,            # 10-degree bins in folded profile
    statistic="mean",
    sb_threshold=0.0,
    show_plot=True,
)

out = EmissionRadio(
    cubename="PKS2242-498",
    str_zap="",
    line="OII",
    rmin_arcsec=1.0,         # exclude central PSF-dominated region
    rmax_arcsec=None,
    nbins_az=18,             # 20-degree bins
    nbins_fold=9,            # 10-degree bins in folded profile
    statistic="mean",
    sb_threshold=0.0,
    show_plot=True,
)

# out = EmissionRadio(
#     cubename="PKS0232-04",
#     str_zap="",
#     line="OII",
#     rmin_arcsec=1.0,         # exclude central PSF-dominated region
#     rmax_arcsec=None,
#     nbins_az=18,             # 20-degree bins
#     nbins_fold=9,            # 10-degree bins in folded profile
#     statistic="mean",
#     sb_threshold=0.0,
#     show_plot=True,
# )




