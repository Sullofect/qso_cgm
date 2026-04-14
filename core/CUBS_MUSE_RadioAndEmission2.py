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
from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog

# Quasars with jet
L = np.array([["PKS0405-123",     106, 129, [5, 7, 10, 11, 13, 16, 17, 20], False, [15], False]], dtype=object)
S = np.array([["3C57",            151,  71, [2], False, [], False],
              ["J0110-1648",       91,  29, [1], False, [2], False]], dtype=object)
A = np.array([["Q0107-0235",      136,  90, [1, 4, 5, 6], True, [], False],
              ["PKS2242-498",     147,  71, [1, 2], True, [], False],
              ["PKS0232-04",      178, 116, [2, 4, 5, 7], False, [], False]], dtype=object)
allType = np.vstack((L, S, A))


radio_lobe = {"PKS0405-123": [[61.9506047, -12.1970858], [61.9523443, -12.1882108]],
              "3C57": [[30.4897392, -11.5461023], [30.4882094, -11.5424713]],
              "J0110-1648": [[17.6418076, -16.8121537], [17.6479416, -16.8076752]],
              "Q0107-0235": [[17.5557241, -2.3355796], [17.5548950, -2.3312814]],
              "PKS2242-498": [[341.2550230, -49.5247734], [341.2412548, -49.5403274]],
              "PKS0232-04": [[38.7779804, -4.0346468], [38.7815349, -4.0344447]]}

def sky_to_pixel(ra_deg, dec_deg, wcs):
    x, y = wcs.all_world2pix(ra_deg, dec_deg, 0)
    return float(x), float(y)


def wrap_180(angle_deg):
    """Wrap angle to [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def angle_diff_180(a_deg, b_deg):
    """
    Smallest difference between two position angles, treating axes modulo 180 deg.
    Returns value in [0, 90].
    """
    d = abs((a_deg - b_deg) % 180.0)
    return min(d, 180.0 - d)


def angle_diff_360(a_deg, b_deg):
    """
    Smallest difference between two directions, modulo 360 deg.
    Returns value in [0, 180].
    """
    d = abs((a_deg - b_deg) % 360.0)
    return min(d, 360.0 - d)


def pa_from_east_ccw(p0, p1):
    """
    Position angle in the image plane:
      0 deg = +x direction
      90 deg = +y direction
    increasing counterclockwise.
    """
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    return np.degrees(np.arctan2(dy, dx)) % 360.0

def quasar_centered_morphology_pa(image, qso_xy):
    img = np.array(image, dtype=float)
    valid = np.isfinite(img)
    yy, xx = np.nonzero(valid)
    flux = img[yy, xx]

    xq, yq = qso_xy
    dx = xx.astype(float) - xq
    dy = yy.astype(float) - yq

    wsum = np.sum(flux)

    Ixx = np.sum(flux * dx * dx) / wsum
    Iyy = np.sum(flux * dy * dy) / wsum
    Ixy = np.sum(flux * dx * dy) / wsum

    # PA of major axis: 0 deg = +x, CCW positive, modulo 180
    major_pa_deg = 0.5 * np.degrees(np.arctan2(2.0 * Ixy, Ixx - Iyy)) % 180.0
    minor_pa_deg = (major_pa_deg + 90.0) % 180.0

    # Covariance matrix and eigenvalues
    cov = np.array([[Ixx, Ixy],
                    [Ixy, Iyy]])
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)  # ascending

    # axis ratio from second moments
    axis_ratio = np.sqrt(eigvals[0] / eigvals[1]) if eigvals[1] > 0 else np.nan

    return {
        "major_pa_deg": major_pa_deg,
        "minor_pa_deg": minor_pa_deg,
        "Ixx": Ixx,
        "Iyy": Iyy,
        "Ixy": Ixy,
        "eigvals": eigvals,
        "axis_ratio": axis_ratio,
    }

def nebula_radio_axis_compare(cubename, str_zap="", line="OIII", npixels=20, nsigma=1.0):
    # --------------------------------------------------------
    # Load quasar info
    # --------------------------------------------------------
    path_qso = '../../MUSEQuBES+CUBS/gal_info/quasars.dat'
    data_qso = ascii.read(path_qso, format='fixed_width')
    data_qso = data_qso[data_qso['name'] == cubename]

    if len(data_qso) == 0:
        raise ValueError(f"Could not find {cubename} in quasars.dat")

    ra_qso = float(data_qso['ra_GAIA'][0])
    dec_qso = float(data_qso['dec_GAIA'][0])

    # --------------------------------------------------------
    # Load emission map
    # --------------------------------------------------------
    UseSeg = (1.5, 'gauss', 1.5, 'gauss')
    path_map = ('../../MUSEQuBES+CUBS/fit_kin/''{}_ESO-DEEP{}_subtracted_{}_SB_3DSeg_{}_{}_{}_{}.fits'
                ).format(cubename, str_zap, line, *UseSeg)
    image = fits.open(path_map)[1].data
    header =  fits.open(path_map)[1].header
    wcs = WCS(header).celestial

    # --------------------------------------------------------
    # Mask emission map
    # --------------------------------------------------------
    i = np.where(allType[:, 0] == cubename)[0][0]
    if line == 'OII':
        nums_seg = allType[i, 3]
        select_seg = allType[i, 4]
    else:
        nums_seg = allType[i, 5]
        select_seg = allType[i, 6]

    path_3Dseg = '../../MUSEQuBES+CUBS/SB/{}_ESO-DEEP{}_subtracted_{}_3DSeg_{}_{}_{}_{}.fits'. \
        format(cubename, str_zap, line, *UseSeg)

    seg_line = fits.open(path_3Dseg)[1].data
    if select_seg:
        nums_seg = np.setdiff1d(np.arange(1, np.max(seg_line) + 1), nums_seg)
    seg_line = np.where(~np.isin(seg_line, nums_seg), seg_line, 0)
    seg_line = np.where(seg_line == 0 , seg_line, 1)
    image = np.where(seg_line == 1, image, np.nan)

    # --------------------------------------------------------
    # Pixel coordinates
    # --------------------------------------------------------
    qso_xy = sky_to_pixel(ra_qso, dec_qso, wcs)
    lobe1_xy = sky_to_pixel(radio_lobe[cubename][0][0], radio_lobe[cubename][0][1], wcs)
    lobe2_xy = sky_to_pixel(radio_lobe[cubename][1][0], radio_lobe[cubename][1][1], wcs)

    # --------------------------------------------------------
    # Calculate PA with qso at center using second moments
    # --------------------------------------------------------
    results = quasar_centered_morphology_pa(image, qso_xy)
    major_pa_deg = results['major_pa_deg']
    minor_pa_deg = results['minor_pa_deg']

    # --------------------------------------------------------
    # Radio lobe position angles from quasar
    # --------------------------------------------------------
    lobe1_pa_deg = pa_from_east_ccw(qso_xy, lobe1_xy)
    lobe2_pa_deg = pa_from_east_ccw(qso_xy, lobe2_xy)

    # --------------------------------------------------------
    # Compare lobe directions to nebular major/minor axes
    # For axes, compare modulo 180.
    # --------------------------------------------------------
    d_lobe1_major = angle_diff_180(lobe1_pa_deg, major_pa_deg)
    d_lobe2_major = angle_diff_180(lobe2_pa_deg, major_pa_deg)

    d_lobe1_minor = angle_diff_180(lobe1_pa_deg, minor_pa_deg)
    d_lobe2_minor = angle_diff_180(lobe2_pa_deg, minor_pa_deg)

    results = {"cubename": cubename,
               "path_map": path_map,
               "qso_xy": qso_xy,
               "lobe1_xy": lobe1_xy,
               "lobe2_xy": lobe2_xy,
               "major_pa_deg": major_pa_deg,
               "minor_pa_deg": minor_pa_deg,
               "lobe1_pa_deg": lobe1_pa_deg,
               "lobe2_pa_deg": lobe2_pa_deg,
               "d_lobe1_major_deg": d_lobe1_major,
               "d_lobe2_major_deg": d_lobe2_major,
               "d_lobe1_minor_deg": d_lobe1_minor,
               "d_lobe2_minor_deg": d_lobe2_minor,
               }


    plot_nebula_axes(image=image,
                     qso_xy=results["qso_xy"],
                     major_pa_deg=results["major_pa_deg"],
                     minor_pa_deg=results["minor_pa_deg"],
                     lobe1_xy=results["lobe1_xy"],
                     lobe2_xy=results["lobe2_xy"],
                     length=40,
                     )

    return results

def plot_nebula_axes(image, qso_xy, major_pa_deg, minor_pa_deg, lobe1_xy, lobe2_xy, length=30):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(image, origin='lower', cmap='gist_heat_r', vmin=-0.05, vmax=5)
    # ax[1].imshow(segmentation, origin='lower', cmap='gist_heat_r')

    # QSO position and lobe positions
    x0, y0 = qso_xy
    ax[0].plot(x0, y0, marker='*', ms=12, color='white', mec='k')
    ax[0].plot(lobe1_xy[0], lobe1_xy[1], 'ro', ms=7)
    ax[0].plot(lobe2_xy[0], lobe2_xy[1], 'ro', ms=7)

    # major axis
    th = np.radians(major_pa_deg)
    dx = length * np.cos(th)
    dy = length * np.sin(th)
    ax[0].plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], color='blue', lw=2, label='Major axis')

    # minor axis
    th = np.radians(minor_pa_deg)
    dx = length * np.cos(th)
    dy = length * np.sin(th)
    ax[0].plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], color='cyan', lw=2, ls='--', label='Minor axis')

    # radio lobe directions
    ax[0].plot([x0, lobe1_xy[0]], [y0, lobe1_xy[1]], color='red', ls='--', lw=1.5)
    ax[0].plot([x0, lobe2_xy[0]], [y0, lobe2_xy[1]], color='red', ls='--', lw=1.5)

    ax[0].legend(frameon=False)
    plt.tight_layout()
    plt.show()

res = nebula_radio_axis_compare(cubename="PKS0232-04",
                                str_zap="",
                                line="OII",
                                npixels=20,
                                nsigma=2.0,
                                )


for k, v in res.items():
    print(k, v)


# L = np.array([["PKS0405-123",     106, 129, [5, 7, 10, 11, 13, 16, 17, 20], False, [15], False]], dtype=object)
# S = np.array([["3C57",            151,  71, [2], False, [], False],
#               ["J0110-1648",       91,  29, [1], False, [2], False]], dtype=object)
# A = np.array([["Q0107-0235",      136,  90, [1, 4, 5, 6], True, [], False],
#               ["PKS2242-498",     147,  71, [1, 2], True, [], False],
#               ["PKS0232-04",      1