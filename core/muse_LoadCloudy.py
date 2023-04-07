import numpy as np


# Load lineratio
def load_cloudy(filename=None, path=None):
    # Line profile
    line = np.genfromtxt(path + filename + '.lin', delimiter=None)
    NeV3346, OII3727, OII3730 = line[:, 2], line[:, 3], line[:, 4]
    NeIII3869, Hdel, Hgam = line[:, 5], line[:, 8], line[:, 9]
    OIII4364, HeII4687, OIII5008 = line[:, 10], line[:, 11], line[:, 13]
    data = np.vstack((NeV3346, OII3727, OII3730, OII3727 + OII3730, NeIII3869, Hdel, Hgam, OIII4364, HeII4687, OIII5008))
    return np.log10(data)


def load_cloudy_nogrid(filename=None, path=None):
    # Line profile
    line = np.genfromtxt(path + filename + '.lin', delimiter=None)
    NeV3346, OII3727, OII3730 = line[2], line[3], line[4]
    NeIII3869, Hdel, Hgam = line[5], line[8], line[9]
    OIII4364, HeII4687, OIII5008 = line[10], line[11], line[13]
    data = np.vstack((NeV3346, OII3727, OII3730, OII3727 + OII3730, NeIII3869, Hdel, Hgam, OIII4364, HeII4687, OIII5008))
    return np.log10(data)


def format_cloudy(filename=None, path=None):
    for i in range(len(filename[0])):
        metal_i = filename[0][i]
        for j in range(len(filename[1])):
            alpha_j = filename[1][j]
            filename_ij = 'alpha_' + str(alpha_j) + '_' + str(metal_i)
            if j == 0:
                output_j = load_cloudy(filename_ij, path=path)
                ind_j = np.array([[alpha_j, metal_i]])
            else:
                ind_jj = np.array([[alpha_j, metal_i]])
                c_i = load_cloudy(filename_ij, path=path)
                output_j = np.dstack((output_j, c_i))
                ind_j = np.dstack((ind_j, ind_jj))
        if i == 0:
            ind = ind_j[:, :, :, np.newaxis]
            output = output_j[:, :, :, np.newaxis]
        else:
            output = np.concatenate((output, output_j[:, :, :, np.newaxis]), axis=3)
            ind =  np.concatenate((ind, ind_j[:, :, :, np.newaxis]), axis=3)
    return output, ind


def format_cloudy_nogrid(filename=None, path=None):
    for k in range(len(filename[2])):
        metal_k = filename[2][k]
        for j in range(len(filename[1])):
            alpha_j = filename[1][j]
            for i in range(len(filename[0])):
                density_i = filename[0][i]
                filename_jki = 'alpha_' + str(alpha_j) + '_' + str(metal_k) + '_' + str(density_i)
                if i == 0:
                    output_i = load_cloudy_nogrid(filename_jki, path=path)[:, 0]
                else:
                    c_i = load_cloudy_nogrid(filename_jki, path=path)[:, 0]
                    output_i = np.vstack((output_i, c_i))
            if j == 0:
                output_j = output_i.T
            else:
                output_j = np.dstack((output_j, output_i.T))
        if k == 0:
            output = output_j[:, :, :, np.newaxis]
        else:
            output = np.concatenate((output, output_j[:, :, :, np.newaxis]), axis=3)
    return output


def format_cloudy_nogrid_BB(filename=None, path=None):
    for k in range(len(filename[2])):
        metal_k = filename[2][k]
        for j in range(len(filename[1])):
            alpha_j = filename[1][j]
            for i in range(len(filename[0])):
                density_i = filename[0][i]
                filename_jki = 'T_' + str(alpha_j) + '_Z_' + str(metal_k) + '_n_' + str(density_i)
                if i == 0:
                    output_i = load_cloudy_nogrid(filename_jki, path=path)[:, 0]
                else:
                    c_i = load_cloudy_nogrid(filename_jki, path=path)[:, 0]
                    output_i = np.vstack((output_i, c_i))
            if j == 0:
                output_j = output_i.T
            else:
                output_j = np.dstack((output_j, output_i.T))
        if k == 0:
            output = output_j[:, :, :, np.newaxis]
        else:
            output = np.concatenate((output, output_j[:, :, :, np.newaxis]), axis=3)
    return output


def format_cloudy_nogrid_AGN(filename=None, path=None):
    for i in range(len(filename[5])):
        x_i = filename[5][i]
        for j in range(len(filename[4])):
            uv_j = filename[4][j]
            for k in range(len(filename[3])):
                ox_k = filename[3][k]
                for ii in range(len(filename[2])):
                    T_ii = filename[2][ii]
                    for jj in range(len(filename[1])):
                        metal_jj = filename[1][jj]
                        for kk in range(len(filename[0])):
                            density_kk = filename[0][kk]
                            filename_jki = 'ox_' + str(ox_k) + 'uv_' + str(uv_j) + 'x_' + str(x_i) + 'T_' + str(T_ii) \
                                           + 'Z_' + str(metal_jj) + 'n_' + str(density_kk)
                            if kk == 0:
                                output_kk = load_cloudy_nogrid(filename_jki, path=path)[:, 0]
                            else:
                                c_kk = load_cloudy_nogrid(filename_jki, path=path)[:, 0]
                                output_kk = np.vstack((output_kk, c_kk))
                        if jj == 0:
                            output_jj = output_kk.T
                        else:
                            output_jj = np.dstack((output_jj, output_kk.T))
                    if ii == 0:
                        output_ii = output_jj[:, :, :, np.newaxis]
                    else:
                        output_ii = np.concatenate((output_ii, output_jj[:, :, :, np.newaxis]), axis=3)
                if k == 0:
                    output_k = output_ii[:, :, :, :, np.newaxis]
                else:
                    output_k = np.concatenate((output_k, output_ii[:, :, :, :, np.newaxis]), axis=4)
            if j == 0:
                output_j = output_k[:, :, :, :, :, np.newaxis]
            else:
                output_j = np.concatenate((output_j, output_k[:, :, :, :, :, np.newaxis]), axis=5)
        if i == 0:
            output_i = output_j[:, :, :, :, :, :, np.newaxis]
        else:
            output_i = np.concatenate((output_i, output_j[:, :, :, :, :, :, np.newaxis]), axis=6)
        print(np.shape(output_i))
    return output_i