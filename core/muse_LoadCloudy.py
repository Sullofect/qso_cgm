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