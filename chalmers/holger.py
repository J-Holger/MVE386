import h5py
import hdf5plugin
import numpy as np
from scipy import sparse


def load_data(file_path_integrated, file_path_master, max_deg, min_deg):
    """
    Load the azimuthal integrated data and returns a dictionary with the keys that are explained below

    Arguments:
     - file_path

    keys in file that will be loaded
        **FROM AZINT FILE**
        'I' : cake plot from radial integration, ordering (image,azimuthal,q)
        'azi', : azimuthal bins
        'mask_file', : file path for mask used for radial integration
        'norm', : weights/norm sum for computing averages for integrated data, (azimuthal and q)
        'polarization_factor', : polarisation factor used for integration
        'poni_file', : file path for pony file
        'q', : q Vektor for integration
        **FROM MASTER FILE**
        'i_t', : diode data, transmittance for 2D map
        'dt' : exposure time from eiger/lambda/diode
        'title' :  scan command from SPOCK
        'swaxs_x' : swaxs_x stage position (encoder reading)
        'swaxs_y' : swaxs_y stage position (theoretical reading)
        'swaxs_rot' : swaxs_y stage position (theoretical reading)
        'time' : time point for triggers for exposure
    """
    # Load integrated data
    data = {}
    items = {
        'I': 'entry/data2d/cake',
        'q': 'entry/data1d/q',
        'azi': 'entry/data2d/azi',
        'mask_file': 'entry/azint/input/mask_file',
        'norm': 'entry/data2d/norm',
        'polarization_factor': 'entry/azint/input/polarization_factor',
        'poni': 'entry/azint/input/poni',
    }
    with h5py.File(file_path_integrated, 'r') as fh:
        for key, name in items.items():
            if key == 'I':
                continue
            elif name in fh:
                data[key] = fh[name][()]

        # We handle the integrated data separately. Since it is only interesting to look at data points where the sum of
        #   all norm q-values for each angle are non-zero, we extract only these indices.
        azimuthal_indices = np.asarray((data['azi'] < max_deg) & (data['azi'] > min_deg)).nonzero()[0]
        data['norm'] = data['norm'][azimuthal_indices, :]
        non_zero_q = np.nonzero(np.sum(data['norm'], axis=0))[0]
        data['norm'] = data['norm'][:, non_zero_q]
        data['q'] = data['q'][non_zero_q]
        data['azi'] = data['azi'][azimuthal_indices]
        data['I'] = fh[items['I']][()][:, :, non_zero_q]
        data['I'] = data['I'][:, azimuthal_indices, :]
        data['I'] = np.sum(data['I'] * data['norm'][...], axis=1) / np.sum(data['norm'], axis=0)[...]

    # Master file
    items = {
        'i_t': 'entry/instrument/albaem-e01_ch1/data',
        'title': 'entry/title',
        'dt': 'entry/exposure/time'
    }
    with h5py.File(file_path_master, 'r') as fh:
        for key, name in items.items():
            if name in fh:
                data[key] = fh[name][()]

    # The command used has information on the snake scan performed, we use it.
    shape = (int((str(data['title']).split(' '))[8]) + 1, int((str(data['title']).split(' '))[4]))
    data['shape'] = shape

    num_rows = shape[0]
    num_cols = shape[1]
    middle_row_first_col = int(shape[0] / 2) * num_cols
    middle_row_mean = np.mean(data['I'][middle_row_first_col:middle_row_first_col+num_cols, :], axis=(0, 1))
    background_pixels = np.empty(0)
    background_mean = 0
    number_of_background_rows = 0
    for row in range(num_rows):
        row_mean = np.mean(data['I'][row*num_cols:row*num_cols+num_cols, :], axis=(0, 1))
        if middle_row_mean - row_mean > 3:
            background_pixels = np.append(background_pixels, np.array([i + (row * num_cols) for i in range(num_cols)]), axis=0)
            number_of_background_rows += 1
            background_mean += row_mean
    data['background_pixels'] = background_pixels
    data['background_mean'] = background_mean / number_of_background_rows

    return data
