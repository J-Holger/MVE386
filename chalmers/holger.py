import h5py
import hdf5plugin
import numpy as np


def load_data(file_path_integrated, file_path_master, max_deg, min_deg, background_threshold=3):
    """
    Load the azimuthal integrated data and returns a dictionary with the keys that are explained below

    Arguments:
     file_path_integrated {str} -- The path to the file containing the integrated data.
     file_path_master {str} -- The path to the file containing the master data.
     max_deg {int} -- The maximum azimuthal degree to load.
     min_deg {int} -- The minimum azimuthal degree to load.

    keys in file that will be loaded
        **FROM AZINT FILE**
        'I' : cake plot from radial integration, ordering (image,azimuthal,q)
        'azi', : azimuthal bins
        'mask_file', : file path for mask used for radial integration
        'norm', : weights/norm sum for computing averages for integrated data, (azimuthal and q)
        'polarization_factor', : polarisation factor used for integration
        'poni', : file path for pony file
        'q', : q vector for integration
        **FROM MASTER FILE**
        'i_t', : diode data, transmittance for 2D map
        'title' :  scan command from SPOCK
        **CONSTRUCTED DATA**
        'shape', : shape of sample pixels, i.e. how to shape the image vector to represent the scan.
        'background_pixels' : a vector containing the indices of the background pixels.
        'background_mean' : a float representing the mean of all the background pixel rows.
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
            # We handle the integrated data separately.
            if key == 'I':
                continue
            elif name in fh:
                data[key] = fh[name][()]

        # First we load the indices of the azimuthal degrees of interest. This step is probably not relevant for the
        #   real implementation. It is used now to reduce amount of data loaded.
        azimuthal_indices = np.asarray((data['azi'] < max_deg) & (data['azi'] > min_deg)).nonzero()[0]

        # Throw away all values for norm that is outside specified angles.
        data['norm'] = data['norm'][azimuthal_indices, :]

        # Find all elements of the norm vector that are non-zero.
        non_zero_q = np.nonzero(np.sum(data['norm'], axis=0))[0]

        # Throw away all data that are outside desired angles and zero q values.
        data['norm'] = data['norm'][:, non_zero_q]
        data['q'] = data['q'][non_zero_q]
        data['azi'] = data['azi'][azimuthal_indices]
        data['I'] = fh[items['I']][()][:, :, non_zero_q]
        data['I'] = data['I'][:, azimuthal_indices, :]

        # Perform the radial (azimuthal) integration and save that as the I matrix.
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

    # Now we find the indices of the pixels that only contain the background plastic film. As a first step we calculate
    #   the mean of the middle row of pixels which we know are not a background row.
    num_rows = shape[0]
    num_cols = shape[1]
    middle_row_first_col = int(shape[0] / 2) * num_cols
    middle_row_mean = np.mean(data['I'][middle_row_first_col:middle_row_first_col+num_cols, :], axis=(0, 1))

    # Now we iterate each row and if the row's mean differ from the middle row by more than the threshold value it is
    #   considered a background row.
    background_pixels = np.empty(0, dtype=np.int8)
    background_mean = 0
    number_of_background_rows = 0
    for row in range(num_rows):
        row_mean = np.mean(data['I'][row*num_cols:row*num_cols+num_cols, :], axis=(0, 1))
        if np.abs(middle_row_mean - row_mean) > background_threshold:
            background_pixels = np.append(background_pixels, np.array([i + (row * num_cols) for i in range(num_cols)]), axis=0)
            number_of_background_rows += 1
            background_mean += row_mean
    data['background_pixels'] = background_pixels
    data['background_mean'] = background_mean / number_of_background_rows

    return data
