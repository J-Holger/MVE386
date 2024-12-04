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
    data['shape'] = (int((str(data['title']).split(' '))[8]) + 1, int((str(data['title']).split(' '))[4]))

    return data
