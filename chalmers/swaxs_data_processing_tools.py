import h5py
import os
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import matplotlib.ticker as mticker
from ipywidgets import interact, widgets
from matplotlib import colors as mcolors
from matplotlib.ticker import FuncFormatter
from scipy import fft
from scipy import signal
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import norm
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"

def calculate_y_limits(data, padding=0.1):
    """
    Calculate y-axis limits with padding for a given dataset.

    Parameters
    ----------
    data : np.ndarray
        The intensity data to calculate limits for.
    padding : float, optional
        Fractional padding to add to the min and max values (default: 0.1).

    Returns
    -------
    tuple
        Tuple of (y_min, y_max) with applied padding.
    """
    y_min = data.min() - padding * abs(data.min())
    y_max = data.max() + padding * abs(data.max())
    return y_min, y_max

def q_to_d(q):
    """Convert q (Å⁻¹) to d (Å), excluding zero or near-zero values."""
    q_safe = np.clip(q, 1e-6, None)  # Replace q <= 0 with a small positive value
    return 2 * np.pi / q_safe
    
def d_to_q(d):
    """Convert d (Å) to q (Å⁻¹), excluding zero or near-zero values."""
    d_safe = np.clip(d, 1e-6, None)
    return 2 * np.pi / d_safe

def select_sample():
    """
    Provides a widget interface for selecting a sample and scan number from the logbook CSV file.

    Returns
    -------
    tuple
        Selected sample name and scan number.
    """
    # Load the CSV file
    csv_path = "logbook_227.csv"
    logbook = pd.read_csv(csv_path, delimiter=";", skipinitialspace=True)

    # Clean up unnecessary columns
    logbook = logbook.drop(columns=[col for col in logbook.columns if "Unnamed" in col or "Name nomenclature" in col], errors='ignore')

    selected_sample = widgets.Output()
    selected_scan = widgets.Output()

    # Define sample selection widget
    def update_sample(sample_name):
        sample_rows = logbook[logbook["Name"] == sample_name]
        with selected_sample:
            selected_sample.clear_output()
            if not sample_rows.empty:
                display(sample_rows)

        # Update scan number dropdown options
        scan_dropdown.options = sample_rows["ScanNbr"].tolist()

    sample_dropdown = widgets.Dropdown(
        options=logbook["Name"].unique(),
        description="Sample:"
    )
    sample_dropdown.observe(lambda change: update_sample(change.new), names="value")

    # Define scan number selection widget
    def update_scan(scan_number):
        global scan_num
        scan_num = scan_number
        with selected_scan:
            selected_scan.clear_output()
            print(f"Selected Scan Number: {scan_number}")

    scan_dropdown = widgets.Dropdown(
        options=[],
        description="ScanNbr:"
    )
    scan_dropdown.observe(lambda change: update_scan(change.new), names="value")

    # Display widgets
    display(sample_dropdown)
    display(selected_sample)
    display(scan_dropdown)
    display(selected_scan)

    # Return the selected values
    return sample_dropdown, scan_dropdown

def construct_file_path(scan_number, method, raw=False, proposal=20240661, visit=2024102408):
    """
    Constructs the file path for SAXS or WAXS data files.

    Parameters
    ----------
    scan_number : int
        The scan number identifying the specific sample.
    method : str
        Scattering method used ('SAXS' or 'WAXS').
    raw : bool, optional
        If True, returns the path to the raw data file in Cartesian coordinates.
        If False, returns the path to the pre-processed file in polar coordinates.
    proposal : int, optional
        Proposal ID for the data collection (default: 20240661).
    visit : int, optional
        Visit ID for the data collection (default: 2024102408).

    Returns
    -------
    str
        Constructed file path to the desired data file.
    """
    detector_map = {"SAXS": "eiger", "WAXS": "lambda"}
    method_upper = method.upper()

    if method_upper not in detector_map:
        raise ValueError("Invalid method. Must be 'SAXS' or 'WAXS'.")

    detector = detector_map[method_upper]
    file_type = "raw" if raw else "process/azint"
    suffix = "" if raw else "_integrated"

    return f"/data/visitors/formax/{proposal}/{visit}/{file_type}/scan-{scan_number:04d}_{detector}{suffix}.h5"

def load_saxs_raw(filepath, image_idx):
    """
    Loads raw SAXS data from an HDF5 file for a specified image.

    Parameters
    ----------
    filepath : str
        Path to the SAXS HDF5 file.
    image_idx : int
        Index of the image to extract.

    Returns
    -------
    np.ndarray
        Extracted raw SAXS data with invalid values set to NaN.
    """
    with h5py.File(filepath, "r") as file:
        data = file["/entry/instrument/eiger/data"][image_idx, :, :].astype(np.float32)
        data[data > 1e9] = np.nan  # Mask invalid values
    return data

def load_waxs_raw(filepath, image_idx):
    """
    Loads and reconstructs raw WAXS data from an HDF5 file for a specified image.

    Parameters
    ----------
    filepath : str
        Path to the WAXS HDF5 file.
    image_idx : int
        Index of the image to extract.

    Returns
    -------
    np.ndarray
        Reconstructed WAXS data as a 2D array.
    """
    with h5py.File(filepath, "r") as file:
        full_shape = file["/entry/instrument/lambda/full_shape"][()]
        rotation = file["/entry/instrument/lambda/rotation"][()]
        x_positions = file["/entry/instrument/lambda/x"][()]
        y_positions = file["/entry/instrument/lambda/y"][()]
        data = file["/entry/instrument/lambda/data"][image_idx, :, :, :]

    # Initialize the full data array with NaNs where we will store the image
    full_data = np.full(full_shape, np.nan, dtype=np.float32)

    # Reconstruct the full image from the four modules
    for i in range(4):
        # Rotate each module's data
        rotated = np.rot90(data[i, :, :], k=-rotation[i] // 90)

        # Insert rotated data into full image
        x_start, y_start = x_positions[i], y_positions[i]
        x_end, y_end = x_start + rotated.shape[1], y_start + rotated.shape[0]
        full_data[y_start:y_end, x_start:x_end] = rotated

    return full_data

def plot_raw_data(data, method, scan_num, image_idx, ax=None):
    """
    Plots raw SAXS or WAXS data.

    Parameters
    ----------
    data : np.ndarray
        Raw data array to be plotted.
    method : str
        Scattering method ('SAXS' or 'WAXS').
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on (default: None).
    image_idx : int
        Index of the image to plot.

    Returns
    -------
    matplotlib.image.AxesImage
        Image object created by the plot.
    """
    if ax is None:
        ax = plt.gca()

    # Set up plotting parameters 
    norm = mcolors.SymLogNorm(linthresh=1e-3, vmin=1e-1, vmax=1e2) if method.upper() == "SAXS" else mcolors.LogNorm()
    origin = "lower" if method.upper() == "SAXS" else "upper"

    
    im = ax.imshow(data, cmap="viridis", norm=norm, origin=origin)
    ax.set_title(f"{method.upper()} Raw Data (Scan number {scan_num}, Image {image_idx})")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")
    plt.colorbar(im, ax=ax, label="Intensity")
    return im

def load_transformed_data(scan_number, method):
    """
    Load the transformed data and return a dictionary with the relevant keys.

    Parameters
    ----------
    scan_number : int
        The scan number identifying the specific sample.
    method : str
        Scattering method used ('SAXS' or 'WAXS').

    Returns
    -------
    data : dict
        A dictionary containing the loaded data.

    Keys in the returned data dictionary:

    **From Integrated (Azimuthally Integrated) File**
    - 'I' : Cake plot from radial integration, shape (image, azimuthal angle, q)
    - 'azi' : Azimuthal bins (degrees)
    - 'q' : q-vector for integration (Å⁻¹)
    - 'mask_file' : File path for mask used in radial integration
    - 'norm' : Weights/norm sum for computing averages for integrated data
    - 'polarization_factor' : Polarization factor used for integration
    - 'poni_file' : File path for PONI file

    **From Raw (Master) File**
    - 'i_t' : Diode data, transmittance for 2D map
    - 'title' : Scan command from the instrument control software
    - 'shape' : Tuple indicating the shape of the scan (rows, columns)
    """
    detector_map = {"SAXS": "eiger", "WAXS": "lambda"}
    method_upper = method.upper()

    if method_upper not in detector_map:
        raise ValueError("Invalid method. Must be 'SAXS' or 'WAXS'.")

    detector = detector_map[method_upper]

    # Get the file path to the integrated data file
    file_path = construct_file_path(scan_number, method, raw=False)

    # Set up dictionary
    data = {}
    items = {
        'I': 'entry/data2d/cake',
        'q': 'entry/data1d/q',
        'azi': 'entry/data2d/azi',
        'mask_file': 'entry/azint/input/mask_file',
        'norm': 'entry/data2d/norm',
        'polarization_factor': 'entry/azint/input/polarization_factor',
        'poni_file': 'entry/azint/input/poni',
    }

    # Load data from the transformed file
    with h5py.File(file_path, 'r') as fh:
        for key, name in items.items():
            # We handle the intensity data separately
            if name in fh:
                data[key] = fh[name][()]
            else:
                print(f"Warning: {name} not found in integrated file.")

    # Find all elements of the norm vector that are non-zero, these also correspond to non-zero q values.
    non_zero_q = np.nonzero(np.sum(data['norm'], axis=0))[0]
    
    # Throw away all data that has zero q values.
    data['norm'] = data['norm'][:, non_zero_q]
    data['q'] = data['q'][non_zero_q]
    data['I'] = data['I'][:, :, non_zero_q]

    # Construct the master file path by removing '_integrated' in the filename
    master_file_path = file_path.replace('process/azint', 'raw')
    master_file_path = master_file_path.replace('_%s_integrated' %detector, '')
    master_items = {
        'i_t': 'entry/instrument/albaem-e01_ch1/data',
        'title': 'entry/title',
    }

    # Load data from the master (raw) file
    if os.path.isfile(master_file_path):
        with h5py.File(master_file_path, 'r') as fh:
            for key, name in master_items.items():
                if name in fh:
                    data[key] = fh[name][()]
                else:
                    print(f"Warning: {name} not found in master file.")
    else:
        print(f"Master file {master_file_path} not found.")
    
    # Parse the title for scan shape information
    title_parts = str(data['title']).split(' ')
    rows = int(title_parts[8]) + 1
    columns = int(title_parts[4])
    data['shape'] = (rows, columns)

    return data

def plot_transformed_data(data, scan_num, image_idx, method):
    """
    Plots either SAXS or WAXS transformed data.

    Parameters
    ----------
    data : dict
        A dictionary containing the transformed data, typically returned by 'load_transformed_data'.
        Expected keys are 'I', 'azi', and 'q'.
    image : int, optional
        Index of the image to plot if multiple images are present in 'I'. Default is 0.
    method : str
        Scattering method used (SAXS or WAXS).

    Returns
    -------
    None
    """

    # Extract data from the dictionary
    I = data.get('I')
    azi = data.get('azi')
    q = data.get('q')

    if I is None or azi is None or q is None:
        print("Required data for plotting is missing.")
        return

    # Convert azimuthal angles from degrees to radians
    azi_radians = np.radians(azi)

    # Create meshgrids for azimuthal angles and q values
    azi_grid, q_grid = np.meshgrid(azi_radians, q)

    # Extract specified image from I, transpose to match meshgrid shape
    I_plot = I[image_idx, :, :].T
    
    # Convert specified scattering method to uppercase
    method_upper = method.upper()
    
    # Set colormap
    cmap = plt.cm.viridis

    if method_upper == 'SAXS':
        # Specify logscale
        norm = mcolors.LogNorm( vmin=1e-1, vmax=1e2)
        
        fig = plt.figure(figsize=(12, 6))
        
        # First subplot of intensity data
        ax1 = fig.add_subplot(121, polar=True)
        im = ax1.pcolormesh(azi_grid, q_grid, I_plot, cmap=cmap, norm=norm)
        
        # Customize the plot
        ax1.set_yticklabels([])
        plt.colorbar(im, ax=ax1, label='Intensity')
        ax1.set_title(r'SAXS Data (Image {}): $I(\phi, q)$'.format(image_idx))
        
        # Select data for zoomed-in plot
        max_q = 250
        I_zoom = I_plot[:max_q, :]
        q_zoom = q[:max_q]
        azi_grid_zoom, q_grid_zoom = np.meshgrid(azi_radians, q_zoom)
        
        # Second subplot of intensity data over a narrower q-range
        ax2 = fig.add_subplot(122, polar=True)
        im = ax2.pcolormesh(azi_grid_zoom, q_grid_zoom, I_zoom, cmap=cmap, norm=norm)
        
        # Customize the plot
        ax2.set_yticklabels([])
        plt.colorbar(im, ax=ax2, label='Intensity')
        ax2.set_title(r'SAXS Data (zoomed in) $I(\phi, q)$')
        
    elif method_upper == 'WAXS':
        # Specify logscale 
        norm = mcolors.LogNorm()
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, polar=True)

        # Plot the intensity data
        im = ax.pcolormesh(azi_grid, q_grid, I_plot, cmap=cmap, norm=norm)

        # Customize the plot
        plt.colorbar(im, ax=ax, label='Intensity')
        ax.set_yticklabels([])
        ax.set_title(r'SAXS Data (Image {}): $I(\phi, q)$'.format(image_idx))
    else:
        raise ValueError("Invalid scattering method.")
    plt.show()

def reshape_I(data):
    """ Reshape the intensity function to have its first two coordinate being the row and the column
    Args:
        data (_type_): The data to use. 

    With I(img_number, azi, q) as input, the output is I(row, column, azi, q)
    """
    I = data['I'].reshape((data['shape'][0],data['shape'][1],data['azi'].shape[0],data['q'].shape[0]))   
    i_t = data["i_t"].reshape((data['shape'][0],data['shape'][1]))  #Also reshape the transmission data 
    I_flipped = np.copy(I)
    i_t_flipped = np.copy(i_t)
    #Since it is a snake scan, we need to flip every other row.
    I_flipped[1::2,:,:,:] = I_flipped[1::2,::-1,:,:]
    i_t_flipped[1::2,:] = i_t_flipped[1::2,::-1]
    data["I"] = I_flipped
    data["i_t"] = i_t_flipped
    return data

def azimuthal_integration(data, azi_min=0, azi_max=360):
    """
    Integrate (average) the intensity between the specified minimum azimuthal angle and the maximum one
    for all q. Handles azimuthal wrapping for ranges like 340° to 20°.

    Parameters
    ----------
    data : dict
        The data dictionary containing the keys 'I', 'q', 'azi', and 'norm'.
        - 'I' : Intensity data, shape (images, azimuthal angles, q).
        - 'q' : Radial q-vector for the integration (Å⁻¹).
        - 'azi' : Azimuthal angles in degrees (0 to 360).
        - 'norm' : Normalization weights for computing averages.

    azi_min : float, optional
        The minimum azimuthal angle for the integration range in degrees (default is 0°).
    azi_max : float, optional
        The maximum azimuthal angle for the integration range in degrees (default is 360°).

    Returns
    -------
    I_azimuthal_integrated : ndarray
        The intensity integrated over the specified azimuthal range for all q.

        - If `data['I']` has 3 dimensions, the result has shape (images, q).
        - If `data['I']` has 4 dimensions, the result has shape (rows, columns, q).

    Notes
    -----
    - Handles wraparound cases where azi_min > azi_max (e.g., 340° to 20°).
    - Uses safe division to avoid division by zero when normalization weights are zero.
    """
    I = data["I"]
    q = data["q"]
    azi = data["azi"]
    w = data["norm"]
    trans = data["i_t"]

    if azi_min > azi_max:  # Handle wraparound
        # Split into two ranges: [azi_min, 360°] and [0°, azi_max]
        idx_range1 = np.where(azi >= azi_min)[0]
        idx_range2 = np.where(azi < azi_max)[0]
        idx_range = np.concatenate([idx_range1, idx_range2])
    else:
        # Regular case
        idx_range = np.where((azi >= azi_min) & (azi < azi_max))[0]

    #Filter the data to includes only the relevant azimuthal angles
    I_azi = I[...,idx_range,:]
    w_azi = w[idx_range]


    # Sum of the weights across all considered azi angles
    norm_sum = np.sum(w_azi, axis=0)  

    # Mask norm_sum where it's zero
    norm_sum_safe = np.where(norm_sum == 0, np.nan, norm_sum)  # Replace zeros with NaN

    ndim = len(np.shape(I))  # So that it works on either original data, or reshaped
    if ndim == 3:
        I_times_norm = I_azi * w_azi[None, :, :]  # Weighted measurements
        I_azi_integrated = np.sum(I_times_norm, axis=1) / norm_sum_safe[None, :]
        I_azi_integrated = I_azi_integrated / trans[:,None]
    elif ndim == 4:
        I_times_norm = I_azi * w_azi[None, None, :, :]  # Weighted measurements
        I_azi_integrated = np.sum(I_times_norm, axis=2) / norm_sum_safe[None, None, :]
        I_azi_integrated = I_azi_integrated / trans[:,:,None]
    else:
        raise ValueError("Incompatible number of dimensions for the intensity")

    # Optionally replace NaN with zeros or another default value
    #I_azi_integrated = np.nan_to_num(I_azi_integrated, nan=0.0)

    return I_azi_integrated

def get_background_noise(I_data, method, shape):
    """
    Identifies background noise rows and calculates the average background noise.

    Background rows are identified by comparing the mean intensity of each row to the mean intensity
    of the middle row. Rows diverging beyond a predefined threshold (based on the scattering method)
    are considered background rows.

    Parameters
    ----------
    I_data : ndarray
        Azimuthally integrated intensity data.
        - If reshaped: Shape is (rows, cols, q), where `rows` and `cols` represent the sample layout.
        - If not reshaped: Shape is (image_idx, q), where `image_idx` follows a snake scan order.
    method : str
        Scattering method used, either 'SAXS' or 'WAXS'. Determines the background threshold:
        - 'SAXS': Threshold is 1500.
        - 'WAXS': Threshold is 9000.
        Note: these should be changed if the function does not behave as expected.
    shape : tuple or str
        Layout of the sample:
        - If "reshaped", the input `I_data` is assumed to be reshaped (rows, cols, q).
        - If tuple, specifies the shape of the snake scan as (rows, cols).

    Returns
    -------
    background_noise : ndarray
        Vector representing the mean background noise for each `q` value (shape: `(q,)`).
    """
    
    if method == 'SAXS':
        background_threshold = 1500
    elif method == 'WAXS':
        background_threshold = 9000
    else:
        print('Invalid scattering method')

    background_rows = []
    background_pixels = []
    which_rows = []
    
    if shape == "reshaped":
        num_rows, num_cols = I_data.shape[:2]
        middle_row_mean = np.nanmean(I_data[30], axis=(0, 1))
        
        for i in range(num_rows):
            row_mean = np.nanmean(I_data[i], axis=(0, 1))
            if (middle_row_mean - row_mean) > background_threshold:
                which_rows.append(i)
                q_wise_row_mean = np.nanmean(I_data[i], axis=0)
                background_rows.append(q_wise_row_mean)
                background_pixels.extend([(i, j) for j in range(num_cols)])
    else:
        num_rows, num_cols = shape
        middle_row_start = (num_rows // 2) * num_cols
        middle_row_end = middle_row_start + num_cols
        middle_row_mean = np.nanmean(I_data[middle_row_start:middle_row_end], axis=(0, 1))

        for i in range(num_rows):
            row_start = i * num_cols
            row_end = row_start + num_cols
            row_mean = np.nanmean(I_data[row_start:row_end], axis=(0, 1))
            if (middle_row_mean - row_mean) > background_threshold:
                which_rows.append(i)
                q_wise_row_mean = np.nanmean(I_data[i * num_cols : i * num_cols + num_cols], axis=0)
                background_rows.append(q_wise_row_mean)
                background_pixels.extend([(i, j) for j in range(num_cols)])

    if background_rows:
        background_noise = np.nanmean(background_rows, axis=0)
    else:
        raise ValueError("No background rows identified based on the threshold.")

    return background_noise, background_pixels

def plot_azimuthal_integrated_data(q_data, I_data, directions, method, row_range, col_range):
    """
    Plot azimuthal integrated intensity for all specified directions for SAXS and WAXS,
    with an additional x-axis for d = 2pi/q.

    Parameters
    ----------
    q_data : numpy.ndarray
        q-values for data.
    I_data : dict
        Dictionary containing radially integrated intensity the data, keyed by direction.
    directions : list
        List of directions to plot (e.g., ['east', 'north', 'west', 'south', 'full']).
    method : str
        Scattering method used (SAXS or WAXS).
    row_range : tuple or int, optional
        If a tuple, specifies the range of rows to use (start_row, end_row).
        If an int, specifies a single row. Default is None.
    col_range : tuple or int, optional
        If a tuple, specifies the range of columns to use (start_col, end_col).
        If an int, specifies a single column.
    """
    # Set up for slicing
    if isinstance(row_range, int):
        row_start, row_end = row_range, row_range + 1
    else:
        row_start, row_end = row_range
    
    if isinstance(col_range, int):
        col_start, col_end = col_range, col_range + 1
    else:
        col_start, col_end = col_range
        
    # Convert specified scattering method to uppercase
    method_upper = method.upper()

    if method_upper == 'SAXS':
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        for direction in directions:
            ax.plot(
                q_data,
                np.nanmean(I_data[direction][row_start:row_end, col_start:col_end, :], axis=(0, 1)) * q_data ** 2,
                label=direction.capitalize()
            )
        
        ax.set_title(f"SAXS Azi. Integrated Data")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$q \, (\mathrm{\AA}^{-1})$")
        ax.set_ylabel(r"$I \cdot q^2 \, (\mathrm{A.U.})$")
        ax.legend()

        # Add secondary x-axis for d
        secax = ax.secondary_xaxis('top', functions = (q_to_d, d_to_q))
        secax.set_xticks([100, 200, 300, 400, 500, 600, 800, 1000])
        secax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x / 100)}"))
        secax.xaxis.offsetText.set_visible(False)                                      # Hide the default offset
        secax.set_xlabel(r"$d \, (\times 10^2 \, \mathrm{\AA})$")
        
    elif method_upper == 'WAXS':
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        for direction in directions:
            ax.plot(
                q_data,
                np.nanmean(I_data[direction][20:30,0:30, :], axis=(0, 1)),
                label=direction.capitalize()
        )
        ax.set_title("WAXS Azi. Integrated Data")
        ax.set_xlabel(r"$q \, (\mathrm{\AA}^{-1})$")
        ax.set_ylabel(r"$I \, (\mathrm{A.U.})$")
        ax.legend()

        # Set y-axis to scientific notation
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.yaxis.get_offset_text().set_fontsize(10)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax.set_ylabel(r"$I \, (\mathrm{A.U.})$", fontsize=12)

        # Adjust the position of the scientific notation offset label
        ax.yaxis.offsetText.set_x(-0.07)  # Adjust X-position

        # Add secondary x-axis for d
        secax = ax.secondary_xaxis('top', functions = (q_to_d, d_to_q))
        secax.set_xlabel(r"$d \, (\mathrm{\AA})$")
        secax.set_xticks([1, 2, 3, 4, 5])

    else:
        raise ValueError("Invalid scattering method.")

    plt.show()

def calculate_symmetry_ratios(q_data, I_data, directions, ratios, method, row_range, col_range, q_range="full"):
    """
    Calculate symmetry ratios for given azimuthally integrated data and direction pairs.

    Parameters
    ----------
    q_data : numpy.ndarray
        q-values for data.
    I_data : dict
        Dictionary containing az integrated intensity data, keyed by direction.
    directions : list
        List of directions to include (e.g., ['east', 'north', 'west', 'south']).
    ratios : dict
        Dictionary specifying direction pairs for symmetry ratio calculations.
        E.g., {'ns': ('north', 'south'), 'ew': ('east', 'west')}.
    q_range : tuple or str, optional
        Tuple specifying the q-range for averaging (q_min, q_max).
        If "full", takes the entire q-range for averaging. Default is "full".
    row_range : tuple or int
        If a tuple, specifies the range of rows to use (start_row, end_row).
        If an int, specifies a single row.
    col_range : tuple or int, optional
        If a tuple, specifies the range of columns to use (start_col, end_col).
        If an int, specifies a single column.
    method : str, optional
        Scattering method ('SAXS' or 'WAXS'), used for reporting purposes. Default is 'SAXS'.

    Returns
    -------
    symmetry_ratios : dict
        Dictionary containing symmetry ratios for each pair of directions.
    """
    # Determine q-range indices
    if q_range == "full":
        q_indices = np.arange(len(q_data))
    else:
        q_min, q_max = q_range
        q_indices = np.where((q_data >= q_min) & (q_data <= q_max))[0]
    
    # Set up for slicing
    if isinstance(row_range, int):
        row_start, row_end = row_range, row_range + 1
    else:
        row_start, row_end = row_range
    
    if isinstance(col_range, int):
        col_start, col_end = col_range, col_range + 1
    else:
        col_start, col_end = col_range

    # Compute symmetry ratios
    symmetry_ratios = {}
    for ratio, (dir1, dir2) in ratios.items():
        if ratio == 'vh':
            # Special case: vertical-horizontal symmetry ratio
            numerator = (
                np.nanmean(I_data['east'][row_start:row_end, col_start:col_end, q_indices], axis=(0, 1, 2)) +
                np.nanmean(I_data['west'][row_start:row_end, col_start:col_end, q_indices], axis=(0, 1, 2))
            )
            denominator = (
                np.nanmean(I_data['north'][row_start:row_end, col_start:col_end, q_indices], axis=(0, 1, 2)) +
                np.nanmean(I_data['south'][row_start:row_end, col_start:col_end, q_indices], axis=(0, 1, 2))
            )
            symmetry_ratios[ratio] = numerator / denominator
        else:
            # General case for other symmetry ratios
            numerator = np.nanmean(I_data[dir1][row_start:row_end, col_start:col_end, q_indices], axis=(0, 1, 2))
            denominator = np.nanmean(I_data[dir2][row_start:row_end, col_start:col_end, q_indices], axis=(0, 1, 2))
            symmetry_ratios[ratio] = numerator / denominator


    # Print results
    print(f"Symmetry ratios for {method} data:")
    for ratio, value in symmetry_ratios.items():
        print(f"{ratio.upper()} symmetry ratio = {value:.6f}")

    return symmetry_ratios

def filter_and_find_peaks(q_data, intensity_data, pixels, kernel_size=11, distance=60, width=4):
    """
    Filters intensity data and identifies peaks for specified pixels.

    Parameters
    ----------
    q_data : np.ndarray
        Array of q-values (Å⁻¹).
    intensity_data : np.ndarray
        Array of intensity values for each pixel.
    pixels : list
        List of pixel indices to process.
    kernel_size : int, optional
        Kernel size for median filtering (default: 11).
    distance : int, optional
        Minimum distance between peaks for peak detection (default: 60).
    width : int, optional
        Minimum width of peaks for peak detection (default: 4).

    Returns
    -------
    dict
        Dictionary containing filtered intensities and peak indices for each pixel.
    """
    # Filter intensities using median filter
    I_filtered = [signal.medfilt(intensity_data[pixel], kernel_size) for pixel in pixels]

    # Identify peaks in filtered data
    peak_indices = [
        signal.find_peaks(filtered_intensity, distance=distance, width=width)[0]
        for filtered_intensity in I_filtered
    ]

    # Plot results
    fig, ax = plt.subplots(1, len(pixels), figsize=(15, 5))
    fig.suptitle(r'Filtered Intensities and Detected Peaks', y=1.08, fontsize=14)

    for i, pixel in enumerate(pixels):
        # Set secondary x-axis for d values
        secax = ax[i].secondary_xaxis('top', functions = (q_to_d, d_to_q))
        secax.set_xlabel(r"$d \, (\mathrm{\AA})$")
        secax.set_xticks([1, 2, 3, 4, 5])
        ax[i].set_xlabel(r"$q \, (\mathrm{\AA}^{-1})$", fontsize=12)

        # Calculate individual y-axis limits with padding
        y_min, y_max = calculate_y_limits(I_filtered[i])
        ax[i].set_ylim(y_min, y_max)

        # Set y-axis to scientific notation
        ax[i].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax[i].yaxis.get_offset_text().set_fontsize(10)
        ax[i].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax[i].set_ylabel(r"$I \, (\mathrm{A.U.})$", fontsize=12)

        # Adjust the position of the scientific notation offset label
        ax[i].yaxis.offsetText.set_x(-0.10)  # Adjust X-position

        # Plot the filtered intensity and detected peaks
        ax[i].plot(q_data, I_filtered[i], linewidth=2)
        for xc in peak_indices[i]:
            ax[i].axvline(x=q_data[xc], ymin=0, ymax=1, c='r', lw=0.5)

        # Add a subplot title
        ax[i].set_title(f'Pixel: ({pixel[0]}, {pixel[1]})', fontsize=10)

    plt.show()

    return {"filtered_intensities": I_filtered, "peak_indices": peak_indices}

def analyze_num_peaks(intensity_data, background_pixels, kernel_size=11, distance=60, width=4):
    """
    Analyzes the number of peaks detected for each pixel, excluding background pixels.

    Parameters
    ----------
    intensity_data : ndarray
        Array of intensity values, shape (rows, cols, q).
    background_pixels : list
        List of pixel indices (row, col) to exclude from analysis.
    kernel_size : int
        Kernel size for median filtering.
    distance : int
        Minimum distance between peaks for peak detection.
    width : int
        Minimum width of peaks for peak detection.

    Returns
    -------
    dict
        Dictionary with the number of peaks as keys and lists of pixel indices as values.
    ndarray
        Array with the number of peaks found for each pixel, shape (rows, cols).
    """
    num_rows, num_cols, _ = intensity_data.shape
    num_peaks = {}
    peaks_found = np.zeros((num_rows, num_cols))  # Initialize array for peak counts

    for row in range(num_rows):
        for col in range(num_cols):
            if (row, col) in background_pixels:
                continue

            # Extract 1D intensity data for the current pixel
            intensity_1d = intensity_data[row, col, :]

            # Apply median filter and find peaks
            filtered_intensity = signal.medfilt(intensity_1d, kernel_size)
            peak_indices = signal.find_peaks(filtered_intensity, distance=distance, width=width)[0]
            num_found = len(peak_indices)

            # Update dictionary and peaks array
            if num_found not in num_peaks:
                num_peaks[num_found] = []
            num_peaks[num_found].append((row, col))
            peaks_found[row, col] = num_found

    return num_peaks, peaks_found

def gaussian_fit(q_values, average_intensity, max_peaks_to_fit=10, prominence=0.001, width=5):
    """
    Performs a Gaussian fit on average intensity data.

    Parameters
    ----------
    q_values : np.ndarray
        Array of q-values (Å⁻¹).
    mean_intensity : np.ndarray
        Array of average intensity values corresponding to q-values.
    max_peaks_to_fit : int, optional
        Maximum number of peaks to fit (default: 10).
    prominence : float, optional
        Prominence threshold for peak detection (default: 0.001).
    width : int, optional
        Minimum width of peaks for peak detection (default: 5).

    Returns
    -------
    np.ndarray
        Fitted parameters of the Gaussian model.
    """
    # Scale input data
    q_scaled = (q_values - np.min(q_values)) / (np.max(q_values) - np.min(q_values))
    intensity_scaled = (mean_intensity - np.min(mean_intensity)) / (np.max(mean_intensity) - np.min(mean_intensity))

    # Find peaks directly in the scaled data
    peak_indices, properties = find_peaks(intensity_scaled, prominence=prominence, width=width)
    num_peaks_to_fit = min(len(peak_indices), max_peaks_to_fit)
    print(f'Number of Peaks to Fit: {num_peaks_to_fit}')
    print(f'Performing Gaussian Fit...')

    # Sort peaks by prominence and select the top peaks
    sorted_indices = np.argsort(properties["prominences"])[::-1]
    selected_peaks = peak_indices[sorted_indices[:num_peaks_to_fit]]
    selected_prominences = properties["prominences"][sorted_indices[:num_peaks_to_fit]]
    selected_widths = properties["widths"][sorted_indices[:num_peaks_to_fit]]

    # Construct initial guesses
    initial_guess = []
    for i in range(num_peaks_to_fit):
        A = selected_prominences[i]
        mu = q_scaled[selected_peaks[i]]
        sigma = selected_widths[i] / (2.355 * (q_scaled[1] - q_scaled[0]))
        initial_guess.extend([A, mu, sigma])

    # Define the Gaussian model
    def multi_gaussian(x, *params):
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            A, mu, sigma = params[i:i + 3]
            y += A * norm.pdf(x, loc=mu, scale=sigma)
        return y

    # Set bounds
    lower_bounds = [0, 0, 0] * num_peaks_to_fit
    upper_bounds = [2 * max(intensity_scaled), 1.0, 0.2] * num_peaks_to_fit
    initial_guess = np.clip(initial_guess, lower_bounds, upper_bounds)

    # Fit the Gaussian model to the data
    try:
        popt, _ = curve_fit(
            multi_gaussian, q_scaled, intensity_scaled,
            p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=1000
        )
    except RuntimeError as e:
        print("Fit failed:", e)
        popt = initial_guess  # Use initial guess if fit fails
        
    print("Fitted Parameters (Scaled):")
    for i in range(num_peaks_to_fit):
        A, mu, sigma = popt[i*3:(i+1)*3]
        print(f"Peak {i+1}: Amplitude (A) = {A:.4f}, Mean (mu) = {mu:.4f}, Std Dev (sigma) = {sigma:.4f}")
    return popt, q_scaled, intensity_scaled, multi_gaussian
