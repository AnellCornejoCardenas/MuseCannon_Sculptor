# Functions for using and analyzing The Cannon predictions

import numpy as np
from astropy.table import Table
import pandas as pd
from astropy import constants
import matplotlib.pyplot as plt
from pathlib import Path
import thecannon as tc
from PyAstronomy import pyasl


# Increasing sigma in telluric regions
def Telluric_bands(filename='/Users/anell/avatar_MuseCannon/data/linelists/ardata.fits'):
    """
    Identify wavelength intervals affected by strong telluric absorption
    using the Arcturus atlas data (Hinkle et al. 2000). This function 
    reads the 'ardata.fits' table, selects wavelengths with transmission 
    values below -0.5, and groups contiguous points into continuous 
    telluric bands separated by more than 100 Å.

    Parameters
    ----------
    - filename : str or astropy.table.Table 
               Path to the input FITS table.

    Returns
    -------
    - telluric_bands : list of [start, end]. Wavelength ranges (in Å) of the
                      main telluric absorption regions.
    """
    table = Table.read(filename)
    m_tell = table['TELLURIC'] < -0.5   #just selecting the wavelengths with transmission < -0.5
    m_wavel = table['WAVELENGTH'][m_tell]

    telluric_bands = []
    
    for i in range(len(m_wavel)-1):
        l = m_wavel[i]
        l_high = m_wavel[i+1]
        
        if i == 0:  tell_list = [l]
        
        if l_high - l > 100:
            tell_list.append(l)
            telluric_bands.append(tell_list)
            tell_list = [l_high]
        else:
            pass

    return telluric_bands


def load_and_format(file):
        '''
        (for been used into the spectral_lines_mask's function)
        
        This function loads a file with a format:
        ____________________________
        H I               3970.078         
        H I   \t    4101.743  
        H I   \t    4340.471
        ...
        Pb I  \t    4057.819
        ____________________________


        into:
        ____________________________
        H_I	3970.078
        H_I	4101.743
        H_I	4340.471
        ...
        Pb_I 4057.819
        ____________________________
        
        
        Parameters:
        ---------- 
        - file: Path to a CSV with 'Element, Wavelength' per line.

        Returns:
        -------
        - A dataFrame with cleaned 'Element' and numeric 'Wavelength'
        '''
        
        df = pd.read_csv(file, sep=',', names=['Element', 'Wavelength'], skipinitialspace=True)
        df['Element'] = df['Element'].str.strip().str.replace(r'\s+', '_', regex=True)
        df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
        return df.dropna()


def spectral_lines_mask(elements, spectrum_shape, wavelength_array,nitrogen_carbon=False,aluminum_titanium=False,replacement=False,
                       atomic_lines_file = '/Users/anell/avatar_MuseCannon/data/linelists/2000vnia.book.....H_APPI_lines.txt',
                       CN_file = '/Users/anell/avatar_MuseCannon/data/linelists/CN_line_positions.txt',
                       CH_MgH_file = '/Users/anell/avatar_MuseCannon/data/linelists/CH_and_MgH_line_positions.txt'):
    '''
    Generates a combined spectral mask for given elements, including atomic and molecular lines.

    Parameters:
    ----------
    - elements: list of elements to censor (e.g., ['Mg', 'Al', 'N', 'C'])
    - spectrum_shape: shape of the spectrum to mask
    - wavelength_array: array of wavelengths corresponding to the spectrum
    - nitrogen_carbon: Boolean flag to include carbon strong features into the nitrogen mask.
    - aluminum_magnesium: Boolean flag to include Mg strong features into the Al mask.
    - replacement: Boolean flag to replace Al and N lines by Mg and C strong features.
    - atomic_lines_file: file with the line list of atomic transitions
    - CN_file: file with the line list of CN transitions
    - CH_MgH_file: file with the line list of transitions of CH and MgH

    Returns:
    -------
    - combined_mask: Boolean mask with the same shape as spectrum_shape, masking the specified elements.
                     The mask returns False where the element lines are found.
    '''

    # Load and merge all line data
    df_atomic = load_and_format(atomic_lines_file)         #atomic strong features
    df_cn = load_and_format(CN_file)                       # CN strong features
    df_ch_mgh = load_and_format(CH_MgH_file)               #CH and MgH strong features
    lines_df = pd.concat([df_atomic, df_cn, df_ch_mgh], ignore_index=True)

    # Constants
    LAMOST_res = 1800                          #LAMOST typical spectral resolution
    c = ((constants.c).to('km/s')).value       #speed of the light
    V_rot = 20
    V_res = c / LAMOST_res
    delta_V = np.sqrt(V_res**2 + V_rot**2)
    factor_delta_lambda = delta_V / c

    # Initialize global mask
    combined_mask = np.ones(spectrum_shape, dtype=bool)

    for element in elements:
        element_mask = np.ones(spectrum_shape, dtype=bool)

        # Match all lines that start with the element (e.g., 'Mg_I', 'MgH', 'CH', etc.)
        # For N, match anything that contains '_N' or starts with 'CN'
        if element == 'N':
            if nitrogen_carbon == False: 
                matched_lines = lines_df[lines_df['Element'].str.contains(r'\bN\b|^CN', regex=True)]
            else:
                if replacement== False: 
                    matched_lines = lines_df[lines_df['Element'].str.contains(r'\bN\b|^CN|^CH|^C_', regex=True)] #we considere C contributions
                else: matched_lines = lines_df[lines_df['Element'].str.contains(r'^CH|^C_', regex=True)] #we considere just C contributions
        
        elif element == 'C':
            matched_lines = lines_df[lines_df['Element'].str.contains(r'^CH|^CN|^C_', regex=True)]   
        
        elif element == 'Mg':
            matched_lines = lines_df[lines_df['Element'].str.contains(r'^Mg|MgH', regex=True)]
        
        elif element == 'Al':
            if aluminum_titanium == False: 
                matched_lines = lines_df[lines_df['Element'].str.contains(r'^Al_', regex=True)]
            else:
                if replacement== False: matched_lines = lines_df[lines_df['Element'].str.contains(r'^Ti_|^Al_', regex=True)] #we considere Al and Ti contributions
                else: matched_lines = lines_df[lines_df['Element'].str.contains(r'^Ti_', regex=True)]
        
        else:
            matched_lines = lines_df[lines_df['Element'].str.startswith(f'{element}_')]

        for l0 in matched_lines['Wavelength'].values:
            delta_lambda = factor_delta_lambda * l0
            mask = (wavelength_array >= (l0 - delta_lambda)) & (wavelength_array <= (l0 + delta_lambda))
            element_mask &= ~mask  # False where elements are found

        combined_mask &= element_mask  # Combine this element’s mask

    return combined_mask


def plot_leading_theta_coefficients(model, wavelength, telluric_file=None, label_range=(5, 10)):
    """
    Plot leading theta coefficients of a trained Cannon model.

    Parameters
    ----------
    model : tc.CannonModel
        Trained Cannon model.

    wavelength : array-like
        Wavelength array corresponding to the model dispersion.

    telluric_file : str or Path, optional
        Path to ardata.fits used to highlight telluric regions.

    label_range : tuple, optional
        Range of label indices to plot (default = (5, 10)).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure.
    """

    in_wavelength_range = wavelength > 0

    fig, axes = plt.subplots(5, 1, figsize=(15, 9), sharex=False)

    if telluric_file is not None:
        tell_bands = Telluric_bands(filename=telluric_file)
    else:
        tell_bands = []

    for i in range(*label_range):

        ylabel = f"[{model.vectorizer.label_names[i].replace('_', '/')}]"

        ax = axes[i - label_range[0]]

        ax.plot(
            wavelength[in_wavelength_range],
            model.theta[in_wavelength_range, i + 1],
            c='C3',
            lw=1,
            alpha=0.8
        )

        ax.set_ylim(-0.05, 0.05)
        ax.set_xlim(4000, 8500)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(r'Wavelength ($\AA$)')

        for band in tell_bands:
            ax.axvspan(band[0], band[1], alpha=0.1,c='grey')

    plt.tight_layout()

    return fig



def train_or_load_model(model, model_file):
    """
    Load an existing Cannon model if available, otherwise train and save it.

    Parameters
    ----------
    model : tc.CannonModel
        Cannon model instance to train or load.

    model_file : str or Path
        Output filename for the model.

    Returns
    -------
    model : tc.CannonModel
        Loaded or trained model.

    theta : ndarray
        Model theta coefficients.

    s2 : ndarray
        Model scatter term.
    """

    model_file = Path(model_file).with_suffix(".model")

    if model_file.exists():
        model = tc.CannonModel.read(model_file)
        theta, s2 = model.theta, model.s2
        print(f"Existing model found: {model_file.name}")

    else:
        print("Training The Cannon...")
        theta, s2, metadata = model.train()
        model.write(model_file, overwrite=True)
        print(f"Model saved as: {model_file.name}")

    return model, theta, s2


import numpy as np
import scipy


import numpy as np
import scipy.ndimage


def normalize_sculptor_spectra(flux, sigma, smooth_sigma=50):
    """
    Apply NaN handling, normalization, and bad-pixel masking
    to Sculptor spectra.
    """

    flux_norm2 = np.array(flux, copy=True)
    flux_norm2_err = np.array(sigma, copy=True)

    test = np.isnan(flux_norm2)
    test2 = np.isnan(flux_norm2_err)

    flux_norm2[test] = np.nanmedian(flux_norm2)
    flux_norm2_err[test] = 999.

    flux_norm2[test2] = np.nanmedian(flux_norm2)
    flux_norm2_err[test2] = 999.

    smoothver = scipy.ndimage.gaussian_filter1d(flux_norm2, smooth_sigma)

    norm_flux = flux_norm2 / smoothver
    norm_sigma = flux_norm2_err / smoothver

    test_bad = np.logical_or(norm_flux < 0.2, norm_flux > 1.3)

    norm_flux[test_bad] = np.nanmedian(norm_flux)
    norm_sigma[test_bad] = 999.

    return norm_flux, norm_sigma


def apply_snr_mask(flux, sigma, original_flux, original_sigma, snr_limit=10):
    """
    Compute SNR and apply SNR mask.
    """

    snr = np.nanmedian(original_flux / original_sigma, axis=1)
    mask = snr > snr_limit

    return flux[mask], sigma[mask], snr, mask


def compute_radial_velocities(
    flux_array,
    muse_wavelength,
    lamost_wavelength,
    template_flux,
    interpolate_to_grid
):
    """
    Compute radial velocities using cross-correlation.
    """

    z_correction_all = []
    velout_all = []

    for s in range(len(flux_array)):

        if s % 10 == 0:
            print('Star :', s)

        output = pyasl.crosscorrRV(
            muse_wavelength[0:-50],
            flux_array[s, 0:-50],
            lamost_wavelength,
            template_flux,
            -250, 250, 0.1,
            mode='doppler'
        )

        output_0 = output[0]
        output_1 = output[1]

        z = np.polyfit(output_0, output_1, 5)
        f = np.poly1d(z)

        newx, newy = interpolate_to_grid(
            output_0,
            output_1,
            np.arange(output_0[0], output_0[-1], 0.01)
        )

        maxy = np.argsort((f(newx)))[-1]
        maxv = newx[maxy]

        velout_all.append(maxv)
        z_correction_all.append(maxv / 299792.458)

    return np.array(z_correction_all), np.array(velout_all)


def restframe_and_interpolate_sculptor(
    flux,
    sigma,
    wavelength,
    z_correction,
    target_wavelength,
    interpolate_to_grid
):
    """
    Apply redshift correction and interpolate spectra
    onto the target wavelength grid.
    """

    final_flux = []
    final_sigma = []

    for s in range(len(flux)):

        # move to rest frame
        wavel_rest = wavelength / (1.0 + z_correction[s])

        # interpolate flux
        wl, flux_interp = interpolate_to_grid(
            wavel_rest,
            flux[s],
            target_wavelength
        )

        # interpolate sigma
        wl, sigma_interp = interpolate_to_grid(
            wavel_rest,
            sigma[s],
            target_wavelength
        )

        final_flux.append(flux_interp)
        final_sigma.append(sigma_interp)

    return np.array(final_flux), np.array(final_sigma)


