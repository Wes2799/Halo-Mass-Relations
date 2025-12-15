import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner
import math
import hmf
from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.optimize import curve_fit
from astropy.cosmology.funcs import z_at_value
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.stats import t
from multiprocessing import cpu_count
from hmf import MassFunction
from astropy.constants import G

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('axes', labelsize='large')
plt.rc('axes', titlesize='large')
plt.rc('ytick', labelsize='large')
plt.rc('legend', fontsize='x-small')

G = G.to(u.M_sun**-1 * u.km**2 * u.s**-2 * u.Mpc).value
bin_width = 0.15
zmax = 0.1
zmax_2 = 0.08
zmin = 0.008
Ncut = 3  # Nth brightest member to select from each group

# Cosmological parameters
h = 0.7
cosmo = FlatLambdaCDM(H0=100*h, Om0=0.3121, Ob0=0.0491, Tcmb0=2.725)

band_2dF = 'BJG'
band_2dF_lim = 19.45
band_GAMA = 'MAG_AUTO_I'
band_GAMA_lim = 19.2


Mhalo = 'Mhalo_CVT'
Mhalo2 = 'Mhalo_VT'
# Mhalo = 'Mhalo_P3M'

vmax_cor = 'Uncorrected_Vmax'

def make_Mhalo_P3M(df):
    # A = 31.111
    # M_A = 10.769
    # beta = 0.889
    # gamma = -0.555
    A = 46.944
    M_A = 10.483
    beta = 0.249
    gamma = -0.601
    Mc = 10**df['P3M']  # Convert log values back to linear scale
    Mh = A * Mc * ((Mc / (10**M_A))**(beta) + (Mc / (10**M_A))**gamma)
    return np.log10(Mh)


def make_Mhalo_CVT(df):
    alpha = 1.030
    vd_lim = 244.634
    n1 = -1.989
    beta = 0.213
    rad_lim = 0.369
    n2 = -1.591
    # Vectorized conditional calculation for Ab and Ac
    Ab = np.where(df['vdisp_gap'] < vd_lim,
                 alpha * ((df['vdisp_gap'] / vd_lim)**n1 - 1),
                 0)
    Ac = np.where(df['sep_max'] < rad_lim,
                 beta * ((df['sep_max'] / rad_lim)**n2 - 1),
                 0)
    A = 5/3 + Ab + Ac
    return np.log10(A * df['vdisp_gap']**2 * df['sep_max'] / G)

def make_group_data(df, band, mhalo, mhalo2, N):
    """
    Create a DataFrame with group data.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing group galaxy data
    band : str
        Band name for magnitude
    mhalo : str
        Halo mass column name
    N : int
        The Nth brightest member to select from each group
        (e.g., 3 for the 3rd brightest)
        
    Returns:
    --------
    group_dat : pandas DataFrame
        DataFrame containing group data
    """
    # Sort the data by group ID and magnitude (ascending, smaller is brighter)
    df = df.sort_values(by=['group_id', str(band)])

    # Group by 'id_group_sky' and pick the 3rd brightest member (index 2 in zero-based indexing)
    third_brightest = df.groupby('group_id').nth(N-1)

    # Select the required columns for the group data
    group_dat = third_brightest[['ra_median', 'dec_median', 'zobs_median', str(band), str(mhalo), str(mhalo2), 'Nm']].reset_index()
    return group_dat

def survey_area_integral(ra_min, ra_max, dec_min, dec_max):
    """
    Compute the survey area using integration with input validation.
    
    Parameters:
    -----------
    ra_min, ra_max : float
        Right Ascension range in degrees
    dec_min, dec_max : float
        Declination range in degrees
        
    Returns:
    --------
    area : float
        Survey area in square degrees
    
    Raises:
    -------
    ValueError: If input parameters are invalid
    """
    # Input validation
    if not (0 <= ra_min < 360 and 0 <= ra_max <= 360):
        raise ValueError("Right Ascension must be between 0 and 360 degrees")
    
    if not (-90 <= dec_min <= dec_max <= 90):
        raise ValueError("Declination must be between -90 and 90 degrees")
    
    # Handle RA wrap-around if needed
    ra_range = ra_max - ra_min if ra_max >= ra_min else (360 - ra_min) + ra_max
    
    def integrand(dec):
        return np.cos(np.radians(dec))
    
    area, _ = quad(integrand, dec_min, dec_max)
    area *= ra_range * (np.pi / 180.0)  # Convert RA range to radians
    return area * (180.0 / np.pi)  # Convert back to square degrees

def compute_Vmax_Cor(df, survey_area, zmin, zmax, band, band_lim):
    """
    Compute Vmax for each galaxy in the DataFrame.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with galaxy data
    survey_area : float
        Survey area in square degrees
    zmin, zmax : float
        Redshift range
    band : str
        Band name for magnitude limit
    band_lim : float
        Magnitude limit for the band
    
    Returns:
    --------
    vmax : Series
        Vmax values for each galaxy
    zmax : Series
        Limiting redshift for each galaxy group used to calculate Vmax
    """
    # Calculate Vmax for each galaxy
    d_L = cosmo.luminosity_distance(df['zobs_median'].values).to(u.pc).value
    # Calculate absolute magnitude using distance modulus
    M = df[str(band)] - 5 * np.log10(d_L / 10)
    dist = 10**((band_lim - M) / 5 + 1) * u.pc  # Calculate distance using distance modulus
    ztest = np.linspace(0, 0.1, 1000000)
    d_L_test = cosmo.luminosity_distance(ztest).to(u.pc).value
    z_interp = interp1d(d_L_test, ztest, bounds_error=False, fill_value=0.1)
    zlim = z_interp(dist)

    # If zmax is less than zlim, set zlim to zmax
    zlim = np.where(zlim > zmax, zmax, zlim)
    
    # Convert to comoving volume
    V_max_full_sky = cosmo.comoving_volume(zlim).value - cosmo.comoving_volume(zmin).value # in h^-3 Mpc^3
    
    # Adjust for survey area
    sky_fraction = (survey_area) / (4 * np.pi * (180/np.pi)**2)
    Vmax = V_max_full_sky * sky_fraction
    
    return Vmax, zlim

def compute_Vmax(df, survey_area, zmin, zmax, band, band_lim):
    """
    Compute Vmax for each galaxy in the DataFrame.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with galaxy data
    survey_area : float
        Survey area in square degrees
    zmin, zmax : float
        Redshift range
    band : str
        Band name for magnitude limit
    band_lim : float
        Magnitude limit for the band
    
    Returns:
    --------
    vmax : Series
        Vmax values for each galaxy
    zmax : Series
        Limiting redshift for each galaxy group used to calculate Vmax
    """
    # Calculate Vmax for each galaxy
    V_max_full_sky = cosmo.comoving_volume(zmax).value - cosmo.comoving_volume(zmin).value # in h^-3 Mpc^3
    
    # Adjust for survey area
    sky_fraction = (survey_area) / (4 * np.pi * (180/np.pi)**2)
    Vmax = V_max_full_sky * sky_fraction
    
    return Vmax, zmax

def compute_HMF(df, bw, mhalo):
    """
    Compute the Halo Mass Function (HMF) for the given DataFrame.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with galaxy data
    bw : float
        Bin width for histogram
    
    Returns:
    --------
    bin_centers : ndarray
        Centers of the bins
    phi: ndarray
         Number density in each bin (in Mpc^-3 dex^-1)
    phi_err: ndarray
         Error in number density in each bin (in Mpc^-3 dex^-1)
    n: ndarray
         Number of galaxies in each bin
    """
    log_mass_min = 11 - 0.5 * bw
    log_mass_max = 15 + 0.5 * bw
    mass_bins = np.arange(log_mass_min, log_mass_max, bw)
    nbins = len(mass_bins) - 1
    bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
    phi = np.zeros(nbins)
    phi_err = np.zeros(nbins)

    # Use histogram to count galaxies in each bin
    hist, _ = np.histogram(df[str(mhalo)], bins=mass_bins, weights=1.0 / df['vmax'])
    phi = np.log10(hist / bw)
    # Calculate the error on the histogram
    hist_err, _ = np.histogram(df[str(mhalo)], bins=mass_bins, weights=(1.0 / df['vmax'])**2)
    phi_err = np.sqrt(hist_err) / bw
    phi_err /=  (10**phi * np.log(10))

    n, _ = np.histogram(df[str(mhalo)], bins=mass_bins)

    return bin_centers, phi, phi_err, n

def schechter_log(log_m, log_phi_star, log_m_star, alpha, beta):
    """Schechter function with logarithmic stellar mass input."""
    m = 10**log_m
    m_star = 10**log_m_star
    phi_star = 10**log_phi_star
    return np.log10(np.log(10) * phi_star * beta * (m / m_star)**(alpha + 1) * np.exp(-(m / m_star)**beta))




#####################################################################################################################################################################################


df = Table.read('/fred/oz004/wvankemp/Halo_Mass_Relations/Data/Obs_Data/V1/Redshift_Group_Galaxies_2dF_GAMA.fits')
df = df.to_pandas()

# df['P3M'] = df.groupby('group_id')['LogSmass'].transform(lambda x: np.log10((10**x).nlargest(min(3, len(x))).sum()))
# df['Mhalo_P3M'] = make_Mhalo_P3M(df)
df['Mhalo_CVT'] = make_Mhalo_CVT(df)
df['Mhalo_VT'] = np.log10( (5/3) * df['vdisp_gap']**2 * df['sep_max'] / G )




total_area = survey_area_integral(340, 26, -35.3, -25.8)   # Total area of the sky in square degrees
print(f"Total area of the sky: {total_area} square degrees")
G23_area = survey_area_integral(339, 351, -35, -30)  # Area of G23 in square degrees
print(f"Area of G23: {G23_area} square degrees")

mask_2dF = (
    ((df['zobs_median'] >= zmin) & (df['zobs_median'] <= zmax_2)) |
    ((df['zobs_median'] >= zmin) & (df['zobs_median'] <= zmax) & 
     ((df['ra_median'] >= 339) & (df['ra_median'] <= 351) & 
       (df['dec_median'] >= -35) & (df['dec_median'] <= -30))))
df_2dF = df[mask_2dF].copy()

mask_G23 = (df['zobs_median'] >= zmin) & (df['zobs_median'] <= zmax) & (df['ra_median'] >= 339) & (df['ra_median'] <= 351) & (df['dec_median'] >= -35) & (df['dec_median'] <= -30)
df_G23 = df[mask_G23].copy()

dg_2dF = make_group_data(df_2dF, band_2dF, Mhalo, Mhalo2, Ncut)
print(f"Number of groups in 2dF: {len(dg_2dF)}, total that has at least {Ncut} members: {len(dg_2dF.dropna())}")
dg_2dF = dg_2dF.dropna()
dg_2dF_G23 = make_group_data(df_G23, band_2dF, Mhalo, Mhalo2, Ncut)
print(f"Number of groups in G23-2dF: {len(dg_2dF_G23)}, total that has at least {Ncut} members: {len(dg_2dF_G23.dropna())}")
dg_2dF_G23 = dg_2dF_G23.dropna()
dg_GAMA = make_group_data(df_G23, band_GAMA, Mhalo, Mhalo2, Ncut)
print(f"Number of groups in G23-GAMA: {len(dg_GAMA)}, total that has at least {Ncut} members: {len(dg_GAMA.dropna())}")
dg_GAMA = dg_GAMA.dropna()



# Create a mask to check if each group is within the G23 RA/Dec range
in_G23 = (
    (dg_2dF['ra_median'] >= 339) & (dg_2dF['ra_median'] <= 351) &
    (dg_2dF['dec_median'] >= -35) & (dg_2dF['dec_median'] <= -30)
)
# Assign zmax based on whether the group is in G23 or not
zmax_array = np.where(in_G23, zmax, zmax_2)

if vmax_cor == 'Corrected_Vmax':
    dg_2dF['vmax'], dg_2dF['zlim'] = compute_Vmax_Cor(dg_2dF, total_area, zmin, zmax_array, band_2dF, band_2dF_lim)
    dg_2dF_G23['vmax'], dg_2dF_G23['zlim'] = compute_Vmax_Cor(dg_2dF_G23, G23_area, zmin, zmax, band_2dF, band_2dF_lim)
    dg_GAMA['vmax'], dg_GAMA['zlim'] = compute_Vmax_Cor(dg_GAMA, G23_area, zmin, zmax, band_GAMA, band_GAMA_lim)
else:
    dg_2dF['vmax'], dg_2dF['zlim'] = compute_Vmax(dg_2dF, total_area, zmin, zmax_array, band_2dF, band_2dF_lim)
    dg_2dF_G23['vmax'], dg_2dF_G23['zlim'] = compute_Vmax(dg_2dF_G23, G23_area, zmin, zmax, band_2dF, band_2dF_lim)
    dg_GAMA['vmax'], dg_GAMA['zlim'] = compute_Vmax(dg_GAMA, G23_area, zmin, zmax, band_GAMA, band_GAMA_lim)

bin_centers_2dF, phi_2dF, phi_err_2dF, n_2dF = compute_HMF(dg_2dF, bin_width, Mhalo)
bin_centers_2dF2, phi_2dF2, phi_err_2dF2, n_2dF2 = compute_HMF(dg_2dF, bin_width, Mhalo2)
bin_centers_2dF_G23, phi_2dF_G23, phi_err_2dF_G23, n_2dF_G23 = compute_HMF(dg_2dF_G23, bin_width, Mhalo)
bin_centers_2dF_G232, phi_2dF_G232, phi_err_2dF_G232, n_2dF_G232 = compute_HMF(dg_2dF_G23, bin_width, Mhalo2)
bin_centers_GAMA, phi_GAMA, phi_err_GAMA, n_GAMA = compute_HMF(dg_GAMA, bin_width, Mhalo)
bin_centers_GAMA2, phi_GAMA2, phi_err_GAMA2, n_GAMA2 = compute_HMF(dg_GAMA, bin_width, Mhalo2)

# Compute the theoretical halo mass function using the hmf package
mass_range = np.logspace(11, 16, 500)  # Adjust to match the number of bins in hmf.dndlog10m
# Initialize the MassFunction object with the given cosmology
mf = MassFunction(cosmo_model=cosmo, Mmin=10, Mmax=16, dlog10m=0.01, sigma_8=0.8150)
# Extract the halo mass function (dn/dlog10M) and convert to log10
hmf_log_mass = np.log10(mf.m / h)  # Use mf.m for the mass array from the MassFunction object
hmf_phi = np.log10(mf.dndlog10m * h**4)  # Use mf.dndlog10m for the number density


Driver_x = np.linspace(10.5, 16, 1000)
d_h = 0.6737
D_mstar = np.log10(10**14.13 * d_h/h)
D_phi_star = np.log10(10**-3.96 * d_h**3 / h**3)
D_alpha = -1.68
D_beta = 0.63
D_xfit = np.linspace(10, 16, 1000)
D_yfit = schechter_log(D_xfit, D_phi_star, D_mstar, D_alpha, D_beta)


# plot the 3 HMF with error bars
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(wspace=0)

axs[0].plot(hmf_log_mass, hmf_phi, color='k', 
         label=r'$\Lambda$ CDM, $\Omega_{m}=0.3121$, $\Omega_{b}=0.0491$' '\n' r'$T_{CMB}=2.725$, $\sigma_{8}=0.8150$', linestyle='-', linewidth=2.5)
axs[0].plot(D_xfit, D_yfit, color='darkgray', label='Driver et al. (2022) GAMA5+SDSS5+REFLEX II', linestyle='--', linewidth=2.5)
axs[0].errorbar(bin_centers_2dF2, phi_2dF2, yerr=phi_err_2dF2, fmt='^', label='SGP - 2dF', color='r', markersize=6)
axs[0].errorbar(bin_centers_GAMA2, phi_GAMA2, yerr=phi_err_GAMA2, fmt='D', label='G23 - GAMA', color='orange', markersize=6)
axs[0].errorbar(bin_centers_2dF_G232, phi_2dF_G232, yerr=phi_err_2dF_G232, fmt='X', label='G23 - 2dF', color='green', markersize=6)
axs[0].set_xlabel(r'$\log$ M$_{\rm{halo}, {VT}}$ [M$_{\odot}$]')


axs[1].plot(hmf_log_mass, hmf_phi, color='k', label=r'$\Lambda$ CDM, $\Omega_{m}=0.3121$, $\Omega_{b}=0.0491$' '\n' r'$T_{CMB}=2.725$, $\sigma_{8}=0.8150$', linestyle='-', linewidth=2.5)
axs[1].plot(D_xfit, D_yfit, color='darkgray', label='Driver et al. (2022) GAMA5+SDSS5+REFLEX II', linestyle='--', linewidth=2.5)
axs[1].errorbar(bin_centers_2dF, phi_2dF, yerr=phi_err_2dF, fmt='^', label='SGP - 2dF', color='r', markersize=6)
axs[1].errorbar(bin_centers_GAMA, phi_GAMA, yerr=phi_err_GAMA, fmt='D', label='G23 - GAMA', color='orange', markersize=6)
axs[1].errorbar(bin_centers_2dF_G23, phi_2dF_G23, yerr=phi_err_2dF_G23, fmt='X', label='G23 - 2dF', color='green', markersize=6)
axs[1].set_xlabel(r'$\log$ M$_{\rm{halo}, {MVT}}$ [M$_{\odot}$]')


for i, ax in enumerate([axs[0], axs[1]]):
    ax.minorticks_on()
    ax.set_ylim(-8.1, -1)
    ax.set_xlim(11.2, 15.2)
    ax.grid(True)
    if i == 0:
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=True, left=True, right=True, direction='in')
        ax.set_ylabel(r'$\log$ $\Phi$ [Mpc$^{-3}$ dex$^{-1}$]')
        ax.legend(loc='lower left', fontsize=10)
    else:
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                       bottom=True, top=True, left=True, right=True, direction='in')

plt.savefig(f'/fred/oz004/wvankemp/Halo_Mass_New/Plots/HMF_{Ncut}_{vmax_cor}.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

