import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.optimize import curve_fit
from astropy.cosmology.funcs import z_at_value
from scipy.interpolate import interp1d
import emcee
import corner
from scipy.integrate import quad


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('axes', labelsize='large')
plt.rc('axes', titlesize='large')
plt.rc('ytick', labelsize='large')
plt.rc('legend', fontsize='x-small')


# Open the FITS file
with fits.open('/fred/oz004/wvankemp/Halo_Mass_Relations/Data/all_galaxies.fits') as hdul:
    data = hdul[1].data  # Assuming the data is in the first extension
# Convert to pandas DataFrame with little-endian format
df = pd.DataFrame({name: data[name].byteswap().newbyteorder() for name in data.names})

with fits.open('/fred/oz004/wvankemp/Halo_Mass_Relations/Data/group_galaxies.fits') as hdul:
    datag = hdul[1].data  # Assuming the data is in the first extension
# Convert to pandas DataFrame with little-endian format
dfg = pd.DataFrame({name: datag[name].byteswap().newbyteorder() for name in datag.names})

# df = df[df['SM']>=7]

# Define cosmology (hless calculations)
h = 0.7
# Define cosmology 
cosmo_model = FlatLambdaCDM(H0=70, Om0=0.3)

mag_limit = 21.2  # apparent magnitude limit in Z VISTA band

def survey_area_integral_shark(ra_min, ra_max, dec_min, dec_max):
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

def k_correction(z):
    """
    Compute k-correction for a given redshift.
    This is a placeholder; replace with a more accurate model if available.
    """
    return (0.1 * z)  # Example approximation

def compute_Vmax(df, survey_area_deg2_1, survey_area_deg2_2):
    """
    Compute the maximum comoving volume for each galaxy.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing galaxy data
    survey_area_deg2 : float
        Survey area in square degrees
        
    Returns:
    --------
    V_max : array
        Maximum volume for each galaxy in h^-3 Mpc^3
    """
    # Calculate absolute magnitudes
    d_L = cosmo_model.luminosity_distance(df['zobs_new'].values).to(u.pc).value
    # Calculate absolute magnitude using distance modulus
    #k_corr = k_correction(df['zobs_new'].values)
    M = df['total_ap_dust_Z_VISTA'] - 5 * np.log10(d_L / 10) #- k_corr
    dist = 10**((mag_limit - M) / 5 + 1) * u.pc  # Calculate distance using distance modulus
    ztest = np.linspace(0, 0.1, 1000000)
    d_L_test = cosmo_model.luminosity_distance(ztest).to(u.pc).value
    z_interp = interp1d(d_L_test, ztest, bounds_error=False, fill_value=0.1)
    zmax = z_interp(dist)
    
    # Convert to comoving volume
    V_max_full_sky = cosmo_model.comoving_volume(zmax).value  # in h^-3 Mpc^3
    
    # Adjust for survey area
    sky_fraction = (survey_area_deg2_1+survey_area_deg2_2) / (4 * np.pi * (180/np.pi)**2),
    V_max = V_max_full_sky * sky_fraction
    
    return V_max


def compute_smf(df, v_max):
    """
    Compute stellar mass function for masses already in log10(h^-1 M_sun).
    
    Parameters:
    -----------
    log_masses : array
        Stellar masses in log10(h^-1 M_sun)
    v_max : array
        Maximum volumes in h^-3 Mpc^3
    mass_bins : array
        Bin edges for mass function in log10(h^-1 M_sun)
        
    Returns:
    --------
    bin_centers : array
        Centers of the mass bins
    phi : array
        Number density in each bin (in h^3 Mpc^-3 dex^-1)
    phi_err : array
        Error on number density
    """
    log_mass_min = df['SM'].min()
    log_mass_max = df['SM'].max()
    bin_width = 0.15  # dex
    mass_bins = np.arange(log_mass_min, log_mass_max + bin_width, bin_width)
    # Initialize arrays
    n_bins = len(mass_bins) - 1
    phi = np.zeros(n_bins)
    phi_err = np.zeros(n_bins)
    bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
    bin_width = mass_bins[1] - mass_bins[0]  # Assuming uniform bins
     
    # hist, bins = np.histogram( df['SM'], bins=mass_bins )

    # Loop through mass bins
    # Use histogram to count galaxies in each bin
    hist, _ = np.histogram(df['SM'], bins=mass_bins, weights=1.0 / v_max)
    phi = hist / bin_width  # Normalize by bin width to get dex^-1

    # Error estimate (Poisson)
    counts, _ = np.histogram(df['SM'], bins=mass_bins)
    phi_err = phi / np.sqrt(counts)
    phi_err[counts == 0] = 0  # Avoid division by zero for empty bins
    
    return bin_centers, phi, phi_err


def double_schechter_log(log_m, log_phi_star1, log_phi_star2, log_m_star, alpha1, alpha2):
    """Double Schechter function with logarithmic stellar mass input."""
    M_ratio = log_m - log_m_star
    phi_star1 = 10**log_phi_star1
    phi_star2 = 10**log_phi_star2
    return np.log(10) * np.exp(-10**M_ratio) * (
        phi_star1 * 10**(M_ratio * (alpha1 + 1)) + phi_star2 * 10**(M_ratio * (alpha2 + 1))
    )

def log_likelihood(theta, log_m, phi, phi_err):
    """Log-likelihood function for MCMC."""
    log_phi_star1, log_phi_star2, log_m_star, alpha1, alpha2 = theta
    model = double_schechter_log(log_m, log_phi_star1, log_phi_star2, log_m_star, alpha1, alpha2)
    residuals = phi - model
    chi = residuals / phi_err
    like = -0.5 * chi**2 - 0.5 * np.log(2 * np.pi * phi_err)
    return  np.nansum(like)

def log_prior(theta):
    """Log-prior function with reasonable constraints."""
    log_phi_star1, log_phi_star2, log_m_star, alpha1, alpha2 = theta
    if -6 < log_phi_star1 < 0 and -6 < log_phi_star2 < 0 and 9 < log_m_star < 12 and -5 < alpha1 < 0 and -5 < alpha2 < 0:
        return 0.0  # Uniform prior within these bounds
    return -np.inf  # Return -inf for parameters outside allowed range

def log_probability(theta, log_m, phi, phi_err):
    """Total log probability combining prior and likelihood."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, log_m, phi, phi_err)




## RA ranges are 157.25 <= ra <= 225.0 and 330.0 <= ra <= 52.0 
## Dec ranges are -3.95 <= dec <= 3.95 and -35.6 <= dec <= -27.0





# Compute areas using the integration method
area_1 = survey_area_integral_shark(157.25, 225.0, -3.95, 3.95)
area_2 = survey_area_integral_shark(330.0, 52.0, -35.6, -27.0)
print(f"Area 1: {area_1:.2f} deg^2")
print(f"Area 2: {area_2:.2f} deg^2")

V_max = compute_Vmax(df, area_1, area_2) 
Vg_max = compute_Vmax(dfg, area_1, area_2)
# Remove h dependence
# V_max = V_max * h**3
# df['SM'] = np.log10(10**df['SM'] * h)

# Compute the stellar mass function
bin_centers, phi, phi_err = compute_smf(df, V_max)
log_phi_err = phi_err / (phi * np.log(10))
bin_centersg, phig, phi_errg = compute_smf(dfg, Vg_max)

bin_centers, phi2, phi2_err = compute_smf(df, np.ones(len(df)) )
vol = (area_1 + area_2) * (np.pi/180.)**2. / (4.*np.pi) * cosmo_model.comoving_volume(0.1).value
phi2 /= vol

# Set up curve_fit
p0 = [0.001, 0.001, 10.5, -1.0, -1.0]
popt, pcov = curve_fit(double_schechter_log, bin_centers, phi, p0=p0, sigma=phi_err, absolute_sigma=True)

# make xfit range and yfit
xfit = np.linspace(6, 12, 1000)
yfit = double_schechter_log(xfit, *popt)


G_mstar = np.log10(10**10.745)
G_phi_star1 = np.log10(10**-2.437 )
G_phi_star2 = np.log10(10**-3.201)
G_alpha1 = -0.466
G_alpha2 = -1.530
G_xfit = np.linspace(6, 12, 1000)
G_yfit = double_schechter_log(G_xfit, G_phi_star1, G_phi_star2, G_mstar, G_alpha1, G_alpha2)

W_mstar = np.log10(10**10.66)
W_phi_star1 = np.log10(2.93e-3)
W_phi_star2 = np.log10(1.5e-3)
W_alpha1 = -0.62
W_alpha2 = -1.5
W_xfit = np.linspace(6, 12, 1000)
W_yfit = double_schechter_log(W_xfit, W_phi_star1, W_phi_star2, W_mstar, W_alpha1, W_alpha2)

B_mstar = np.log10(10**10.66)
B_phi_star1 = np.log10(3.96e-3)
B_phi_star2 = np.log10(0.79e-3)
B_alpha1 = -0.35
B_alpha2 = -1.47
B_xfit = np.linspace(6, 12, 1000)
B_yfit = double_schechter_log(B_xfit, B_phi_star1, B_phi_star2, B_mstar, B_alpha1, B_alpha2)

# Initial guess
# Filter data for the x range of 8 to 11.5
mask = (bin_centers >= 8) & (bin_centers <= 11.5)
filtered_bin_centers = bin_centers[mask]
filtered_phi = phi[mask]
filtered_phi_err = phi_err[mask]

p0 = [np.log10(0.001), np.log10(0.001), 10.5, -1.0, -1.0]
popt, _ = curve_fit(double_schechter_log, filtered_bin_centers, filtered_phi, p0=p0, sigma=filtered_phi_err, absolute_sigma=True)

# Set up MCMC
ndim = 5            # Number of dimensions (parameters)
nwalkers = 35       # Number of walkers (try reducing slightly for better sampling)
nsteps = 20000      # Number of steps (sufficient for reasonable convergence)
discard = 1000      # Number of steps to discard as burn-in
pos = popt + 1e-5 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(filtered_bin_centers, filtered_phi, filtered_phi_err))

# Run MCMC
sampler.run_mcmc(pos, nsteps, progress=True)

# Extract samples and discard burn-in
flat_samples = sampler.get_chain(discard=discard, thin=10, flat=True)

# Plot parameter distributions
fig = corner.corner(flat_samples, labels=["log_phi_star1", "log_phi_star2", "log_m_star", "alpha1", "alpha2"], quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_fmt=".3f",
                    truths=np.median(flat_samples, axis=0))
                    
plt.savefig('/fred/oz004/wvankemp/Halo_Mass_Relations/Plots/Shark_SMF_corner.png', dpi=300, bbox_inches='tight')
plt.show()

# Compute best-fit parameters from median of posterior distribution
best_fit_params = np.median(flat_samples, axis=0)

# Generate best-fit curve
xfit2 = np.linspace(7, 12, 1000)
yfit2 = double_schechter_log(xfit2, *best_fit_params)


plt.figure(figsize=(5.4, 4))
# Plot data points
log_phi_errg = phi_errg / (phig * np.log(10))
plt.plot(B_xfit, np.log10(B_yfit), color='blue', label='Baldry +12', linestyle='--', linewidth=2)
plt.plot(G_xfit, np.log10(G_yfit), color='green', label='Driver +22', linestyle='--', linewidth=2)
# plt.plot(W_xfit, np.log10(W_yfit), color='purple', label='Wright +17', linestyle='--', linewidth=2)
plt.errorbar(bin_centers, np.log10(phi), yerr=log_phi_err, fmt='o', 
             markersize=6, capsize=4, color='k', label='SMF', alpha=0.8)
# plt.errorbar(bin_centersg, np.log10(phig), yerr=log_phi_errg, fmt='o',
            #  markersize=6, capsize=4, color='red', label='Group SMF', alpha=0.3)
# plt.scatter( bin_centers, np.log10(phi2), s=5, marker='x', label='ned' )
# plt.plot(xfit, np.log10(yfit), color='k', label='Data Schechter Fit', linestyle='--', linewidth=2)
# plt.plot(xfit2, np.log10(yfit2), color='magenta', label='MCMC Fit', linestyle='-.', linewidth=2)
# plt.plot(bin_centers, np.log10(phi), 'r-', label='Data', linewidth=2)
# Set axis labels and scale
plt.xlabel(r'$\log$ M$_{\star}$ [M$_{\odot}$]')
plt.ylabel(r'$\log$ $\Phi$ [Mpc$^{-3}$ dex$^{-1}$]')
plt.ylim(-6.5, 0)
plt.xlim(7.5, 12.4)
plt.legend(loc='upper right')
plt.minorticks_on()
plt.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                          bottom=True, top=True, left=True, right=True, direction='in')
    
plt.tight_layout()
plt.savefig('/fred/oz004/wvankemp/Halo_Mass_Relations/Plots/Shark_SMF.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()