import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner
import math
import hmf
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
plt.rc('axes', labelsize=14)


G = G.to(u.M_sun**-1 * u.km**2 * u.s**-2 * u.Mpc).value

bin_width = 0.1  # dex of bin width

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))


def M200_Disp_Cor(vd, rad):
    """
    Calculate M200 using velocity dispersion and radius corrections.
    """
    alpha = 1.052
    vd_lim = 227.327
    n1 = -1.926
    beta = 0.326
    rad_lim = 0.320
    n2 = -1.378

    # Use np.where for vectorized conditional logic
    Ab = np.where(vd < vd_lim, alpha * ((vd / vd_lim)**n1 - 1), 0)
    Ac = np.where(rad < rad_lim, beta * ((rad / rad_lim)**n2 - 1), 0)
    A = 5 / 3 + Ab + Ac

    return np.log10(A * vd**2 * rad / G)

with fits.open('/fred/oz004/wvankemp/Halo_Mass_Relations/Data/group_galaxies.fits') as hdul:
    datag = hdul[1].data  # Assuming the data is in the first extension
# Convert to pandas DataFrame with little-endian format
df = pd.DataFrame({name: datag[name].byteswap().newbyteorder() for name in datag.names})

df['Mhalo_Disp'] = M200_Disp_Cor(df['vdisp_gap'], df['sep_max'])

# Define cosmology
h = 0.7
cosmo_model = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.0491, Tcmb0=2.725)

mag_limit = 21.2  # apparent magnitude limit in Z VISTA band

def make_group_data(dat, N):
    """
    Create a DataFrame with group data.
    
    Parameters:
    -----------
    dat : pandas DataFrame
        DataFrame containing group galaxy data
        
    Returns:
    --------
    group_dat : pandas DataFrame
        DataFrame containing group data
    """
    # Sort the data by group ID and magnitude (ascending, smaller is brighter)
    dat = dat.sort_values(by=['id_group_sky', 'total_ap_dust_Z_VISTA'])

    # Group by 'id_group_sky' and pick the 3rd brightest member (index 2 in zero-based indexing)
    third_brightest = dat.groupby('id_group_sky').nth(N-1)

    # Select the required columns for the group data
    group_dat = third_brightest[['ra', 'dec', 'zobs_new', 'mvir_hosthalo', 'total_ap_dust_Z_VISTA', 'Nm']].reset_index()
    
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
    M = df['total_ap_dust_Z_VISTA'] - 5 * np.log10(d_L / 10)
    dist = 10**((mag_limit - M) / 5 + 1) * u.pc  # Calculate distance using distance modulus
    ztest = np.linspace(0, 0.1, 1000000)
    d_L_test = cosmo_model.luminosity_distance(ztest).to(u.pc).value
    z_interp = interp1d(d_L_test, ztest, bounds_error=False, fill_value=0.1)
    zmax = z_interp(dist)
    
    # Convert to comoving volume
    V_max_full_sky = cosmo_model.comoving_volume(zmax).value  # in h^-3 Mpc^3
    
    # Adjust for survey area
    sky_fraction = (survey_area_deg2_1+survey_area_deg2_2) / (4 * np.pi * (180/np.pi)**2)
    V_max = V_max_full_sky * sky_fraction
    
    return V_max

def compute_hmf(df, v_max, bw):
    """
    Compute halo mass function for masses already in log10(M_sun).
    
    Parameters:
    -----------
    log_masses : array
        halo masses in log10(M_sun)
    v_max : array
        Maximum volumes in Mpc^3
    mass_bins : array
        Bin edges for mass function in log10(M_sun)
        
    Returns:
    --------
    bin_centers : array
        Centers of the mass bins
    phi : array
        Number density in each bin (in Mpc^-3 dex^-1)
    phi_err : array
        Error on number density
    n : array
        Number of galaxies in each bin
    """
    log_mass_min = df['mvir_hosthalo'].min()
    log_mass_max = df['mvir_hosthalo'].max()
    bin_width = bw  # dex
    mass_bins = np.arange(log_mass_min, log_mass_max + bin_width, bin_width)
    # Initialize arrays
    n_bins = len(mass_bins) - 1
    phi = np.zeros(n_bins)
    phi_err = np.zeros(n_bins)
    bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])
    bin_width = mass_bins[1] - mass_bins[0]  # Assuming uniform bins
     
    # hist, bins = np.histogram( df['SM'], bins=mass_bins )

    # Use histogram to count galaxies in each bin
    hist, _ = np.histogram(df['mvir_hosthalo'], bins=mass_bins, weights=1.0 / v_max)
    phi = np.log10( hist / bin_width)
    # Calculate the error on the histogram
    hist_err, _ = np.histogram(df['mvir_hosthalo'], bins=mass_bins, weights=(1.0 / v_max)**2)
    phi_err = np.sqrt(hist_err) / bin_width
    phi_err /=  (10**phi * np.log(10))
    n, _ = np.histogram(df['mvir_hosthalo'], bins=mass_bins)
    
    return bin_centers, phi, phi_err, n

def schechter_log(log_m, log_phi_star, log_m_star, alpha, beta):
    """Schechter function with logarithmic stellar mass input."""
    m = 10**log_m
    m_star = 10**log_m_star
    phi_star = 10**log_phi_star
    return np.log10(np.log(10) * phi_star * beta * (m / m_star)**(alpha + 1) * np.exp(-(m / m_star)**beta))




# def log_likelihood(theta, bin_centers, phi, phi_err):
#     """
#     Log-likelihood function for the Schechter function fit.
#     Define the log-likelihood using a Student's t-distribution
#     """
#     log_phi_star, log_m_star, alpha, beta = theta
#     model = schechter_log(bin_centers, log_phi_star, log_m_star, alpha, beta)
#     residuals = phi - model
#     nu = 4  # Degrees of freedom for the Student's t-distribution
#     return np.sum(t.logpdf(residuals / phi_err, df=nu) - np.log(phi_err))


def log_likelihood(theta, bin_centers, phi, phi_err):
    log_phi_star, log_m_star, alpha, beta = theta
    model = schechter_log(bin_centers, log_phi_star, log_m_star, alpha, beta)
    residuals = phi - model
    sigma = phi_err
    chi = residuals / sigma
    return -0.5 * np.nansum(chi**2 - 0.5 * np.log(2 * np.pi * sigma**2))


# Define the log-prior
def log_prior(theta):
    log_phi_star, log_m_star, alpha, beta = theta
    if -5 < log_phi_star < 0 and 12 < log_m_star < 15 and -2.25 < alpha < 0.5 and 0 < beta < 1.25:
        return 0.0
    return -np.inf

# Define the log-probability
def log_probability(theta, bin_centers, phi, phi_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, bin_centers, phi, phi_err)


# Create group data with N cutoff member
dg = make_group_data(df, N=3)

# Compute areas using the integration method
area_1 = survey_area_integral(157.25, 225.0, -3.95, 3.95)
area_2 = survey_area_integral(330.0, 52.0, -35.6, -27.0)
print(f"Area 1: {area_1:.2f} deg^2")
print(f"Area 2: {area_2:.2f} deg^2")

# Compute Vmax
V_max = compute_Vmax(dg, area_1, area_2)

# Compute HMF 
bin_centers, phi, phi_err, n = compute_hmf(dg, V_max, bin_width) # dex of bin width
# Initial guess and MCMC setup
ndim = 4  # Number of parameters
nwalkers = 32  # Number of walkers
initial_guess = [-2, 12, -1.8, 0.75]  # Initial guess for [log_phi_star, log_m_star, alpha, beta]
nsteps = 1000
burnin = 100
pos = initial_guess + 1e-3 * np.random.randn(nwalkers, ndim)
# pos = initial_guess + 0.1 * np.random.randn(nwalkers, ndim)

# MCMC sampling
max_steps = 1000000
check_interval = int(max_steps / 20)
converged = False

# Filter the data for phi that is not NaN and bin_centers >= x
valid_indices =  ~np.isnan(phi_err) & ~np.isnan(phi) & (bin_centers >= 12.1)
filtered_bin_centers = bin_centers[valid_indices]
filtered_phi = phi[valid_indices]
filtered_phi_err = phi_err[valid_indices]

# Quick Sampler
# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(filtered_bin_centers, filtered_phi, filtered_phi_err))
# sampler.run_mcmc(pos, nsteps, progress=True)
# samples = sampler.get_chain(discard=burnin, thin=10, flat=True)

# MCMC sampling with convergence check using autocorrelation time
max_steps = 2000000  # Maximum number of steps to attempt
check_interval = 2000  # Check for convergence every 500 steps
converged = False

# Initialize the sampler
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(filtered_bin_centers, filtered_phi, filtered_phi_err))

# Run the sampler dynamically
for sample in sampler.sample(pos, iterations=max_steps, progress=True):
    if sampler.iteration % check_interval == 0:
        try:
            # Estimate the autocorrelation time
            tau = sampler.get_autocorr_time(tol=0)
            print(f"Iteration {sampler.iteration}: Autocorrelation time: {tau}")

            # Check if the chains have likely converged
            if np.all(tau * 50 < sampler.iteration) and np.all(np.abs(tau - sampler.get_autocorr_time(tol=0)) / tau < 0.01):
                print("Chains have likely converged.")
                converged = True
                break
        except emcee.autocorr.AutocorrError as e:
            # Autocorrelation time could not be estimated yet
            print(f"Iteration {sampler.iteration}: Autocorrelation time could not be estimated yet. {e}")

# If not converged, warn the user
if not converged:
    print("Warning: Chains may not have fully converged. Consider running for more steps.")

# Determine burn-in and thinning based on the final autocorrelation time
if converged:
    tau = sampler.get_autocorr_time(tol=0)
    burnin = int(5 * np.max(tau))  # Discard 5 times the maximum autocorrelation time
    thin = int(np.max(tau) / 2)  # Thin by half the maximum autocorrelation time
else:
    burnin = int(max_steps * 0.1)  # Default burn-in if not converged
    thin = int(max_steps / 1000)  # Default thinning if not converged

# Flatten the chain and discard burn-in
samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

# Compute the best-fit parameters and uncertainties
best_fit = np.median(samples, axis=0)
uncertainties = np.std(samples, axis=0)
print(f"Best-fit parameters: {best_fit}")
print(f"Uncertainties: {uncertainties}")

# Plot the corner plot
labels = [r"$\log \phi^*$", r"$\log M^*$", r"$\alpha$", r"$\beta$"]
fig = corner.corner(
    samples, labels=labels, truths=best_fit, 
    quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".2f", title_kwargs={"fontsize": 12}
)
# Save the corner plot
plt.savefig('/fred/oz004/wvankemp/Halo_Mass_Relations/Plots/Shark_HMF_corner.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

Driver_x = np.linspace(100, 16, 1000)
d_h = 0.6737
D_mstar = np.log10(10**14.13 * d_h/h)
D_phi_star = np.log10(10**-3.96 * d_h**3 / h**3)
D_alpha = -1.68
D_beta = 0.63
D_xfit = np.linspace(10, 16, 1000)
D_yfit = schechter_log(D_xfit, D_phi_star, D_mstar, D_alpha, D_beta)


xfit = np.linspace(10, 16, 1000)
yfit = schechter_log(xfit, best_fit[0], best_fit[1], best_fit[2], best_fit[3])

p_h = 0.6737
p_mstar = np.log10(10**14.43 * d_h/h)
p_phi_star = np.log10(10**-4.49 * d_h**3 / h**3)
p_alpha = -1.85 
p_beta = 0.77
p_xfit = np.linspace(10, 16, 1000)
p_yfit = schechter_log(p_xfit, p_phi_star, p_mstar, p_alpha, p_beta)

# Compute the theoretical halo mass function using the hmf package
# Define the mass range for the theoretical HMF
# Update sigma8 to 0.8150 in the MassFunction object
mass_range = np.logspace(11, 16, 500)  # Adjust to match the number of bins in hmf.dndlog10m
# Initialize the MassFunction object with the given cosmology
mf = MassFunction(cosmo_model=cosmo_model, Mmin=11, Mmax=16, dlog10m=0.01, sigma_8=0.8150)
# Extract the halo mass function (dn/dlog10M) and convert to log10
hmf_log_mass = np.log10(mf.m / h)  # Use mf.m for the mass array from the MassFunction object
hmf_phi = np.log10(mf.dndlog10m * h**4)  # Use mf.dndlog10m for the number density


# Plot the HMF
plt.figure(figsize=(5.4, 4))

# Plot the theoretical HMF
plt.plot(hmf_log_mass, hmf_phi, color='k', 
         label=r'$\Lambda$ CDM, $\Omega_{m}=0.3121$, $\Omega_{b}=0.0491$' '\n' 
               r'$T_{CMB}=2.725$, $\sigma_{8}=0.8150$', 
         linestyle='-', linewidth=2.5)
# Plot data points
plt.plot(D_xfit, D_yfit, color='darkgray', label='Driver et al. (2022) GAMA5+SDSS5+REFLEX II', linestyle='--', linewidth=2.5)
# plt.plot(p_xfit, p_yfit, color='k', label='Planck18', linestyle='--', linewidth=2)
plt.plot(xfit, yfit, color='magenta', label=r'SHARK HMF Fit', linestyle='--', linewidth=2.5)
# Separate points included in MCMC fitting and those not included
included_indices = valid_indices
excluded_indices = ~valid_indices
# Plot included points as error bars with filled diamonds
plt.errorbar(bin_centers[included_indices], phi[included_indices], yerr=phi_err[included_indices],
             fmt='o', color='b', ecolor='b', markersize=5, label='SHARK - Included in Fit', alpha=0.7, capsize=3)

# Plot excluded points as error bars with unfilled diamonds
plt.errorbar(bin_centers[excluded_indices], phi[excluded_indices], yerr=phi_err[excluded_indices],
             fmt='o', markerfacecolor='none', markeredgecolor='b', ecolor='b', markersize=5, label='SHARK - Excluded from Fit', alpha=0.7, capsize=3)
# Set axis labels and scale
plt.xlabel(r'$\log$ M$_{halo}$ [M$_{\odot}$]')
plt.ylabel(r'$\log$ $\Phi$ [Mpc$^{-3}$ dex$^{-1}$]')
plt.ylim(-8.1, -1)
plt.xlim(11.05, 15.4)
# plt.grid(True, alpha=0.3)
plt.legend(loc='lower left', fontsize=8)
plt.minorticks_on()
plt.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                bottom=True, top=True, left=True, right=True, direction='in')
plt.tight_layout()
plt.savefig('/fred/oz004/wvankemp/Halo_Mass_Relations/Plots/Shark_HMF.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print("log phi:", phi)
print("log phi err:", phi_err)
print("log mass centers:", bin_centers)
print("number of groups in each bin:", n)