import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import seaborn as sns
import pandas as pd
import scipy.optimize as opt
import cmasher as cmr
import multiprocessing as mp
from scipy.optimize import curve_fit
from astropy.table import Table
from astropy.io import fits
from matplotlib.colors import LogNorm
from scipy.optimize import minimize


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('axes', labelsize='large')
plt.rc('axes', titlesize='large')
plt.rc('ytick', labelsize='large')
plt.rc('legend', fontsize='x-small')


def running_stats(x, y, bin_width=0.3):
    bins = np.arange(min(x), max(x) + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    Medians = np.zeros_like(bin_centers)
    p16 = np.zeros_like(bin_centers)
    p84 = np.zeros_like(bin_centers)
    p2_5 = np.zeros_like(bin_centers)
    p97_5 = np.zeros_like(bin_centers)
    
    for i in range(len(bin_centers)):
        in_bin = (x >= bins[i]) & (x < bins[i + 1])
        if np.sum(in_bin) > 0:
            Medians[i] = np.median(y[in_bin])
            p16[i] = np.percentile(y[in_bin], 16)
            p84[i] = np.percentile(y[in_bin], 84)
            p2_5[i] = np.percentile(y[in_bin], 2.5)
            p97_5[i] = np.percentile(y[in_bin], 97.5)
        else:
            Medians[i] = np.nan
            p16[i] = np.nan
            p84[i] = np.nan
            p2_5[i] = np.nan
            p97_5[i] = np.nan
    
    return bin_centers, Medians, p16, p84, p2_5, p97_5

def S2HM(Mc, A, M_A, beta, gamma):
    Mc = 10**Mc  # Convert log values back to linear scale
    Mh = A * Mc * ((Mc / (10**M_A))**(beta) + (Mc / (10**M_A))**gamma)
    return np.log10(Mh)  # Convert back to log scale for fitting

# Define log-likelihood function 
def log_likelihood(params, x, y):
    A, M_A, beta, gamma = params
    model = S2HM(x, A, M_A, beta, gamma)
    residuals = y - model
    sigma = np.std(residuals)
    chi = residuals / sigma
    return   -0.5 * np.nansum(chi**2 + np.log(2 * np.pi * sigma**2))

# Define log-prior function
def log_prior(params):
    A, M_A, beta, gamma = params
    if 39 < A < 70 and 8.5 < M_A < 13.5 and 0.06 < beta < 2 and -2 < gamma < -0.3:
        return 0.0  # Uniform prior
    return -np.inf  # Invalid values

# Define log-probability function
def log_probability(params, x, y):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, x, y)






def MCMC_and_Plots(SAGE, GAEA, band, zlim):
    ###########################################################
    ## Primary 3 Stellar Mass (3 Most massive) - Mvir Host Halo

    SHARK = Table.read(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/{band}/group_catalog_{zlim}.fits')
    SHARK = SHARK.to_pandas()
    SHARK = SHARK[(SHARK['vdisp_gap'] >= (SHARK['rad'] * 150 - 200)) & (SHARK['vdisp_gap'] <= (SHARK['rad'] * 150 + 400)) & (SHARK['rad'] < 12) & (SHARK['vdisp_gap'] < 1800)].reset_index(drop=True)

    # Set up MCMC
    ndim = 4  # Number of parameters
    nwalkers = 40  # Number of walkers
    nsteps = 50000  # Number of MCMC steps
    initial = [45, 10.5, 0.2, -0.8]  # Initial parameter guesses
    burnin= 5000
    pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)  # Random perturbations

    # # Run MCMC
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(SHARK['mstar_cen3'], SHARK['mvir_hosthalo']))
    # sampler.run_mcmc(pos, nsteps, progress=True)

    # # Get results
    # samples = sampler.get_chain(discard=burnin, thin=50, flat=True)  # Discard burn-in

    # MCMC sampling with convergence check using autocorrelation time
    max_steps = 2000000  # Maximum number of steps to attempt
    check_interval = 1000  # Check for convergence every 500 steps
    converged = False


    # Define the number of CPU cores to use
    N = mp.cpu_count()  # Automatically detect the number of CPU cores


    # Create a multiprocessing pool
    with mp.Pool(N) as pool:
        # Initialize the sampler with the pool
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, 
            args=(SHARK['mstar_cen3'], SHARK['mvir_hosthalo']),
            pool=pool
        )

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

    # Plot parameter distributions
    fig = corner.corner(samples, labels=["A", r"M$_A$", r"$\beta$", r"$\gamma$"], 
                        quantiles=[0.16, 0.5, 0.84], 
                        show_titles=True, 
                        title_fmt=".3f", 
                        truths=np.median(samples, axis=0))
    plt.savefig(f'/fred/oz004/wvankemp/Halo_Mass_New/Plots/{band}/SHMR_Corner_{zlim}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print median values and uncertainties
    A_mcmc, M_A_mcmc, beta_mcmc, gamma_mcmc = np.median(samples, axis=0)
    print(f"A = {A_mcmc:.3f}, M_A = {M_A_mcmc:.3f}, beta = {beta_mcmc:.3f}, gamma = {gamma_mcmc:.3f}")

    

    ####################################################################################################################################
    ######################################### 3 panel plot w/ SAGE + GAEA - SHMR w/ residuals ##########################################
    ####################################################################################################################################
    # Plot the data and the fit Primary - Halo Mass

    SHARK_HM_P3 = S2HM(SHARK['mstar_cen3'], A_mcmc, M_A_mcmc, beta_mcmc, gamma_mcmc)
    SAGE_HM_P3 = S2HM(SAGE['M_Cen3'], A_mcmc, M_A_mcmc, beta_mcmc, gamma_mcmc)
    GAEA_HM_P3 = S2HM(GAEA['P3M'], A_mcmc, M_A_mcmc, beta_mcmc, gamma_mcmc)

    SHARK_Delta_HM = SHARK_HM_P3 - SHARK['mvir_hosthalo']
    SAGE_Delta_HM = SAGE_HM_P3 - SAGE['M_200']
    GAEA_Delta_HM = GAEA_HM_P3 - GAEA['Mhalo']

    x_fit = np.linspace(7, 13.5, 1000)
    y_fit = S2HM(x_fit, A_mcmc, M_A_mcmc, beta_mcmc, gamma_mcmc)


    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True, height_ratios=[1, 0.3])
    fig.subplots_adjust(hspace=0.3, wspace=0)

    # Common labels
    x_label_1 = r'$\log$ $\Sigma$M$_{\star, \rm{3}}$ [M$_\odot$]'
    y_label_1 = r'$\log$ M$_{\rm{halo}}$ [M$_\odot$]'
    

    x_label_2 = r'$\log$ $\Sigma$M$_{\star, \rm{3}}$ [M$_\odot$]'
    y_label_2 = r'$\Delta$ $\log$ M$_{\rm{halo}}$ [M$_\odot$]'

    for ax in axes[0]:
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                    bottom=True, top=True, left=True, right=True, direction='in')
        ax.set_ylabel(y_label_1)
        ax.plot(x_fit, y_fit, color='k', lw=2.5, linestyle='--')
        ax.set_xlim(8.5, 13.1)
        ax.set_ylim(10.6, 15.2)
        ax.set_xlabel(x_label_1)
        ax.set_ylabel(y_label_1)
        ax.minorticks_on()

    for ax in axes[1]:
        ax.set_ylabel(y_label_2)
        ax.set_xlabel(x_label_2)
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                    bottom=True, top=True, left=True, right=True, direction='in')
        ax.set_xlim(8.5, 13.1)
        ax.set_ylim(-1, 1)
        ax.minorticks_on()
        ax.axhline(0, color='gray', linestyle='-', linewidth=1)

    # Plot SHARK data
    counts, xedges, yedges, im1 = axes[0][0].hist2d(SHARK['mstar_cen3'], SHARK['mvir_hosthalo'], bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[8.5, 13.1], [10.6, 15.2]])
    axes[0][0].set_title('SHARK')

    # Plot SAGE data
    counts, xedges, yedges, im2 = axes[0][1].hist2d(SAGE['M_Cen3'], SAGE['M_200'], bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[8.5, 13.1], [10.6, 15.2]])
    axes[0][1].set_title('SAGE')

    # Plot GAEA data
    counts, xedges, yedges, im3 = axes[0][2].hist2d(GAEA['P3M'], GAEA['Mhalo'], bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[8.5, 13.1], [10.6, 15.2]])
    axes[0][2].set_title('GAEA')

    # Add colorbars for top row only
    for i, im in enumerate([im1, im2, im3]):
        cbar = fig.colorbar(im, ax=axes[0, i], location='bottom', pad=0.02)
        cbar.set_label('log(N)')

    # Plot residuals for SHARK
    shark_centre_mhalo, shark_median_mhalo, shark_p16_mhalo, shark_p84_mhalo, shark_p2_5_mhalo, shark_p97_5_mhalo = running_stats(SHARK['mstar_cen3'], SHARK_Delta_HM, 0.2)
    shark_mean_delta = np.nanmean(shark_median_mhalo)
    shark_mean_uncertainty = np.nanmean((shark_p84_mhalo - shark_p16_mhalo)/2)
    axes[1][0].errorbar(shark_centre_mhalo, shark_median_mhalo, yerr=[shark_p84_mhalo - shark_median_mhalo, shark_median_mhalo - shark_p16_mhalo], fmt='o', color='k', label='SHARK Median', markersize=5, capsize=3)
    # Add text to SHARK plot
    axes[1][0].text(0.01, 0.35, f'Mean Δ = {shark_mean_delta:.2f}\nMean σ = {shark_mean_uncertainty:.2f}', transform=axes[1][0].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot residuals for SAGE
    sage_centre_mhalo, sage_median_mhalo, sage_p16_mhalo, sage_p84_mhalo, sage_p2_5_mhalo, sage_p97_5_mhalo = running_stats(SAGE['M_Cen3'], SAGE_Delta_HM, 0.2)
    sage_mean_delta = np.nanmean(sage_median_mhalo)
    sage_mean_uncertainty = np.nanmean((sage_p84_mhalo - sage_median_mhalo)/2)
    axes[1][1].errorbar(sage_centre_mhalo, sage_median_mhalo, yerr=[sage_p84_mhalo - sage_median_mhalo, sage_median_mhalo - sage_p16_mhalo], fmt='o', color='k', label='SAGE Median', markersize=5, capsize=3)
    # Add text to SAGE plot
    axes[1][1].text(0.01, 0.35, f'Mean Δ = {sage_mean_delta:.2f}\nMean σ = {sage_mean_uncertainty:.2f}', transform=axes[1][1].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot residuals for GAEA
    gaea_centre_mhalo, gaea_median_mhalo, gaea_p16_mhalo, gaea_p84_mhalo, gaea_p2_5_mhalo, gaea_p97_5_mhalo = running_stats(GAEA['P3M'], GAEA_Delta_HM, 0.2)
    gaea_mean_delta = np.nanmean(gaea_median_mhalo)
    gaea_mean_uncertainty = np.nanmean((gaea_p84_mhalo - gaea_median_mhalo)/2)
    axes[1][2].errorbar(gaea_centre_mhalo, gaea_median_mhalo, yerr=[gaea_p84_mhalo - gaea_median_mhalo, gaea_median_mhalo - gaea_p16_mhalo], fmt='o', color='k', label='GAEA Median', markersize=5, capsize=3)
    # Add text to GAEA plot
    axes[1][2].text(0.01, 0.35, f'Mean Δ = {gaea_mean_delta:.2f}\nMean σ = {gaea_mean_uncertainty:.2f}', transform=axes[1][2].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))   

    plt.savefig(f'/fred/oz004/wvankemp/Halo_Mass_New/Plots/{band}/SHMR_Comparison_{zlim}.png')
    plt.close()


    ####################################################################################################################################
    ############################# 3 panel plot w/ SAGE + GAEA - Mhalo vs Mhalo, 3 w/ residuals w/ residuals ############################
    ####################################################################################################################################

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True, height_ratios=[1, 0.3])
    fig.subplots_adjust(hspace=0.3, wspace=0)

    # Common labels
    x_label_1 = r'$\log$ M$_{\rm{halo}, 3}$ [M$_\odot$]'
    y_label_1 = r'$\log$ M$_{\rm{halo}}$ [M$_\odot$]'
    

    x_label_2 = r'$\log$ M$_{\rm{halo}, 3}$ [M$_\odot$]'
    y_label_2 = r'$\Delta$ $\log$ M$_{\rm{halo}}$ [M$_\odot$]'

    for ax in axes[0]:
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                    bottom=True, top=True, left=True, right=True, direction='in')
        ax.plot([9, 16], [9, 16], color='black', linestyle='--', linewidth=2.5)
        ax.set_xlim(10.6, 15.2)
        ax.set_ylim(10.6, 15.2);
        ax.set_xlabel(x_label_1)
        ax.set_ylabel(y_label_1)
        ax.minorticks_on()

    for ax in axes[1]:
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                    bottom=True, top=True, left=True, right=True, direction='in')
        ax.set_xlabel(x_label_2)
        ax.set_ylabel(y_label_2)
        ax.set_xlim(10.6, 15.2)
        ax.set_ylim(-1, 1)
        ax.minorticks_on()
        ax.axhline(0, color='gray', linestyle='-', linewidth=1)

    # Plot SHARK data
    counts, xedges, yedges, im1 = axes[0][0].hist2d(SHARK_HM_P3, SHARK['mvir_hosthalo'], bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[10.6, 15.2], [10.6, 15.2]])
    axes[0][0].set_title('SHARK')

    # Plot SAGE data
    counts, xedges, yedges, im2 = axes[0][1].hist2d(SAGE_HM_P3, SAGE['M_200'], bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[10.6, 15.2], [10.6, 15.2]])
    axes[0][1].set_title('SAGE')

    # Plot GAEA data
    counts, xedges, yedges, im3 = axes[0][2].hist2d(GAEA_HM_P3, GAEA['Mhalo'], bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[10.6, 15.2], [10.6, 15.2]])
    axes[0][2].set_title('GAEA')

    # Add colorbars for top row only
    for i, im in enumerate([im1, im2, im3]):
        cbar = fig.colorbar(im, ax=axes[0, i], location='bottom', pad=0.02)
        cbar.set_label('log(N)')

    # Plot residuals for SHARK
    # Only include SHARK_HM_P3 >= 10.6
    mask = SHARK_HM_P3 >= 10.6
    shark_centre_mhalo, shark_median_mhalo, shark_p16_mhalo, shark_p84_mhalo, shark_p2_5_mhalo, shark_p97_5_mhalo = running_stats(SHARK_HM_P3[mask], SHARK_Delta_HM[mask], 0.2)
    shark_mean_delta = np.nanmean(shark_median_mhalo)
    shark_mean_uncertainty = np.nanmean((shark_p84_mhalo - shark_p16_mhalo)/2)
    axes[1][0].errorbar(shark_centre_mhalo, shark_median_mhalo, yerr=[shark_p84_mhalo - shark_median_mhalo, shark_median_mhalo - shark_p16_mhalo], fmt='o', color='k', label='SHARK Median', markersize=5, capsize=3)
    # Add text to SHARK plot
    axes[1][0].text(0.01, 0.35, f'Mean Δ = {shark_mean_delta:.2f}\nMean σ = {shark_mean_uncertainty:.2f}', transform=axes[1][0].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot residuals for SAGE
    mask = SAGE_HM_P3 >= 10.6
    sage_centre_mhalo, sage_median_mhalo, sage_p16_mhalo, sage_p84_mhalo, sage_p2_5_mhalo, sage_p97_5_mhalo = running_stats(SAGE_HM_P3[mask], SAGE_Delta_HM[mask], 0.2)
    sage_mean_delta = np.nanmean(sage_median_mhalo)
    sage_mean_uncertainty = np.nanmean((sage_p84_mhalo - sage_median_mhalo)/2)
    axes[1][1].errorbar(sage_centre_mhalo, sage_median_mhalo, yerr=[sage_p84_mhalo - sage_median_mhalo, sage_median_mhalo - sage_p16_mhalo], fmt='o', color='k', label='SAGE Median', markersize=5, capsize=3)
    # Add text to SAGE plot
    axes[1][1].text(0.01, 0.35, f'Mean Δ = {sage_mean_delta:.2f}\nMean σ = {sage_mean_uncertainty:.2f}', transform=axes[1][1].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot residuals for GAEA
    mask = GAEA_HM_P3 >= 10.6
    gaea_centre_mhalo, gaea_median_mhalo, gaea_p16_mhalo, gaea_p84_mhalo, gaea_p2_5_mhalo, gaea_p97_5_mhalo = running_stats(GAEA_HM_P3[mask], GAEA_Delta_HM[mask], 0.2)
    gaea_mean_delta = np.nanmean(gaea_median_mhalo)
    gaea_mean_uncertainty = np.nanmean((gaea_p84_mhalo - gaea_median_mhalo)/2)
    axes[1][2].errorbar(gaea_centre_mhalo, gaea_median_mhalo, yerr=[gaea_p84_mhalo - gaea_median_mhalo, gaea_median_mhalo - gaea_p16_mhalo], fmt='o', color='k', label='GAEA Median', markersize=5, capsize=3)
    # Add text to GAEA plot
    axes[1][2].text(0.01, 0.35, f'Mean Δ = {gaea_mean_delta:.2f}\nMean σ = {gaea_mean_uncertainty:.2f}', transform=axes[1][2].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))   

    plt.savefig(f'/fred/oz004/wvankemp/Halo_Mass_New/Plots/{band}/Mhalo_Mhalo3_Comparison_{zlim}.png')
    plt.close()





# Load SAGE data (concatenate 10 files)
sage_data_list = []
for i in range(10):
    sage_file = f"/fred/oz004/wvankemp/Halo_Mass_New/Data/SAGE/data_group_catalog_{i}.fits"
    sage_data_list.append(pd.DataFrame(fits.getdata(sage_file)))
SAGE = pd.concat(sage_data_list, ignore_index=True)
SAGE = SAGE[(SAGE['vdisp_gap'] >= (SAGE['radius'] * 150 - 200)) & (SAGE['vdisp_gap'] <= (SAGE['radius'] * 150 + 400)) & (SAGE['radius'] < 12) & (SAGE['vdisp_gap'] < 1800)].reset_index(drop=True)
SAGE = SAGE.apply(lambda col: col.map(lambda x: x.real if np.iscomplexobj(x) else x))

# Load GAEA data
gaea_file = "/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Group_Catalog_1e8.fits"
GAEA = pd.DataFrame(fits.getdata(gaea_file))
GAEA = GAEA[(GAEA['vdisp_gap'] >= (GAEA['sep_max'] * 150 - 200)) & (GAEA['vdisp_gap'] <= (GAEA['sep_max'] * 150 + 400)) & (GAEA['sep_max'] < 12) & (GAEA['vdisp_gap'] < 1800)].reset_index(drop=True)
GAEA = GAEA[(GAEA['Mhalo'] <= 13.5) | ((GAEA['Mhalo'] > 13.5) & (GAEA['Mhalo'] - GAEA['Mhalo_VT'] < 1.2))].reset_index(drop=True)
GAEA = GAEA.apply(lambda col: col.map(lambda x: x.real if np.iscomplexobj(x) else x))




# MCMC_and_Plots(SAGE, GAEA, 'Z_band', '01')
# MCMC_and_Plots(SAGE, GAEA, 'Z_band', '02')
MCMC_and_Plots(SAGE, GAEA, 'Z_band', '03')

# MCMC_and_Plots(SAGE, GAEA, 'i_band', '01')
# MCMC_and_Plots(SAGE, GAEA, 'i_band', '02')
# MCMC_and_Plots(SAGE, GAEA, 'i_band', '03')

# MCMC_and_Plots(SAGE, GAEA, 'r_band', '01')
# MCMC_and_Plots(SAGE, GAEA, 'r_band', '02')
# MCMC_and_Plots(SAGE, GAEA, 'r_band', '03')
