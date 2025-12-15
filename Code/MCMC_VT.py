import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cmasher as cmr
import emcee
import corner
import astropy.units as u
import multiprocessing as mp
from scipy.optimize import curve_fit
from astropy.table import Table
from astropy.io import fits
from matplotlib.colors import LogNorm
from astropy.constants import G

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('axes', labelsize='large')
plt.rc('axes', titlesize='large')
plt.rc('ytick', labelsize='large')
plt.rc('legend', fontsize='x-small')

# Gravitational constant in units of Msolar^−1 km^2 s^−2 Mpc
G = G.to(u.M_sun**-1 * u.km**2 * u.s**-2 * u.Mpc)


def Mhalo_CVT(alpha, vd_lim, n1, beta, rad_lim, n2, vd, rad):    
    # Replace if-else with np.where
    Ab = np.where(vd < vd_lim, alpha * ((vd / vd_lim)**n1 - 1), 0)
    Ac = np.where(rad < rad_lim, beta * ((rad / rad_lim)**n2 - 1), 0)
    
    A = 5/3 + Ab + Ac
    return np.log10(A * vd**2 * rad / G)


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

# Define the function to fit
def model(A, vdisp, sep):
    return np.log10(A * vdisp**2 * sep / G.value)


def compute_A_power(vdisp, sep, alpha, vdisp_lim, beta, sep_lim, n1, n2):
    Ab = alpha * ((vdisp / vdisp_lim)**n1 - 1) if vdisp < vdisp_lim else 0
    Ac = beta * ((sep / sep_lim)**n2 - 1) if sep < sep_lim else 0
    return 5/3 + Ab + Ac  


# Log-likelihood function
def log_likelihood(theta, vdisp, sep, y, compute_A_func):
    alpha, vdisp_lim, beta, sep_lim, n1, n2 = theta
    A = np.array([compute_A_func(v, s, alpha, vdisp_lim, beta, sep_lim, n1, n2) for v, s in zip(vdisp, sep)])
    model_vals = model(A, vdisp, sep)
    residuals = y - model_vals
    sigma = np.std(residuals)  # Standard deviation of residuals as uncertainty
    chi = residuals / sigma
    return -0.5 * np.nansum(chi**2 + np.log(2 * np.pi * sigma**2))


# Log-prior function
def log_prior(theta):
    alpha, vdisp_lim, beta, sep_lim, n1, n2 = theta
    if 0 < alpha < 5 and 0 < vdisp_lim < 600 and 0 < beta < 2 and 0 < sep_lim < 1 and -5 < n1 < -0.1 and -5 < n2 < -0.1:
        return 0.0  # Uniform prior
    return -np.inf  # Log-prior is -inf outside bounds

# Posterior function (log-probability)
def log_probability(theta, vdisp, sep, y, compute_A_power):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, vdisp, sep, y, compute_A_power)


def MCMC_and_Plots(SAGE, GAEA, band, zlim):

    SHARK = Table.read(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/{band}/group_catalog_{zlim}.fits')
    SHARK = SHARK.to_pandas()
    # SHARK = SHARK[( (SHARK['mvir_hosthalo'] < 11.5) & (SHARK['vdisp_gap'] < 450) & (SHARK['rad'] < 0.3) ) |
    #             ( (SHARK['mvir_hosthalo'] >= 11.5) & (SHARK['mvir_hosthalo'] < 12) & (SHARK['vdisp_gap'] < 550) & (SHARK['rad'] < 0.5) ) |
    #             ( (SHARK['mvir_hosthalo'] >= 12) & (SHARK['mvir_hosthalo'] < 12.5) & (SHARK['vdisp_gap'] < 650) & (SHARK['rad'] < 0.8) ) |
    #             ( (SHARK['mvir_hosthalo'] >= 12.5) & (SHARK['mvir_hosthalo'] < 13) & (SHARK['vdisp_gap'] < 750) & (SHARK['rad'] < 1.0) ) |
    #             ( (SHARK['mvir_hosthalo'] >= 13) & (SHARK['mvir_hosthalo'] < 13.5) & (SHARK['vdisp_gap'] < 900) & (SHARK['rad'] < 1.6) ) |
    #             ( (SHARK['mvir_hosthalo'] >= 13.5) & (SHARK['mvir_hosthalo'] < 14) & (SHARK['vdisp_gap'] < 1000) & (SHARK['rad'] < 2.5) ) |
    #             ( (SHARK['mvir_hosthalo'] >= 14) & (SHARK['mvir_hosthalo'] < 14.5) & (SHARK['vdisp_gap'] < 1100) & (SHARK['rad'] < 3) ) |
    #             ( (SHARK['mvir_hosthalo'] >= 14.5) & (SHARK['vdisp_gap'] < 1400) & (SHARK['rad'] < 6) )].reset_index(drop=True)
    SHARK = SHARK[(SHARK['vdisp_gap'] >= (SHARK['rad'] * 150 - 200)) & (SHARK['vdisp_gap'] <= (SHARK['rad'] * 150 + 400)) & (SHARK['rad'] < 12) & (SHARK['vdisp_gap'] < 1800)].reset_index(drop=True)

    # Set up MCMC
    ndim = 6  # Number of parameters
    nwalkers = 40  # Number of walkers
    initial = [1, 250, 0.5, 0.5, -1.5, -1]  # Initial parameter guesses
    pos = initial + 1e-2 * np.random.randn(nwalkers, ndim)  # Random perturbations


    # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(SHARK['vdisp_gap'], SHARK['sep_max'], SHARK['mvir_hosthalo'], compute_A_power))
    # sampler.run_mcmc(pos, nsteps, progress=True)
    # samples = sampler.get_chain(discard=burnin, thin=50, flat=True)

    # MCMC sampling with convergence check using autocorrelation time
    max_steps = 2000000  # Maximum number of steps to attempt
    check_interval = 1000  # Check for convergence every 500 steps
    converged = False
    # Define the number of CPU cores to use
    N = mp.cpu_count()  # Automatically detect the number of CPU cores

    # Create a multiprocessing pool
    with mp.Pool(N) as pool:
    # Initialize the sampler with the pool
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(SHARK['vdisp_gap'], SHARK['rad'], SHARK['mvir_hosthalo'], compute_A_power), pool=pool)

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

    # Log-likelihoods power: -1346.9708364477467
    # AIC power: 2705.9416728954934
    # BIC power: 2746.397118321517

    # Compare log-likelihoods
    log_likelihood_power = log_likelihood(np.median(samples, axis=0), SHARK['vdisp_gap'], SHARK['rad'], SHARK['mvir_hosthalo'], compute_A_power)
    print(f"Log-likelihoods power: {log_likelihood_power}")
    AIC_power = 2 * ndim - 2 * log_likelihood_power
    print(f"AIC power: {AIC_power}")
    BIC_power = ndim * np.log(len(SHARK)) - 2 * log_likelihood_power
    print(f"BIC power: {BIC_power}")

    # Plot Log liklihood of vdisp_lim
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(sampler.get_log_prob(flat=True)[:, 1])
    # ax.set_xlabel('Step')
    # ax.set_ylabel(r'$\log$ likelihood')
    # plt.show()


    #Plot the corner plot

    fig = corner.corner(samples, labels=[r"$\alpha$", r"$\sigma$$_{lim}$", r"$\beta$", r"R$_{lim}$", r"$n_1$", r"$n_2$"],
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True,
                    title_fmt=".3f",
                    truths=np.median(samples, axis=0))
    plt.savefig(f'/fred/oz004/wvankemp/Halo_Mass_New/Plots/{band}/CVT_Corner_{zlim}.png', dpi=300, bbox_inches='tight')
    plt.show()



    # Plot the data of power:
    alpha, vdisp_lim, beta, sep_lim, n1, n2 = np.median(samples, axis=0)
    print(f"Best-fit parameters: alpha={alpha}, vdisp_lim={vdisp_lim}, beta={beta}, sep_lim={sep_lim}, n1={n1}, n2={n2}")
    SHARK['Mhalo_CVT'] = Mhalo_CVT(alpha, vdisp_lim, n1, beta, sep_lim, n2, SHARK['vdisp_gap'], SHARK['rad'])
    SHARK['Delta_Mhalo_VT'] = SHARK['mhalo_vt'] - SHARK['mvir_hosthalo']
    SHARK['Delta_Mhalo_CVT'] = SHARK['Mhalo_CVT'] - SHARK['mvir_hosthalo']


    SAGE['Mhalo_CVT'] = Mhalo_CVT(alpha, vdisp_lim, n1, beta, sep_lim, n2, SAGE['vdisp_gap'], SAGE['radius'])
    SAGE['Delta_Mhalo_VT'] = SAGE['M_200_Disp'] - SAGE['M_200']
    SAGE['Delta_Mhalo_CVT'] = SAGE['Mhalo_CVT'] - SAGE['M_200']


    GAEA['Mhalo_VT'] = np.log10( 5/3 * GAEA['vdisp_gap']**2 * GAEA['sep_max'] / G.value)
    GAEA['Mhalo_CVT'] = Mhalo_CVT(alpha, vdisp_lim, n1, beta, sep_lim, n2, GAEA['vdisp_gap'], GAEA['sep_max'])
    GAEA['Delta_Mhalo_VT'] = GAEA['Mhalo_VT'] - GAEA['Mhalo']
    GAEA['Delta_Mhalo_CVT'] = GAEA['Mhalo_CVT'] - GAEA['Mhalo']


    ####################################################################################################################################
    ################################### 3 panel plot w/ SAGE + GAEA - Mhalo vs Mhalo_VT w/ residuals  ##################################
    ####################################################################################################################################

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True, height_ratios=[1, 0.3])
    fig.subplots_adjust(hspace=0.3, wspace=0)

    # Common labels
    x_label_1 = r'$\log$ M$_{\rm{halo}, \rm{VT}}$ [M$_\odot$]'
    y_label_1 = r'$\log$ M$_{\rm{halo}}$ [M$_\odot$]'
    

    x_label_2 = r'$\log$ M$_{\rm{halo}, \rm{VT}}$ [M$_\odot$]'
    y_label_2 = r'$\Delta$ $\log$ M$_{\rm{halo}}$ [M$_\odot$]'

    for ax in axes[0]:
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                    bottom=True, top=True, left=True, right=True, direction='in')
        ax.plot([9, 16], [9, 16], color='black', linestyle='--', linewidth=2.5)
        ax.set_xlim(9.6, 15.2)
        ax.set_ylim(9.6, 15.2);
        ax.set_xlabel(x_label_1)
        ax.set_ylabel(y_label_1)
        ax.minorticks_on()

    for ax in axes[1]:
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                    bottom=True, top=True, left=True, right=True, direction='in')
        ax.set_xlabel(x_label_2)
        ax.set_ylabel(y_label_2)
        ax.set_xlim(9.6, 15.2)
        ax.set_ylim(-1, 1)
        ax.minorticks_on()
        ax.axhline(0, color='gray', linestyle='-', linewidth=1)

    # Plot SHARK data
    counts, xedges, yedges, im1 = axes[0][0].hist2d(SHARK['mhalo_vt'], SHARK['mvir_hosthalo'], bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[9.6, 15.2], [9.6, 15.2]])
    axes[0][0].set_title('SHARK')

    # Plot SAGE data
    counts, xedges, yedges, im2 = axes[0][1].hist2d(SAGE['M_200_Disp'], SAGE['M_200'], bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[9.6, 15.2], [9.6, 15.2]])
    axes[0][1].set_title('SAGE')

    # Plot GAEA data
    counts, xedges, yedges, im3 = axes[0][2].hist2d(GAEA['Mhalo_VT'], GAEA['Mhalo'], bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[9.6, 15.2], [9.6, 15.2]])
    axes[0][2].set_title('GAEA')

    # Add colorbars for top row only
    for i, im in enumerate([im1, im2, im3]):
        cbar = fig.colorbar(im, ax=axes[0, i], location='bottom', pad=0.02)
        cbar.set_label('log(N)')

    # Plot residuals for SHARK
    shark_centre_mhalo, shark_median_mhalo, shark_p16_mhalo, shark_p84_mhalo, shark_p2_5_mhalo, shark_p97_5_mhalo = running_stats(SHARK['mhalo_vt'], SHARK['Delta_Mhalo_VT'], 0.2)
    shark_mean_delta = np.nanmean(shark_median_mhalo)
    shark_mean_uncertainty = np.nanmean((shark_p84_mhalo - shark_p16_mhalo)/2)
    axes[1][0].errorbar(shark_centre_mhalo, shark_median_mhalo, yerr=[shark_p84_mhalo - shark_median_mhalo, shark_median_mhalo - shark_p16_mhalo], fmt='o', color='k', label='SHARK Median', markersize=5, capsize=3)
    # Add text to SHARK plot
    axes[1][0].text(0.01, 0.95, f'Mean Δ = {shark_mean_delta:.2f}\nMean σ = {shark_mean_uncertainty:.2f}', transform=axes[1][0].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot residuals for SAGE
    sage_centre_mhalo, sage_median_mhalo, sage_p16_mhalo, sage_p84_mhalo, sage_p2_5_mhalo, sage_p97_5_mhalo = running_stats(SAGE['M_200_Disp'], SAGE['Delta_Mhalo_VT'], 0.2)
    sage_mean_delta = np.nanmean(sage_median_mhalo)
    sage_mean_uncertainty = np.nanmean((sage_p84_mhalo - sage_median_mhalo)/2)
    axes[1][1].errorbar(sage_centre_mhalo, sage_median_mhalo, yerr=[sage_p84_mhalo - sage_median_mhalo, sage_median_mhalo - sage_p16_mhalo], fmt='o', color='k', label='SAGE Median', markersize=5, capsize=3)
    # Add text to SAGE plot
    axes[1][1].text(0.01, 0.95, f'Mean Δ = {sage_mean_delta:.2f}\nMean σ = {sage_mean_uncertainty:.2f}', transform=axes[1][1].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot residuals for GAEA
    gaea_centre_mhalo, gaea_median_mhalo, gaea_p16_mhalo, gaea_p84_mhalo, gaea_p2_5_mhalo, gaea_p97_5_mhalo = running_stats(GAEA['Mhalo_VT'], GAEA['Delta_Mhalo_VT'], 0.2)
    gaea_mean_delta = np.nanmean(gaea_median_mhalo)
    gaea_mean_uncertainty = np.nanmean((gaea_p84_mhalo - gaea_median_mhalo)/2)
    axes[1][2].errorbar(gaea_centre_mhalo, gaea_median_mhalo, yerr=[gaea_p84_mhalo - gaea_median_mhalo, gaea_median_mhalo - gaea_p16_mhalo], fmt='o', color='k', label='GAEA Median', markersize=5, capsize=3)
    # Add text to GAEA plot
    axes[1][2].text(0.01, 0.95, f'Mean Δ = {gaea_mean_delta:.2f}\nMean σ = {gaea_mean_uncertainty:.2f}', transform=axes[1][2].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.savefig(f'/fred/oz004/wvankemp/Halo_Mass_New/Plots/{band}/VT_Comparison_{zlim}.png')
    plt.close()



    ####################################################################################################################################
    ################################### 3 panel plot w/ SAGE + GAEA - Mhalo vs Mhalo_CVT w/ residuals  #################################
    ####################################################################################################################################

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True, height_ratios=[1, 0.3])
    fig.subplots_adjust(hspace=0.3, wspace=0)

    # Common labels
    x_label_1 = r'$\log$ M$_{\rm{halo}, \rm{MVT}}$ [M$_\odot$]'
    y_label_1 = r'$\log$ M$_{\rm{halo}}$ [M$_\odot$]'
    

    x_label_2 = r'$\log$ M$_{\rm{halo}, \rm{MVT}}$ [M$_\odot$]'
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
    counts, xedges, yedges, im1 = axes[0][0].hist2d(SHARK['Mhalo_CVT'], SHARK['mvir_hosthalo'], bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[10.6, 15.2], [10.6, 15.2]])
    axes[0][0].set_title('SHARK')

    # Plot SAGE data
    counts, xedges, yedges, im2 = axes[0][1].hist2d(SAGE['Mhalo_CVT'], SAGE['M_200'], bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[10.6, 15.2], [10.6, 15.2]])
    axes[0][1].set_title('SAGE')

    # Plot GAEA data
    counts, xedges, yedges, im3 = axes[0][2].hist2d(GAEA['Mhalo_CVT'], GAEA['Mhalo'], bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[10.6, 15.2], [10.6, 15.2]])
    axes[0][2].set_title('GAEA')

    # Add colorbars for top row only
    for i, im in enumerate([im1, im2, im3]):
        cbar = fig.colorbar(im, ax=axes[0, i], location='bottom', pad=0.02)
        cbar.set_label('log(N)')

    # Plot residuals for SHARK
    mask = SHARK['Mhalo_CVT'] >= 10.6
    shark_centre_mhalo, shark_median_mhalo, shark_p16_mhalo, shark_p84_mhalo, shark_p2_5_mhalo, shark_p97_5_mhalo = running_stats(SHARK['Mhalo_CVT'][mask], SHARK['Delta_Mhalo_CVT'][mask], 0.2)
    shark_mean_delta = np.nanmean(shark_median_mhalo)
    shark_mean_uncertainty = np.nanmean((shark_p84_mhalo - shark_p16_mhalo)/2)
    axes[1][0].errorbar(shark_centre_mhalo, shark_median_mhalo, yerr=[shark_p84_mhalo - shark_median_mhalo, shark_median_mhalo - shark_p16_mhalo], fmt='o', color='k', label='SHARK Median', markersize=5, capsize=3)
    # Add text to SHARK plot
    axes[1][0].text(0.01, 0.95, f'Mean Δ = {shark_mean_delta:.2f}\nMean σ = {shark_mean_uncertainty:.2f}', transform=axes[1][0].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot residuals for SAGE
    mask = SAGE['Mhalo_CVT'] >= 10.6
    sage_centre_mhalo, sage_median_mhalo, sage_p16_mhalo, sage_p84_mhalo, sage_p2_5_mhalo, sage_p97_5_mhalo = running_stats(SAGE['Mhalo_CVT'][mask], SAGE['Delta_Mhalo_CVT'][mask], 0.2)
    sage_mean_delta = np.nanmean(sage_median_mhalo)
    sage_mean_uncertainty = np.nanmean((sage_p84_mhalo - sage_median_mhalo)/2)
    axes[1][1].errorbar(sage_centre_mhalo, sage_median_mhalo, yerr=[sage_p84_mhalo - sage_median_mhalo, sage_median_mhalo - sage_p16_mhalo], fmt='o', color='k', label='SAGE Median', markersize=5, capsize=3)
    # Add text to SAGE plot
    axes[1][1].text(0.01, 0.95, f'Mean Δ = {sage_mean_delta:.2f}\nMean σ = {sage_mean_uncertainty:.2f}', transform=axes[1][1].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot residuals for GAEA
    mask = GAEA['Mhalo_CVT'] >= 10.6
    gaea_centre_mhalo, gaea_median_mhalo, gaea_p16_mhalo, gaea_p84_mhalo, gaea_p2_5_mhalo, gaea_p97_5_mhalo = running_stats(GAEA['Mhalo_CVT'][mask], GAEA['Delta_Mhalo_CVT'][mask], 0.2)
    gaea_mean_delta = np.nanmean(gaea_median_mhalo)
    gaea_mean_uncertainty = np.nanmean((gaea_p84_mhalo - gaea_median_mhalo)/2)
    axes[1][2].errorbar(gaea_centre_mhalo, gaea_median_mhalo, yerr=[gaea_p84_mhalo - gaea_median_mhalo, gaea_median_mhalo - gaea_p16_mhalo], fmt='o', color='k', label='GAEA Median', markersize=5, capsize=3)
    # Add text to GAEA plot
    axes[1][2].text(0.01, 0.95, f'Mean Δ = {gaea_mean_delta:.2f}\nMean σ = {gaea_mean_uncertainty:.2f}', transform=axes[1][2].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))   

    plt.savefig(f'/fred/oz004/wvankemp/Halo_Mass_New/Plots/{band}/CVT_Comparison_{zlim}.png')
    plt.close()






# Load SAGE data (concatenate 10 files)
sage_data_list = []
for i in range(10):
    sage_file = f"/fred/oz004/wvankemp/Halo_Mass_New/Data/SAGE/data_group_catalog_{i}.fits"
    sage_data_list.append(pd.DataFrame(fits.getdata(sage_file)))
SAGE = pd.concat(sage_data_list, ignore_index=True)
# SAGE = SAGE[( (SAGE['M_200'] < 11.5) & (SAGE['vdisp_gap'] < 450) & (SAGE['radius'] < 0.3) ) |
#             ( (SAGE['M_200'] >= 11.5) & (SAGE['M_200'] < 12) & (SAGE['vdisp_gap'] < 550) & (SAGE['radius'] < 0.5) ) |
#             ( (SAGE['M_200'] >= 12) & (SAGE['M_200'] < 12.5) & (SAGE['vdisp_gap'] < 650) & (SAGE['radius'] < 0.8) ) |
#             ( (SAGE['M_200'] >= 12.5) & (SAGE['M_200'] < 13) & (SAGE['vdisp_gap'] < 750) & (SAGE['radius'] < 1.0) ) |
#             ( (SAGE['M_200'] >= 13) & (SAGE['M_200'] < 13.5) & (SAGE['vdisp_gap'] < 900) & (SAGE['radius'] < 1.6) ) |
#             ( (SAGE['M_200'] >= 13.5) & (SAGE['M_200'] < 14) & (SAGE['vdisp_gap'] < 1000) & (SAGE['radius'] < 2.5) ) |
#             ( (SAGE['M_200'] >= 14) & (SAGE['M_200'] < 14.5) & (SAGE['vdisp_gap'] < 1100) & (SAGE['radius'] < 3) ) |
#             ( (SAGE['M_200'] >= 14.5) & (SAGE['vdisp_gap'] < 1400) & (SAGE['radius'] < 6) )].reset_index(drop=True)
SAGE = SAGE[(SAGE['vdisp_gap'] >= (SAGE['radius'] * 150 - 200)) & (SAGE['vdisp_gap'] <= (SAGE['radius'] * 150 + 400)) & (SAGE['radius'] < 12) & (SAGE['vdisp_gap'] < 1800)].reset_index(drop=True)
SAGE = SAGE.apply(lambda col: col.map(lambda x: x.real if np.iscomplexobj(x) else x))

# Load GAEA data
gaea_file = "/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Group_Catalog_1e8.fits"
GAEA = pd.DataFrame(fits.getdata(gaea_file))
# GAEA = GAEA[( (GAEA['Mhalo'] < 11.5) & (GAEA['vdisp_gap'] < 450) & (GAEA['sep_max'] < 0.3) ) |
#             ( (GAEA['Mhalo'] >= 11.5) & (GAEA['Mhalo'] < 12) & (GAEA['vdisp_gap'] < 550) & (GAEA['sep_max'] < 0.5) ) |
#             ( (GAEA['Mhalo'] >= 12) & (GAEA['Mhalo'] < 12.5) & (GAEA['vdisp_gap'] < 650) & (GAEA['sep_max'] < 0.8) ) |
#             ( (GAEA['Mhalo'] >= 12.5) & (GAEA['Mhalo'] < 13) & (GAEA['vdisp_gap'] < 750) & (GAEA['sep_max'] < 1.0) ) |
#             ( (GAEA['Mhalo'] >= 13) & (GAEA['Mhalo'] < 13.5) & (GAEA['vdisp_gap'] < 900) & (GAEA['sep_max'] < 1.6) ) |
#             ( (GAEA['Mhalo'] >= 13.5) & (GAEA['Mhalo'] < 14) & (GAEA['vdisp_gap'] < 1000) & (GAEA['sep_max'] < 2.5) ) |
#             ( (GAEA['Mhalo'] >= 14) & (GAEA['Mhalo'] < 14.5) & (GAEA['vdisp_gap'] < 1100) & (GAEA['sep_max'] < 3) ) |
#             ( (GAEA['Mhalo'] >= 14.5) & (GAEA['vdisp_gap'] < 1400) & (GAEA['sep_max'] < 6) )].reset_index(drop=True)


GAEA = GAEA[(GAEA['vdisp_gap'] >= (GAEA['sep_max'] * 150 - 200)) & (GAEA['vdisp_gap'] <= (GAEA['sep_max'] * 150 + 400)) & (GAEA['sep_max'] < 12) & (GAEA['vdisp_gap'] < 1800)].reset_index(drop=True)
GAEA = GAEA[(GAEA['Mhalo'] <= 13.5) | ((GAEA['Mhalo'] > 13.5) & (GAEA['Mhalo'] - GAEA['Mhalo_VT'] < 1.2))].reset_index(drop=True)

# Convert complex numbers to real numbers if necessary
GAEA = GAEA.apply(lambda col: col.map(lambda x: x.real if np.iscomplexobj(x) else x))


MCMC_and_Plots(SAGE, GAEA, 'Z_band', '01')
MCMC_and_Plots(SAGE, GAEA, 'Z_band', '02')
MCMC_and_Plots(SAGE, GAEA, 'Z_band', '03')


MCMC_and_Plots(SAGE, GAEA, 'i_band', '01')
MCMC_and_Plots(SAGE, GAEA, 'i_band', '02')
MCMC_and_Plots(SAGE, GAEA, 'i_band', '03')



MCMC_and_Plots(SAGE, GAEA, 'r_band', '01')
MCMC_and_Plots(SAGE, GAEA, 'r_band', '02')
MCMC_and_Plots(SAGE, GAEA, 'r_band', '03')