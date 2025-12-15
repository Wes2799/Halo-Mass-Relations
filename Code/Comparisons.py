import numpy as np
from astropy.io import fits
import cmasher as cmr
import os
from astropy.table import vstack
import matplotlib.pyplot as plt
import pandas as pd  
from matplotlib.colors import LogNorm
from astropy import units as u
from astropy.constants import G
from matplotlib.ticker import MultipleLocator

G = G.to(u.M_sun**-1 * u.km**2 * u.s**-2 * u.Mpc).value

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

def Moster_SM2HM(Mhalo, N, M1, beta, gamma):
    Mhalo = 10**Mhalo  # Convert log values back to linear scale
    M1 = 10**M1  # Convert M1 to linear scale
    Mstar = Mhalo * 2 * N / ((Mhalo / M1)**(-beta) + (Mhalo / M1)**(gamma))
    return np.log10(Mstar)  # Convert back to log scale for fitting

def load_data():
    # Load SHARK v2.0 data
    shark_file = "/fred/oz004/wvankemp/Halo_Mass_Relations/Data/groups.fits"
    shark_data = pd.DataFrame(fits.getdata(shark_file))
    
    # Load SAGE data (concatenate 10 files)
    sage_data_list = []
    for i in range(10):
        sage_file = f"/fred/oz004/wvankemp/Halo_Mass_Relations/Data/SAGE/data_group_catalog_{i}.fits"
        if os.path.exists(sage_file):
            sage_data_list.append(pd.DataFrame(fits.getdata(sage_file)))
    sage_data = pd.concat(sage_data_list, ignore_index=True)
    sage_data = sage_data[(sage_data['radius'] < 5) & (sage_data['M_200_Disp_Correction'] > 10.5) & (sage_data['M_200_Disp_Correction'] < 16)].reset_index(drop=True)
    sage_data = sage_data.applymap(lambda x: x.real if np.iscomplexobj(x) else x)
    
    # Load GAEA data
    gaea_file = "/fred/oz004/wvankemp/Halo_Mass_Relations/Data/GAEA_sims/GAEA_Group_Catalog_1e8.fits"
    gaea_data = pd.DataFrame(fits.getdata(gaea_file))
    gaea_data = gaea_data[(gaea_data['n_members'] < 15) & (gaea_data['sep_max'] < 0.7) & (gaea_data['vdisp_gap'] < 1000) | 
                          (gaea_data['n_members'] >= 15) & (gaea_data['sep_max'] < 6) & (gaea_data['sep_max'] > 0.01) & (gaea_data['z_obs_median'] <= 0.1)].reset_index(drop=True)
    gaea_data = gaea_data.applymap(lambda x: x.real if np.iscomplexobj(x) else x)
    
    return shark_data, sage_data, gaea_data

def M200_Disp_Cor(vd, rad):
    alpha = 1.052
    vd_lim = 227.327
    n1 = -1.926
    beta = 0.326
    rad_lim = 0.320
    n2 = -1.378
    G = 4.301e-9  # Add missing G constant (Mpc (km/s)^2 / M_sun)
    
    # Replace if-else with np.where
    Ab = np.where(vd < vd_lim, alpha * ((vd / vd_lim)**n1 - 1), 0)
    Ac = np.where(rad < rad_lim, beta * ((rad / rad_lim)**n2 - 1), 0)
    
    A = 5/3 + Ab + Ac
    return np.log10(A * vd**2 * rad / G)

def S2HM(Mc, A, M_A, beta, gamma):
    Mc = 10**Mc  # Convert log values back to linear scale
    Mh = A * Mc * ((Mc / (10**M_A))**(beta) + (Mc / (10**M_A))**gamma)
    return np.log10(Mh)  # Convert back to log scale for fitting



def plot_figure1(shark_data, sage_data, gaea_data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    x_fit = np.linspace(7, 12.5, 1000)
    y_fit = S2HM(x_fit, 31.109, 10.768, 0.889, -0.555)

    # Common labels
    x_label = r'$\log$ $\Sigma$M$_{\star, \rm{3}}$ [M$_\odot$]'
    y_label = r'$\log$ M$_{\rm{halo}}$ [M$_\odot$]'
    
    # Set axis labels for all subplots
    for ax in axes:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(7.7, 12.8)
        ax.set_ylim(10.4, 15.4)
        ax.minorticks_on()
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=True, left=True, right=True, direction='in')
    
    # SHARK plot
    x_shark = shark_data['C3SM']
    y_shark = shark_data['mvir_hosthalo']
    counts, xedges, yedges, im1 = axes[0].hist2d(x_shark, y_shark, bins=100, cmap=cmr.iceburn, norm=LogNorm(), 
                                                range=[[7.7, 12.8], [10.4, 15.4]])
    axes[0].plot(x_fit, y_fit, color='black', linestyle='--', label='SHARK v2.0 Fit', linewidth=2.5)
    axes[0].set_title('SHARK')
    
    # SAGE plot
    x_sage = sage_data['M_Cen3']
    y_sage = sage_data['M_200']
    counts, xedges, yedges, im2 = axes[1].hist2d(x_sage, y_sage, bins=100, cmap=cmr.iceburn, norm=LogNorm(),
                                                range=[[7.7, 12.8], [10.4, 15.4]])
    axes[1].plot(x_fit, y_fit, color='black', linestyle='--', label='Shark v2.0 Fit', linewidth=2.5)
    axes[1].set_title('SAGE')
    
    # GAEA plot
    x_gaea = gaea_data['CSM3']
    y_gaea = gaea_data['Mhalo']
    counts, xedges, yedges, im3 = axes[2].hist2d(x_gaea, y_gaea, bins=100, cmap=cmr.iceburn, norm=LogNorm(),
                                                range=[[7.7, 12.8], [10.4, 15.4]])
    axes[2].plot(x_fit, y_fit, color='black', linestyle='--', label='Shark v2.0 Fit', linewidth=2.5)
    axes[2].set_title('GAEA')
    
    # Add colorbars
    for i, (im, ax) in enumerate(zip([im1, im2, im3], axes)):
        cbar = plt.colorbar(im, ax=ax, location='bottom', pad=0.02)
        cbar.set_label('log(N)')
    
    plt.savefig('/fred/oz004/wvankemp/Halo_Mass_Relations/Plots/C3HM_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_figure1_2(shark_data, sage_data, gaea_data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    x_fit = np.linspace(7, 12.5, 1000)
    y_fit = S2HM(x_fit, 39.788, 10.611, 0.926, -0.609)

    # Common labels
    x_label = r'$\log$ M$_{\star, \rm{Primary}}$ [M$_\odot$]'
    y_label = r'$\log$ M$_{\rm{halo}}$ [M$_\odot$]'
    
    # Set axis labels for all subplots
    for ax in axes:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(7.7, 12.8)
        ax.set_ylim(10.4, 15.4)
        ax.minorticks_on()
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=True, left=True, right=True, direction='in')
    
    # SHARK plot
    x_shark = shark_data['CSM']
    y_shark = shark_data['mvir_hosthalo']
    counts, xedges, yedges, im1 = axes[0].hist2d(x_shark, y_shark, bins=100, cmap=cmr.iceburn, norm=LogNorm(), 
                                                range=[[7.7, 12.8], [10.4, 15.4]])
    axes[0].plot(x_fit, y_fit, color='black', linestyle='--', label='SHARK v2.0 Fit', linewidth=2.5)
    axes[0].set_title('SHARK')
    
    # SAGE plot
    x_sage = sage_data['M_Cen']
    y_sage = sage_data['M_200']
    counts, xedges, yedges, im2 = axes[1].hist2d(x_sage, y_sage, bins=100, cmap=cmr.iceburn, norm=LogNorm(),
                                                range=[[7.7, 12.8], [10.4, 15.4]])
    axes[1].plot(x_fit, y_fit, color='black', linestyle='--', label='Shark v2.0 Fit', linewidth=2.5)
    axes[1].set_title('SAGE')
    
    # GAEA plot
    x_gaea = gaea_data['CSM']
    y_gaea = gaea_data['Mhalo']
    counts, xedges, yedges, im3 = axes[2].hist2d(x_gaea, y_gaea, bins=100, cmap=cmr.iceburn, norm=LogNorm(),
                                                range=[[7.7, 12.8], [10.4, 15.4]])
    axes[2].plot(x_fit, y_fit, color='black', linestyle='--', label='Shark v2.0 Fit', linewidth=2.5)
    axes[2].set_title('GAEA')
    
    # Add colorbars
    for i, (im, ax) in enumerate(zip([im1, im2, im3], axes)):
        cbar = plt.colorbar(im, ax=ax, location='bottom', pad=0.02)
        cbar.set_label('log(N)')
    
    plt.savefig('/fred/oz004/wvankemp/Halo_Mass_Relations/Plots/CM_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_figure1_3(shark_data, sage_data, gaea_data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # y_fit = np.linspace(7, 12.5, 1000)
    # x_fit = S2HM(y_fit, 31.109, 10.768, 0.889, -0.555)

    N = 0.0351
    M1 = 11.590
    beta = 1.376
    gamma = 0.608
    x_fit = np.linspace(10, 16, 1000)
    y_fit = Moster_SM2HM(x_fit, N, M1, beta, gamma)

    # Common labels
    y_label = r'$\log$ M$_{\star, \rm{Primary}}$ [M$_\odot$]'
    x_label = r'$\log$ M$_{\rm{halo}}$ [M$_\odot$]'
    
    # Set axis labels for all subplots
    for ax in axes:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(7.7, 12.8)
        ax.set_xlim(10.4, 15.4)
        ax.minorticks_on()
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=True, left=True, right=True, direction='in')
    
    # SHARK plot
    y_shark = shark_data['CSM']
    x_shark = shark_data['mvir_hosthalo']
    counts, xedges, yedges, im1 = axes[0].hist2d(x_shark, y_shark, bins=100, cmap=cmr.iceburn, norm=LogNorm(), 
                                                range=[[10.4, 15.4], [7.7, 12.8]])
    axes[0].plot(x_fit, y_fit, color='black', linestyle='--', label='SHARK v2.0 Fit', linewidth=2.5)
    axes[0].set_title('SHARK')
    
    # SAGE plot
    y_sage = sage_data['M_Cen']
    x_sage = sage_data['M_200']
    counts, xedges, yedges, im2 = axes[1].hist2d(x_sage, y_sage, bins=100, cmap=cmr.iceburn, norm=LogNorm(),
                                                range=[[10.4, 15.4], [7.7, 12.8]])
    axes[1].plot(x_fit, y_fit, color='black', linestyle='--', label='Shark v2.0 Fit', linewidth=2.5)
    axes[1].set_title('SAGE')
    
    # GAEA plot
    y_gaea = gaea_data['CSM']
    x_gaea = gaea_data['Mhalo']
    counts, xedges, yedges, im3 = axes[2].hist2d(x_gaea, y_gaea, bins=100, cmap=cmr.iceburn, norm=LogNorm(),
                                                range=[[10.4, 15.4], [7.7, 12.8]])
    axes[2].plot(x_fit, y_fit, color='black', linestyle='--', label='Shark v2.0 Fit', linewidth=2.5)
    axes[2].set_title('GAEA')
    
    # Add colorbars
    for i, (im, ax) in enumerate(zip([im1, im2, im3], axes)):
        cbar = plt.colorbar(im, ax=ax, location='bottom', pad=0.02)
        cbar.set_label('log(N)')
    
    plt.savefig('/fred/oz004/wvankemp/Halo_Mass_Relations/Plots/SM_HM_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_figure1_4(shark_data, sage_data, gaea_data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # y_fit = np.linspace(7, 12.5, 1000)
    # x_fit = S2HM(y_fit, 31.109, 10.768, 0.889, -0.555)
    # y_fit = np.log10(10**y_fit / 10**x_fit)  # Adjust y_fit for the new scale

    N = 0.0351
    M1 = 11.590
    beta = 1.376
    gamma = 0.608
    x_fit = np.linspace(10, 16, 1000)
    y_fit = Moster_SM2HM(x_fit, N, M1, beta, gamma)
    y_fit = np.log10(10**y_fit / 10**x_fit)  # Adjust y_fit for the new scale
    # print(y_fit)
    # print(x_fit)
    # Common labels
    y_label = r'$\log$ (M$_{\star}$/M${halo}$)'
    x_label = r'$\log$ M$_{\rm{halo}}$ [M$_\odot$]'
    
    # Set axis labels for all subplots
    for ax in axes:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(10.4, 15.4)
        ax.set_ylim(-0.8, -4.1)
        ax.minorticks_on()
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=True, left=True, right=True, direction='in')
    
    # SHARK plot
    x_shark = shark_data['mvir_hosthalo']
    y_shark = np.log10(10**shark_data['CSM'] / 10**x_shark) 
    # print(y_shark)
    counts, xedges, yedges, im1 = axes[0].hist2d(x_shark, y_shark, bins=100, cmap=cmr.iceburn, norm=LogNorm(), 
                                                range=[[10.4, 15.4], [-4.1, -0.8]])
    axes[0].plot(x_fit, y_fit, color='black', linestyle='--', label='SHARK v2.0 Fit', linewidth=2.5)
    axes[0].set_title('SHARK')
    
    # SAGE plot
    x_sage = sage_data['M_200']
    y_sage = np.log10(10**sage_data['M_Cen'] / 10**x_sage)
    counts, xedges, yedges, im2 = axes[1].hist2d(x_sage, y_sage, bins=100, cmap=cmr.iceburn, norm=LogNorm(),
                                                range=[[10.4, 15.4], [-4.1, -0.8]])
    axes[1].plot(x_fit, y_fit, color='black', linestyle='--', label='Shark v2.0 Fit', linewidth=2.5)
    axes[1].set_title('SAGE')
    
    # GAEA plot
    x_gaea = gaea_data['Mhalo']
    y_gaea = np.log10(10**gaea_data['CSM'] / 10**x_gaea)
    counts, xedges, yedges, im3 = axes[2].hist2d(x_gaea, y_gaea, bins=100, cmap=cmr.iceburn, norm=LogNorm(),
                                                range=[[10.4, 15.4], [-4.1, -0.8]])
    axes[2].plot(x_fit, y_fit, color='black', linestyle='--', label='Shark v2.0 Fit', linewidth=2.5)
    axes[2].set_title('GAEA')
    
    # Add colorbars
    for i, (im, ax) in enumerate(zip([im1, im2, im3], axes)):
        cbar = plt.colorbar(im, ax=ax, location='bottom', pad=0.02)
        cbar.set_label('log(N)')
    plt.savefig('/fred/oz004/wvankemp/Halo_Mass_Relations/Plots/SMHM_HM_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_figure2(shark_data, sage_data, gaea_data):
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), constrained_layout=True, height_ratios=[1, 0.3])
    fig.subplots_adjust(hspace=0.3)
    

    # Set axis labels for top row subplots
    for ax in axes[0]:
        ax.set_xlabel(r'$\log$ M$_{\rm{halo}, \rm{Calibrated} \, \rm{VT} }$ [M$_\odot$]')
        ax.set_ylabel(r'$\log$ M$_{\rm{halo}}$ [M$_\odot$]')
        ax.set_xlim(10.6, 15.2)
        ax.set_ylim(10.6, 15.2)
        ax.minorticks_on()
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=True, left=True, right=True, direction='in')
        ax.plot([9, 16], [9, 16], color='black', linestyle='--', linewidth=2.5)
    
    # Different settings for bottom row
    for ax in axes[1]:
        ax.set_xlabel(r'$\log$ M$_{\rm{halo}, {Calibrated} \, \rm{VT} }$ [M$_\odot$]')
        ax.set_ylabel(r'$\Delta$ $\log$ M$_{\rm{halo}}$ [M$_\odot$]')
        ax.set_xlim(10.6, 15.2)
        ax.set_ylim(-1, 1)
        ax.minorticks_on()
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=True, left=True, right=True, direction='in')
        ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    
    # SHARK plot
    x_shark = shark_data['Mhalo_Disp']
    y_shark = shark_data['mvir_hosthalo']
    delta_shark = shark_data['Mhalo_Disp'] - shark_data['mvir_hosthalo']
    shark_centre_mhalo, shark_median_mhalo, shark_p16_mhalo, shark_p84_mhalo, shark_p2_5_mhalo, shark_p97_5_mhalo = running_stats(x_shark, delta_shark, 0.2)
    counts, xedges, yedges, im1 = axes[0][0].hist2d(x_shark, y_shark, bins=100, cmap=cmr.iceburn, norm=LogNorm(), range=[[10.6, 15.2], [10.6, 15.2]])
    axes[0][0].set_title('SHARK')

    # SAGE plot
    x_sage = sage_data['M_200_Disp_Correction']
    y_sage = sage_data['M_200']
    delta_sage = sage_data['M_200_Disp_Correction'] - sage_data['M_200']
    sage_centre_mhalo, sage_median_mhalo, sage_p16_mhalo, sage_p84_mhalo, sage_p2_5_mhalo, sage_p97_5_mhalo = running_stats(x_sage, delta_sage, 0.2)
    counts, xedges, yedges, im2 = axes[0][1].hist2d(x_sage, y_sage, bins=100, cmap=cmr.iceburn, norm=LogNorm(), range=[[10.6, 15.2], [10.6, 15.2]])
    axes[0][1].set_title('SAGE')

    # GAEA plot
    x_gaea = gaea_data['Mhalo_Disp']
    y_gaea = gaea_data['Mhalo']
    delta_gaea = gaea_data['Mhalo_Disp'] - gaea_data['Mhalo']
    gaea_centre_mhalo, gaea_median_mhalo, gaea_p16_mhalo, gaea_p84_mhalo, gaea_p2_5_mhalo, gaea_p97_5_mhalo = running_stats(x_gaea, delta_gaea, 0.2)
    counts, xedges, yedges, im3 = axes[0][2].hist2d(x_gaea, y_gaea, bins=100, cmap=cmr.iceburn, norm=LogNorm(), range=[[10.6, 15.2], [10.6, 15.2]])
    axes[0][2].set_title('GAEA')
    
    # Add colorbars for top row only
    for i, im in enumerate([im1, im2, im3]):
        cbar = fig.colorbar(im, ax=axes[0, i], location='bottom', pad=0.02)
        cbar.set_label('log(N)')

    # Bottom row plots - error plots for delta values
    axes[1][0].errorbar(shark_centre_mhalo, shark_median_mhalo, yerr=[shark_p84_mhalo - shark_median_mhalo, shark_median_mhalo - shark_p16_mhalo],
                        fmt='o', color='k', label='SHARK Median', markersize=5, capsize=3)
    # Calculate mean delta and mean uncertainty for SHARK
    shark_mean_delta = np.nanmean(shark_median_mhalo)
    shark_mean_uncertainty = np.nanmean((shark_p84_mhalo - shark_p16_mhalo)/2)
    # Add text to SHARK plot
    axes[1][0].text(0.01, 0.95, f'Mean Δ = {shark_mean_delta:.2f}\nMean σ = {shark_mean_uncertainty:.2f}', 
                   transform=axes[1][0].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    axes[1][1].errorbar(sage_centre_mhalo, sage_median_mhalo, yerr=[sage_p84_mhalo - sage_median_mhalo, sage_median_mhalo - sage_p16_mhalo],
                        fmt='o', color='k', label='SAGE Median', markersize=5, capsize=3)      
    # Calculate mean delta and mean uncertainty for SAGE
    sage_mean_delta = np.nanmean(sage_median_mhalo)
    sage_mean_uncertainty = np.nanmean((sage_p84_mhalo - sage_p16_mhalo)/2)
    # Add text to SAGE plot
    axes[1][1].text(0.01, 0.95, f'Mean Δ = {sage_mean_delta:.2f}\nMean σ = {sage_mean_uncertainty:.2f}', 
                   transform=axes[1][1].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    axes[1][2].errorbar(gaea_centre_mhalo, gaea_median_mhalo, yerr=[gaea_p84_mhalo - gaea_median_mhalo, gaea_median_mhalo - gaea_p16_mhalo],
                        fmt='o', color='k', label='GAEA Median', markersize=5, capsize=3)
    # Calculate mean delta and mean uncertainty for GAEA
    gaea_mean_delta = np.nanmean(gaea_median_mhalo)
    gaea_mean_uncertainty = np.nanmean((gaea_p84_mhalo - gaea_p16_mhalo)/2)
    # Add text to GAEA plot
    axes[1][2].text(0.01, 0.95, f'Mean Δ = {gaea_mean_delta:.2f}\nMean σ = {gaea_mean_uncertainty:.2f}', 
                   transform=axes[1][2].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    print(sage_centre_mhalo)
    print(sage_median_mhalo)

    plt.savefig('/fred/oz004/wvankemp/Halo_Mass_Relations/Plots/Disp_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_figure3(shark_data, sage_data, gaea_data):
    fig, axes = plt.subplots(3, 3, figsize=(15, 8), constrained_layout=True, height_ratios=[1, 0.3, 0.3])
    fig.subplots_adjust(hspace=0.3)
    
    # Common labels
    x_label = r'$\log$ M$_{\rm{halo}, 3}$ [M$_\odot$]'
    y_label = r'$\log$ M$_{\rm{halo}}$ [M$_\odot$]'
    
    # Set axis labels for top row subplots
    for ax in axes[0]:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(10.6, 15.2)
        ax.set_ylim(10.6, 15.2)
        ax.minorticks_on()
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=True, left=True, right=True, direction='in')
        ax.plot([9, 16], [9, 16], color='black', linestyle='--', linewidth=2.5)
    
    # Different settings for middle row
    for ax in axes[1]:
        ax.set_xlabel(r'$\log$ M$_{\rm{halo}, 3}$ [M$_\odot$]')
        ax.set_ylabel(r'$\Delta$ $\log$ M$_{\rm{halo}}$ [M$_\odot$]')
        ax.set_xlim(10.6, 15.2)
        ax.set_ylim(-1, 1)
        ax.minorticks_on()
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=True, left=True, right=True, direction='in')
        ax.axhline(0, color='gray', linestyle='-', linewidth=1)

    # Different settings for bottom row
    for ax in axes[2]:
        ax.set_xlabel(r'$\log$ $\Sigma$M$_{\star, \rm3}$ [M$_\odot$]')
        ax.set_ylabel(r'$\Delta$ $\log$ M$_{\rm{halo}}$ [M$_\odot$]')
        ax.set_xlim(7.7, 12.8)
        ax.set_ylim(-1, 1)
        ax.minorticks_on()
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=True, left=True, right=True, direction='in')
        ax.axhline(0, color='gray', linestyle='-', linewidth=1)

    # SHARK plot
    x_shark = shark_data['Mhalo_P3M']
    y_shark = shark_data['mvir_hosthalo']
    delta_shark = shark_data['Mhalo_P3M'] - shark_data['mvir_hosthalo']
    shark_centre_mhalo, shark_median_mhalo, shark_p16_mhalo, shark_p84_mhalo, shark_p2_5_mhalo, shark_p97_5_mhalo = running_stats(x_shark, delta_shark, 0.2)
    shark_centre_P3M, shark_median_P3M, shark_p16_P3M, shark_p84_P3M, shark_p2_5_P3M, shark_p97_5_P3M = running_stats(shark_data['C3SM'], delta_shark, 0.2)
    counts, xedges, yedges, im1 = axes[0][0].hist2d(x_shark, y_shark, bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[10.6, 15.2], [10.6, 15.2]])
    axes[0][0].set_title('SHARK')
    
    # SAGE plot
    x_sage = sage_data['M_200_C3M']
    y_sage = sage_data['M_200']
    delta_sage = sage_data['M_200_C3M'] - sage_data['M_200']
    sage_centre_mhalo, sage_median_mhalo, sage_p16_mhalo, sage_p84_mhalo, sage_p2_5_mhalo, sage_p97_5_mhalo = running_stats(x_sage, delta_sage, 0.2)
    sage_centre_P3M, sage_median_P3M, sage_p16_P3M, sage_p84_P3M, sage_p2_5_P3M, sage_p97_5_P3M = running_stats(sage_data['M_Cen3'], delta_sage, 0.2)
    counts, xedges, yedges, im2 = axes[0][1].hist2d(x_sage, y_sage, bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[10.6, 15.2], [10.6, 15.2]])
    axes[0][1].set_title('SAGE')
    
    # GAEA plot
    x_gaea = gaea_data['Mhalo_P3M']
    y_gaea = gaea_data['Mhalo']
    delta_gaea = gaea_data['Mhalo_P3M'] - gaea_data['Mhalo']
    gaea_centre_mhalo, gaea_median_mhalo, gaea_p16_mhalo, gaea_p84_mhalo, gaea_p2_5_mhalo, gaea_p97_5_mhalo = running_stats(x_gaea, delta_gaea, 0.2)
    gaea_centre_P3M, gaea_median_P3M, gaea_p16_P3M, gaea_p84_P3M, gaea_p2_5_P3M, gaea_p97_5_P3M = running_stats(gaea_data['CSM3'], delta_gaea, 0.2)
    counts, xedges, yedges, im3 = axes[0][2].hist2d(x_gaea, y_gaea, bins=100, cmap=cmr.iceburn, 
                                                norm=LogNorm(), range=[[10.6, 15.2], [10.6, 15.2]])
    axes[0][2].set_title('GAEA')
    
    # Add colorbars for top row only
    for i, im in enumerate([im1, im2, im3]):
        cbar = fig.colorbar(im, ax=axes[0, i], location='bottom', pad=0.02)
        cbar.set_label('log(N)')
    
    # Middle row plots - error plots for delta values
    axes[1][0].errorbar(shark_centre_mhalo, shark_median_mhalo, yerr=[shark_p84_mhalo - shark_median_mhalo, shark_median_mhalo - shark_p16_mhalo],
                        fmt='o', color='k', label='SHARK Median', markersize=5, capsize=3)
    # Calculate mean delta and mean uncertainty for SHARK
    shark_mean_delta = np.nanmean(shark_median_mhalo)
    shark_mean_uncertainty = np.nanmean((shark_p84_mhalo - shark_p16_mhalo)/2)
    # Add text to SHARK plot
    axes[1][0].text(0.01, 0.35, f'Mean Δ = {shark_mean_delta:.2f}\nMean σ = {shark_mean_uncertainty:.2f}', 
                   transform=axes[1][0].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[1][1].errorbar(sage_centre_mhalo, sage_median_mhalo, yerr=[sage_p84_mhalo - sage_median_mhalo, sage_median_mhalo - sage_p16_mhalo],
                        fmt='o', color='k', label='SAGE Median', markersize=5, capsize=3)
    # Calculate mean delta and mean uncertainty for SAGE
    sage_mean_delta = np.nanmean(sage_median_mhalo)
    sage_mean_uncertainty = np.nanmean((sage_p84_mhalo - sage_p16_mhalo)/2)
    # Add text to SAGE plot
    axes[1][1].text(0.01, 0.35, f'Mean Δ = {sage_mean_delta:.2f}\nMean σ = {sage_mean_uncertainty:.2f}',
                   transform=axes[1][1].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[1][2].errorbar(gaea_centre_mhalo, gaea_median_mhalo, yerr=[gaea_p84_mhalo - gaea_median_mhalo, gaea_median_mhalo - gaea_p16_mhalo],
                        fmt='o', color='k', label='GAEA Median', markersize=5, capsize=3)
    # Calculate mean delta and mean uncertainty for GAEA
    gaea_mean_delta = np.nanmean(gaea_median_mhalo)
    gaea_mean_uncertainty = np.nanmean((gaea_p84_mhalo - gaea_p16_mhalo)/2)
    # Add text to GAEA plot
    axes[1][2].text(0.01, 0.35, f'Mean Δ = {gaea_mean_delta:.2f}\nMean σ = {gaea_mean_uncertainty:.2f}',
                     transform=axes[1][2].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Bottom row plots - error plots for delta values
    axes[2][0].errorbar(shark_centre_P3M, shark_median_P3M, yerr=[shark_p84_P3M - shark_median_P3M, shark_median_P3M - shark_p16_P3M],
                        fmt='o', color='k', label='SHARK Median', markersize=5, capsize=3)
    # Calculate mean delta and mean uncertainty for SHARK P3M
    shark_mean_delta_P3M = np.nanmean(shark_median_P3M)
    shark_mean_uncertainty_P3M = np.nanmean((shark_p84_P3M - shark_p16_P3M)/2)
    # Add text to SHARK P3M plot
    axes[2][0].text(0.01, 0.35, f'Mean Δ = {shark_mean_delta_P3M:.2f}\nMean σ = {shark_mean_uncertainty_P3M:.2f}',
                   transform=axes[2][0].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[2][1].errorbar(sage_centre_P3M, sage_median_P3M, yerr=[sage_p84_P3M - sage_median_P3M, sage_median_P3M - sage_p16_P3M],
                        fmt='o', color='k', label='SAGE Median', markersize=5, capsize=3)
    # Calculate mean delta and mean uncertainty for SAGE P3M
    sage_mean_delta_P3M = np.nanmean(sage_median_P3M)
    sage_mean_uncertainty_P3M = np.nanmean((sage_p84_P3M - sage_p16_P3M)/2)
    # Add text to SAGE P3M plot
    axes[2][1].text(0.01, 0.35, f'Mean Δ = {sage_mean_delta_P3M:.2f}\nMean σ = {sage_mean_uncertainty_P3M:.2f}',
                   transform=axes[2][1].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[2][2].errorbar(gaea_centre_P3M, gaea_median_P3M, yerr=[gaea_p84_P3M - gaea_median_P3M, gaea_median_P3M - gaea_p16_P3M],
                        fmt='o', color='k', label='GAEA Median', markersize=5, capsize=3)
    # Calculate mean delta and mean uncertainty for GAEA P3M
    gaea_mean_delta_P3M = np.nanmean(gaea_median_P3M)
    gaea_mean_uncertainty_P3M = np.nanmean((gaea_p84_P3M - gaea_p16_P3M)/2)
    # Add text to GAEA P3M plot
    axes[2][2].text(0.01, 0.35, f'Mean Δ = {gaea_mean_delta_P3M:.2f}\nMean σ = {gaea_mean_uncertainty_P3M:.2f}',
                     transform=axes[2][2].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.savefig('/fred/oz004/wvankemp/Halo_Mass_Relations/Plots/P3M_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_figure4(shark_data, sage_data, gaea_data):
    fig, axes = plt.subplots(2, 3, figsize=(15, 6), constrained_layout=True, height_ratios=[1, 0.3])
    fig.subplots_adjust(hspace=0.3)
    
    
    # Set axis labels for top row subplots
    for ax in axes[0]:
        ax.set_xlabel(r'$\log$ M$_{\rm{halo}, \rm{VT} }$ [M$_\odot$]')
        ax.set_ylabel(r'$\log$ M$_{\rm{halo}}$ [M$_\odot$]')
        ax.set_xlim(9.6, 15.2)
        ax.set_ylim(9.6, 15.2)
        ax.minorticks_on()
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=True, left=True, right=True, direction='in')
        ax.plot([9, 16], [9, 16], color='black', linestyle='--', linewidth=2.5)
    
    # Different settings for bottom row
    for ax in axes[1]:
        ax.set_xlabel(r'$\log$ M$_{\rm{halo}, \rm{VT} }$ [M$_\odot$]')
        ax.set_ylabel(r'$\Delta$ $\log$ M$_{\rm{halo}}$ [M$_\odot$]')
        ax.set_xlim(9.6, 15.2)
        ax.set_ylim(-1, 1)
        ax.minorticks_on()
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=True, left=True, right=True, direction='in')
        ax.axhline(0, color='gray', linestyle='-', linewidth=1)
    
    # SHARK plot
    x_shark = shark_data['mhalo_disp']
    y_shark = shark_data['mvir_hosthalo']
    delta_shark = shark_data['mhalo_disp'] - shark_data['mvir_hosthalo']
    shark_centre_mhalo, shark_median_mhalo, shark_p16_mhalo, shark_p84_mhalo, shark_p2_5_mhalo, shark_p97_5_mhalo = running_stats(x_shark, delta_shark, 0.2)
    counts, xedges, yedges, im1 = axes[0][0].hist2d(x_shark, y_shark, bins=100, cmap=cmr.iceburn, norm=LogNorm(), range=[[9.6, 15.2], [9.6, 15.2]])
    axes[0][0].set_title('SHARK')

    # SAGE plot
    x_sage = sage_data['M_200_Disp']
    y_sage = sage_data['M_200']
    delta_sage = sage_data['M_200_Disp'] - sage_data['M_200']
    sage_centre_mhalo, sage_median_mhalo, sage_p16_mhalo, sage_p84_mhalo, sage_p2_5_mhalo, sage_p97_5_mhalo = running_stats(x_sage, delta_sage, 0.2)
    counts, xedges, yedges, im2 = axes[0][1].hist2d(x_sage, y_sage, bins=100, cmap=cmr.iceburn, norm=LogNorm(), range=[[9.6, 15.2], [9.6, 15.2]])
    axes[0][1].set_title('SAGE')

    # GAEA plot
    x_gaea = np.log10(5/3 * gaea_data['vdisp_gap']**2 * gaea_data['sep_max'] / G)
    y_gaea = gaea_data['Mhalo']
    delta_gaea = x_gaea - y_gaea
    # Filter out invalid values
    valid = np.isfinite(x_gaea) & np.isfinite(y_gaea) & np.isfinite(delta_gaea)
    x_gaea = x_gaea[valid]
    y_gaea = y_gaea[valid]
    delta_gaea = delta_gaea[valid]
    gaea_centre_mhalo, gaea_median_mhalo, gaea_p16_mhalo, gaea_p84_mhalo, gaea_p2_5_mhalo, gaea_p97_5_mhalo = running_stats(x_gaea, delta_gaea, 0.2)
    counts, xedges, yedges, im3 = axes[0][2].hist2d(x_gaea, y_gaea, bins=100, cmap=cmr.iceburn, norm=LogNorm(), range=[[9.6, 15.2], [9.6, 15.2]])
    axes[0][2].set_title('GAEA')

    # Add colorbars for top row only
    for i, im in enumerate([im1, im2, im3]):
        cbar = fig.colorbar(im, ax=axes[0, i], location='bottom', pad=0.02)
        cbar.set_label('log(N)')

    # Bottom row plots - error plots for delta values
    axes[1][0].errorbar(shark_centre_mhalo, shark_median_mhalo, yerr=[shark_p84_mhalo - shark_median_mhalo, shark_median_mhalo - shark_p16_mhalo],
                        fmt='o', color='k', label='SHARK Median', markersize=5, capsize=3)
    # Calculate mean delta and mean uncertainty for SHARK
    shark_mean_delta = np.nanmean(shark_median_mhalo)
    shark_mean_uncertainty = np.nanmean((shark_p84_mhalo - shark_p16_mhalo)/2)
    # Add text to SHARK plot
    axes[1][0].text(0.01, 0.95, f'Mean Δ = {shark_mean_delta:.2f}\nMean σ = {shark_mean_uncertainty:.2f}',
                     transform=axes[1][0].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[1][1].errorbar(sage_centre_mhalo, sage_median_mhalo, yerr=[sage_p84_mhalo - sage_median_mhalo, sage_median_mhalo - sage_p16_mhalo],
                        fmt='o', color='k', label='SAGE Median', markersize=5, capsize=3)
    # Calculate mean delta and mean uncertainty for SAGE
    sage_mean_delta = np.nanmean(sage_median_mhalo)
    sage_mean_uncertainty = np.nanmean((sage_p84_mhalo - sage_p16_mhalo)/2)
    # Add text to SAGE plot
    axes[1][1].text(0.01, 0.95, f'Mean Δ = {sage_mean_delta:.2f}\nMean σ = {sage_mean_uncertainty:.2f}',
                     transform=axes[1][1].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[1][2].errorbar(gaea_centre_mhalo, gaea_median_mhalo, yerr=[gaea_p84_mhalo - gaea_median_mhalo, gaea_median_mhalo - gaea_p16_mhalo],
                        fmt='o', color='k', label='GAEA Median', markersize=5, capsize=3)
    # Calculate mean delta and mean uncertainty for GAEA
    gaea_mean_delta = np.nanmean(gaea_median_mhalo)
    gaea_mean_uncertainty = np.nanmean((gaea_p84_mhalo - gaea_p16_mhalo)/2)
    # Add text to GAEA plot
    axes[1][2].text(0.01, 0.95, f'Mean Δ = {gaea_mean_delta:.2f}\nMean σ = {gaea_mean_uncertainty:.2f}',
                     transform=axes[1][2].transAxes, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))       
    
    
    
    plt.savefig('/fred/oz004/wvankemp/Halo_Mass_Relations/Plots/Disp_Uncor_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize='large')
    plt.rc('axes', labelsize='large')
    plt.rc('axes', titlesize='large')
    plt.rc('ytick', labelsize='large')
    # Create output directory if it doesn't exist
    os.makedirs('/fred/oz004/wvankemp/Halo_Mass_Relations/Plots', exist_ok=True)
    
    # Load data
    shark_data, sage_data, gaea_data = load_data()
    shark_data['Mhalo_Disp'] = M200_Disp_Cor(shark_data['vdisp_gap'], shark_data['sep_max'])
    shark_data['Mhalo_P3M'] = S2HM(shark_data['C3SM'], 31.109, 10.768, 0.889, -0.555)
    # Create the plots
    plot_figure1(shark_data, sage_data, gaea_data)
    plot_figure1_2(shark_data, sage_data, gaea_data)
    plot_figure1_3(shark_data, sage_data, gaea_data)
    plot_figure1_4(shark_data, sage_data, gaea_data)
    plot_figure2(shark_data, sage_data, gaea_data)
    plot_figure3(shark_data, sage_data, gaea_data)
    plot_figure4(shark_data, sage_data, gaea_data)
    
if __name__ == "__main__":
    main()