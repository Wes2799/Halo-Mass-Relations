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

G = G.to(u.M_sun**-1 * u.km**2 * u.s**-2 * u.Mpc)

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

def MVT(vd, rad):
    alpha = 1.030
    vd_lim = 244.634
    n1 = -1.989
    beta = 0.213
    rad_lim = 0.369 # has h dependence
    n2 = -1.591
    Ab = np.where(vd < vd_lim, alpha * ((vd / vd_lim)**n1 - 1), 0)
    Ac = np.where(rad < rad_lim, beta * ((rad / rad_lim)**n2 - 1), 0)
    A = 5/3 + Ab + Ac
    return np.log10(A * vd**2 * rad / G)

def plot(SAGE, GAEA, band, zlim):
    SHARK = Table.read(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/{band}/group_catalog_{zlim}.fits')
    SHARK = SHARK.to_pandas()
    SHARK = SHARK[(SHARK['vdisp_gap'] >= (SHARK['rad'] * 150 - 200)) & (SHARK['vdisp_gap'] <= (SHARK['rad'] * 150 + 400)) & (SHARK['rad'] < 12) & (SHARK['vdisp_gap'] < 1800)].reset_index(drop=True)

    SHARK['Mhalo_CVT'] = MVT(SHARK['vdisp_gap'], SHARK['rad'])
    SHARK['Delta_Mhalo_CVT'] = SHARK['Mhalo_CVT'] - SHARK['mvir_hosthalo']

    SAGE['Mhalo_CVT'] = MVT(SAGE['vdisp_gap'], SAGE['radius'])
    SAGE['Delta_Mhalo_CVT'] = SAGE['Mhalo_CVT'] - SAGE['M_200']

    GAEA['Mhalo_CVT'] = MVT(GAEA['vdisp_gap'], GAEA['sep_max'])
    GAEA['Delta_Mhalo_CVT'] = GAEA['Mhalo_CVT'] - GAEA['Mhalo']

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

    plt.savefig(f'/fred/oz004/wvankemp/Halo_Mass_New/Plots/{band}/CVT_Comparison_{zlim}.png', dpi=300, bbox_inches='tight')
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

# Convert complex numbers to real numbers if necessary
GAEA = GAEA.apply(lambda col: col.map(lambda x: x.real if np.iscomplexobj(x) else x))


plot(SAGE, GAEA, 'Z_band', '01')
plot(SAGE, GAEA, 'Z_band', '02')
plot(SAGE, GAEA, 'Z_band', '03')


plot(SAGE, GAEA, 'i_band', '01')
plot(SAGE, GAEA, 'i_band', '02')
plot(SAGE, GAEA, 'i_band', '03')


plot(SAGE, GAEA, 'r_band', '01')
plot(SAGE, GAEA, 'r_band', '02')
plot(SAGE, GAEA, 'r_band', '03')