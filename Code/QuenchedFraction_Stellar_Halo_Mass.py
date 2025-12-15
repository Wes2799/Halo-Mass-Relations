import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from astropy.table import Table
from astropy.io import fits
from scipy.special import erf
from astropy.constants import c
from astropy import units as u 
from tqdm import tqdm

def add_ssfr(sfr, sm):
    ssfr = sfr - sm
    return ssfr

# def kde_2d(xdata, ydata, xgrid, ygrid, xkernel_fwhm, ykernel_fwhm=None, weights = None ):
#     if ykernel_fwhm is None:
#         ykernel_fwhm = xkernel_fwhm
    
#     xsigma = xkernel_fwhm / (2. * np.sqrt(2. * np.log(2.)))
#     ysigma = ykernel_fwhm / (2. * np.sqrt(2. * np.log(2.)))


#     # xdata and xgrid are (assumed to be) both 1d vectors
#     dx = np.asarray(xdata)[:, None, None] - xgrid[None, :, None]
#     dy = np.asarray(ydata)[:, None, None] - ygrid[None, None, :]
#     # dx is a 3d arrays with shape = (xdata.size, xgrid.size, 1)
#     # dy is a 3d arrays with shape = (ydata.size, 1, ygrid.size)
    
#     w = np.exp( -0.5 * ( (dx/xsigma)**2. + (dy/ysigma)**2. )
#                 ) / ( 2.*np.pi*xsigma*ysigma )
#     if weights is not None:
#         w *= np.asarray(weights)[:, None, None]
#     # this is still a 3d array; result is sum over data points
#     result = np.sum(w, axis=0)
#     return result


def kde_2d(xdata, ydata, xgrid, ygrid, xkernel_fwhm, ykernel_fwhm=None, weights=None):
    if ykernel_fwhm is None:
        ykernel_fwhm = xkernel_fwhm

    xsigma = xkernel_fwhm / (2. * np.sqrt(2. * np.log(2.)))
    ysigma = ykernel_fwhm / (2. * np.sqrt(2. * np.log(2.)))
    norm = 2. * np.pi * xsigma * ysigma

    xdata = np.asarray(xdata)
    ydata = np.asarray(ydata)
    if weights is not None:
        weights = np.asarray(weights)
    else:
        weights = np.ones_like(xdata)

    result = np.zeros((len(xgrid), len(ygrid)))
    for i, x in enumerate(tqdm(xgrid, desc="KDE 2D Progress")):
        dx = (xdata - x) / xsigma
        for j, y in enumerate(ygrid):
            dy = (ydata - y) / ysigma
            w = np.exp(-0.5 * (dx**2 + dy**2)) * weights
            result[i, j] = np.sum(w) / norm
    return result


def Quenched_Stats(ssfr, sm, sfr):
    quenched_indices = np.where((ssfr < -11) | np.isneginf(ssfr) | np.isnan(ssfr))[0]
    quenched_SM = sm[quenched_indices].reset_index(drop=True)
    not_quenched_indices = np.where(ssfr >= -11)[0]
    quenched_SFR = sfr[quenched_indices].reset_index(drop=True)
    not_quenched_SM = sm[not_quenched_indices].reset_index(drop=True)
    not_quenched_SFR = sfr[not_quenched_indices].reset_index(drop=True)
    boot = np.isin(np.arange(len(ssfr)), quenched_indices)
    boot_2 = np.isin(np.arange(len(ssfr)), quenched_indices).astype(int)
    return (quenched_SM, quenched_SFR, not_quenched_SM, not_quenched_SFR, boot, boot_2)

print("Loading data...")

SHARK = Table.read(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/{band}/group_catalog_{zlim}.fits')
SHARK = SHARK.to_pandas()
SHARK = SHARK[( (SHARK['mvir_hosthalo'] < 13) & (SHARK['vdisp_gap'] < 650) & (SHARK['rad'] < 1.5) ) |
            ( (SHARK['mvir_hosthalo'] >= 13) & (SHARK['mvir_hosthalo'] < 14) & (SHARK['vdisp_gap'] < 900) & (SHARK['rad'] < 4) ) |
            ( (SHARK['mvir_hosthalo'] >= 14) & (SHARK['vdisp_gap'] < 1400) & (SHARK['rad'] < 12) )].reset_index(drop=True)


# Load SAGE data (concatenate 10 files)
sage_data_list = []
for i in range(10):
    sage_file = f"/fred/oz004/wvankemp/Halo_Mass_New/Data/SAGE/data_group_catalog_{i}.fits"
    sage_data_list.append(pd.DataFrame(fits.getdata(sage_file)))
SAGE = pd.concat(sage_data_list, ignore_index=True)
SAGE = SAGE[( (SAGE['M_200'] < 13) & (SAGE['vdisp_gap'] < 650) & (SAGE['radius'] < 1.5) ) |
            ( (SAGE['M_200'] >= 13) & (SAGE['M_200'] < 14) & (SAGE['vdisp_gap'] < 900) & (SAGE['radius'] < 4) ) |
            ( (SAGE['M_200'] >= 14) & (SAGE['vdisp_gap'] < 1400) & (SAGE['radius'] < 12) )].reset_index(drop=True)
SAGE = SAGE.apply(lambda col: col.map(lambda x: x.real if np.iscomplexobj(x) else x))

# Load GAEA data
gaea_file = "/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Group_Catalog_1e8.fits"
GAEA = pd.DataFrame(fits.getdata(gaea_file))
GAEA = GAEA[( (GAEA['Mhalo'] < 13) & (GAEA['vdisp_gap'] < 650) & (GAEA['sep_max'] < 1.5) & (GAEA['z_obs_median'] < 0.1) ) |
            ( (GAEA['Mhalo'] >= 13) & (GAEA['Mhalo'] < 14) & (GAEA['vdisp_gap'] < 900) & (GAEA['sep_max'] < 4) & (GAEA['z_obs_median'] < 0.1) ) |
            ( (GAEA['Mhalo'] >= 14) & (GAEA['vdisp_gap'] < 1400) & (GAEA['sep_max'] < 12) & (GAEA['z_obs_median'] < 0.1) )].reset_index(drop=True)
GAEA = GAEA.apply(lambda col: col.map(lambda x: x.real if np.iscomplexobj(x) else x))


print("Calculating sSFR...")
SHARK['sSFR'] = add_ssfr(SHARK['log_sfr_total'], SHARK['log_mstar_total'])
SAGE['sSFR'] = add_ssfr(SAGE['Total_Star_Formation_Rate'], SAGE['Total_Stellar_Mass'])
GAEA['sSFR'] = add_ssfr(GAEA['SFR'], GAEA['STELLAR_MASS'])


#########################################################################################
#### Quenched Stats for SHARK, SAGE, and GAEA

print("Calculating Quenched Stats...")

SHARK_quenched_SM, SHARK_quenched_SFR, SHARK_not_quenched_SM, SHARK_not_quenched_SFR, SHARK_boot, SHARK_Binary_QF = Quenched_Stats(SHARK['sSFR'], SHARK['log_mstar_total'], SHARK['log_sfr_total'])
SAGE_quenched_SM, SAGE_quenched_SFR, SAGE_not_quenched_SM, SAGE_not_quenched_SFR, SAGE_boot, SAGE_Binary_QF = Quenched_Stats(SAGE['sSFR'], SAGE['Total_Stellar_Mass'], SAGE['Total_Star_Formation_Rate'])
GAEA_quenched_SM, GAEA_quenched_SFR, GAEA_not_quenched_SM, GAEA_not_quenched_SFR, GAEA_boot, GAEA_Binary_QF = Quenched_Stats(GAEA['sSFR'], GAEA['STELLAR_MASS'], GAEA['SFR'])

#########################################################################################
##################### Quenched Fraction: Stellar Mass vs Halo Mass ######################
#########################################################################################

print("Calculating Quenched Fraction...")

threshold = 25  # Adjust as needed
# Calculate histogram quenched fraction for field galaxies in each zone
bins = np.arange(8, 12.5 + 0.1, 0.1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# SHARK
SHARK_counts_total, _ = np.histogram(SHARK['log_mstar_total'], bins=bins)
SHARK_counts_quenched, _ = np.histogram(SHARK['log_mstar_total'][SHARK_Binary_QF.astype(bool)], bins=bins)
SHARK_quenched_frac = np.zeros_like(bin_centers)
SHARK_mask = SHARK_counts_total > 0
SHARK_quenched_frac[SHARK_mask] = SHARK_counts_quenched[SHARK_mask] / SHARK_counts_total[SHARK_mask]
SHARK_colors = plt.get_cmap('coolwarm')(SHARK_quenched_frac)

# SAGE
SAGE_counts_total, _ = np.histogram(SAGE['Total_Stellar_Mass'], bins=bins)
SAGE_counts_quenched, _ = np.histogram(SAGE['Total_Stellar_Mass'][SAGE_Binary_QF.astype(bool)], bins=bins)
SAGE_quenched_frac = np.zeros_like(bin_centers)
SAGE_mask = SAGE_counts_total > 0
SAGE_quenched_frac[SAGE_mask] = SAGE_counts_quenched[SAGE_mask] / SAGE_counts_total[SAGE_mask]
SAGE_colors = plt.get_cmap('coolwarm')(SAGE_quenched_frac)  

# GAEA
GAEA_counts_total, _ = np.histogram(GAEA['STELLAR_MASS'], bins=bins)
GAEA_counts_quenched, _ = np.histogram(GAEA['STELLAR_MASS'][GAEA_Binary_QF.astype(bool)], bins=bins)
GAEA_quenched_frac = np.zeros_like(bin_centers)
GAEA_mask = GAEA_counts_total > 0
GAEA_quenched_frac[GAEA_mask] = GAEA_counts_quenched[GAEA_mask] / GAEA_counts_total[GAEA_mask]
GAEA_colors = plt.get_cmap('coolwarm')(GAEA_quenched_frac)  

SHARK_x_min, SHARK_x_max = min(SHARK['log_mstar_total'])-0.1, max(SHARK['log_mstar_total'])+0.1
SAGE_x_min, SAGE_x_max = min(SAGE['Total_Stellar_Mass'])-0.1, max(SAGE['Total_Stellar_Mass'])+0.1
GAEA_x_min, GAEA_x_max = min(GAEA['STELLAR_MASS'])-0.1, max(GAEA['STELLAR_MASS'])+0.1

SHARK_y_min, SHARK_y_max = min(SHARK['mvir_hosthalo'])-0.1, max(SHARK['mvir_hosthalo'])+0.1
SAGE_y_min, SAGE_y_max = min(SAGE['Central_Galaxy_Mvir'])-0.1, max(SAGE['Central_Galaxy_Mvir'])+0.1
GAEA_y_min, GAEA_y_max = min(GAEA['M_HALO'])-0.1, max(GAEA['M_HALO'])+0.1

SHARK_xgrid = np.arange(SHARK_x_min, SHARK_x_max, 0.02)
SAGE_xgrid = np.arange(SAGE_x_min, SAGE_x_max, 0.02)
GAEA_xgrid = np.arange(GAEA_x_min, GAEA_x_max, 0.02)

SHARK_ygrid = np.arange(SHARK_y_min, SHARK_y_max, 0.02)
SAGE_ygrid = np.arange(SAGE_y_min, SAGE_y_max, 0.02)
GAEA_ygrid = np.arange(GAEA_y_min, GAEA_y_max, 0.02)

SHARK_dens2d = kde_2d(SHARK['log_mstar_total'], SHARK['mvir_hosthalo'], SHARK_xgrid, SHARK_ygrid, 0.15, 0.3)
SAGE_dens2d = kde_2d(SAGE['Total_Stellar_Mass'], SAGE['Central_Galaxy_Mvir'], SAGE_xgrid, SAGE_ygrid, 0.15, 0.3)
GAEA_dens2d = kde_2d(GAEA['STELLAR_MASS'], GAEA['M_HALO'], GAEA_xgrid, GAEA_ygrid, 0.15, 0.3)

SHARK_dens2d_1 = kde_2d(SHARK['log_mstar_total'], SHARK['mvir_hosthalo'], SHARK_xgrid, SHARK_ygrid, 0.15, 0.3, SHARK_Binary_QF)
SAGE_dens2d_1 = kde_2d(SAGE['Total_Stellar_Mass'], SAGE['Central_Galaxy_Mvir'], SAGE_xgrid, SAGE_ygrid, 0.15, 0.3, SAGE_Binary_QF)
GAEA_dens2d_1 = kde_2d(GAEA['STELLAR_MASS'], GAEA['M_HALO'], GAEA_xgrid, GAEA_ygrid, 0.15, 0.3, GAEA_Binary_QF)

SHARK_weighted_dens2d = SHARK_dens2d_1 / SHARK_dens2d
SAGE_weighted_dens2d = SAGE_dens2d_1 / SAGE_dens2d
GAEA_weighted_dens2d = GAEA_dens2d_1 / GAEA_dens2d

SHARK_masked_weighted_dens2d = np.ma.masked_where(SHARK_dens2d < threshold, SHARK_weighted_dens2d)
SAGE_masked_weighted_dens2d = np.ma.masked_where(SAGE_dens2d < threshold, SAGE_weighted_dens2d)
GAEA_masked_weighted_dens2d = np.ma.masked_where(GAEA_dens2d < threshold, GAEA_weighted_dens2d)

print("Plotting...")

fig = plt.figure(figsize=(14, 5))
gs = fig.add_gridspec(1, 3, wspace=0)

ax_SHARK = fig.add_subplot(gs[0, 0])
ax_SAGE = fig.add_subplot(gs[0, 1])
ax_GAEA = fig.add_subplot(gs[0, 2])

# Density plot for SHARK
ax_SHARK.minorticks_on()
ax_SHARK.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                 bottom=True, top=True, left=True, right=True, direction='in')
ax_SHARK.set_xlabel(r'$\log$ M$_{\star}$ [M$_{\odot}$]')
ax_SHARK.set_ylabel(r'$\log$ M$_{\mathrm{halo}}$ [M$_\odot$]')
ax_SHARK.set_xlim(min([SHARK_x_min, SAGE_x_min, GAEA_x_min]), max([SHARK_x_max, SAGE_x_max, GAEA_x_max]))
ax_SHARK.set_ylim(min([SHARK_y_min, SAGE_y_min, GAEA_y_min]), max([SHARK_y_max, SAGE_y_max, GAEA_y_max]))
ax_SHARK.set_title('SHARK', fontsize=16)
# ax_SHARK.scatter(SHARK['log_mstar_total'], SHARK['mvir_hosthalo'], c='k', s=1, alpha=0.02)
ax_SHARK.contour(SHARK_xgrid, SHARK_ygrid, SHARK_dens2d.T, colors='k', linewidths=0.75, levels=12, zorder=-3, linestyles='solid', alpha=0.7)
cf_1 = ax_SHARK.contourf(SHARK_xgrid, SHARK_ygrid, SHARK_masked_weighted_dens2d.T, cmap='coolwarm', zorder=-4, vmin=0, vmax=1, levels=256)

# Density plot for SAGE
ax_SAGE.minorticks_on()
ax_SAGE.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                 bottom=True, top=True, left=True, right=True, direction='in')
ax_SAGE.set_xlabel(r'$\log$ M$_{\star}$ [M$_{\odot}$]')
ax_SAGE.set_xlim(min([SHARK_x_min, SAGE_x_min, GAEA_x_min]), max([SHARK_x_max, SAGE_x_max, GAEA_x_max]))
ax_SAGE.set_ylim(min([SHARK_y_min, SAGE_y_min, GAEA_y_min]), max([SHARK_y_max, SAGE_y_max, GAEA_y_max]))
ax_SAGE.set_title('SAGE', fontsize=16)
# Plot 1 in 100 points for SAGE
scatter_indices = np.arange(0, len(SAGE['Total_Stellar_Mass']), 25)
# ax_SAGE.scatter(SAGE['Total_Stellar_Mass'].iloc[scatter_indices], SAGE['Central_Galaxy_Mvir'].iloc[scatter_indices], c='k', s=1, alpha=0.02)
ax_SAGE.contour(SAGE_xgrid, SAGE_ygrid, SAGE_dens2d.T, colors='k', linewidths=0.75, levels=12, zorder=-3, linestyles='solid', alpha=0.7)
cf_2 = ax_SAGE.contourf(SAGE_xgrid, SAGE_ygrid, SAGE_masked_weighted_dens2d.T, cmap='coolwarm', zorder=-4, vmin=0, vmax=1, levels=256)

# # Density plot for GAEA
ax_GAEA.minorticks_on()
ax_GAEA.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                 bottom=True, top=True, left=True, right=True, direction='in')
ax_GAEA.set_xlabel(r'$\log$ M$_{\star}$ [M$_{\odot}$]')
ax_GAEA.set_xlim(min([SHARK_x_min, SAGE_x_min, GAEA_x_min]), max([SHARK_x_max, SAGE_x_max, GAEA_x_max]))
ax_GAEA.set_ylim(min([SHARK_y_min, SAGE_y_min, GAEA_y_min]), max([SHARK_y_max, SAGE_y_max, GAEA_y_max]))
ax_GAEA.set_title('GAEA', fontsize=16)
scatter_indices = np.arange(0, len(GAEA['STELLAR_MASS']), 100)
# ax_GAEA.scatter(GAEA['STELLAR_MASS'].iloc[scatter_indices], GAEA['M_HALO'].iloc[scatter_indices], c='k', s=1, alpha=0.02)
ax_GAEA.contour(GAEA_xgrid, GAEA_ygrid, GAEA_dens2d.T, colors='k', linewidths=0.75, levels=12, zorder=-3, linestyles='solid', alpha=0.7)
cf_3 = ax_GAEA.contourf(GAEA_xgrid, GAEA_ygrid, GAEA_masked_weighted_dens2d.T, cmap='coolwarm', zorder=-4, vmin=0, vmax=1, levels=256)

# Colorbar for density plots
cbar = fig.colorbar(cf_1, ax=[ax_SHARK, ax_SAGE, ax_GAEA], pad=0.12, location='bottom', aspect=50)
cbar.set_label('Quenched Fraction', rotation=0, labelpad=0)
cbar.set_ticks(np.arange(0, 1.2, 0.2))

plt.savefig('/fred/oz004/wvankemp/Halo_Mass_Relations/Plots/QF_Mstar_Mhalo_Sims.png', dpi=300)
plt.show()
plt.close(fig)