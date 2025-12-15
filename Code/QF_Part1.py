import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from astropy.table import Table
from astropy.io import fits
from scipy.special import erf
from astropy.constants import c
from astropy.constants import G
from astropy import units as u 
from tqdm import tqdm
import pickle
import multiprocessing

G = G.to(u.M_sun**-1 * u.km**2 * u.s**-2 * u.Mpc).value
band = 'i_band'
zlim = '01'


def add_ssfr(sfr, sm):
    ssfr = sfr - sm
    return ssfr

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

# Note: You'll need to define 'band' and 'zlim' variables before running this
# band = 'your_band_value'
# zlim = 'your_zlim_value'

SHARK = Table.read(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/{band}/group_galaxies_{zlim}.fits')
SHARK = SHARK.to_pandas()
SHARK = SHARK[(SHARK['vdisp_gap'] >= (SHARK['rad'] * 150 - 200)) & (SHARK['vdisp_gap'] <= (SHARK['rad'] * 150 + 400)) & (SHARK['rad'] < 12) & (SHARK['vdisp_gap'] < 1800)].reset_index(drop=True)


# Load SAGE data (concatenate 10 files)
sage_data_list = []
for i in range(10):
    sage_file = f"/fred/oz004/wvankemp/Halo_Mass_New/Data/SAGE/data_group_{i}.fits"
    sage_data_list.append(pd.DataFrame(fits.getdata(sage_file)))
SAGE = pd.concat(sage_data_list, ignore_index=True)
SAGE = SAGE[(SAGE['vdisp_gap'] >= (SAGE['radius'] * 150 - 200)) & (SAGE['vdisp_gap'] <= (SAGE['radius'] * 150 + 400)) & (SAGE['radius'] < 12) & (SAGE['vdisp_gap'] < 1800)].reset_index(drop=True)
SAGE = SAGE.apply(lambda col: col.map(lambda x: x.real if np.iscomplexobj(x) else x))

# Load GAEA data
GAEA = Table.read("/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Group_Galaxies_1e8.fits")
GAEA = GAEA.to_pandas()
GAEA = GAEA[(GAEA['vdisp_gap'] >= (GAEA['sep_max'] * 150 - 200)) & (GAEA['vdisp_gap'] <= (GAEA['sep_max'] * 150 + 400)) & (GAEA['sep_max'] < 12) & (GAEA['vdisp_gap'] < 1800)].reset_index(drop=True)
GAEA = GAEA[(GAEA['Mhalo'] <= 13.5) | ((GAEA['Mhalo'] > 13.5) & (GAEA['Mhalo'] - GAEA['Mhalo_VT'] < 1.2))].reset_index(drop=True)
GAEA = GAEA.apply(lambda col: col.map(lambda x: x.real if np.iscomplexobj(x) else x))

print("Calculating sSFR...")
SHARK['sSFR'] = add_ssfr(SHARK['log_sfr_total'], SHARK['log_mstar_total'])
SAGE['sSFR'] = add_ssfr(SAGE['Total_Star_Formation_Rate'], SAGE['Total_Stellar_Mass'])
GAEA['sSFR'] = add_ssfr(GAEA['SFR'], GAEA['SM'])

#########################################################################################
#### Quenched Stats for SHARK, SAGE, and GAEA

print("Calculating Quenched Stats...")

SHARK_quenched_SM, SHARK_quenched_SFR, SHARK_not_quenched_SM, SHARK_not_quenched_SFR, SHARK_boot, SHARK_Binary_QF = Quenched_Stats(SHARK['sSFR'], SHARK['log_mstar_total'], SHARK['log_sfr_total'])
SAGE_quenched_SM, SAGE_quenched_SFR, SAGE_not_quenched_SM, SAGE_not_quenched_SFR, SAGE_boot, SAGE_Binary_QF = Quenched_Stats(SAGE['sSFR'], SAGE['Total_Stellar_Mass'], SAGE['Total_Star_Formation_Rate'])
GAEA_quenched_SM, GAEA_quenched_SFR, GAEA_not_quenched_SM, GAEA_not_quenched_SFR, GAEA_boot, GAEA_Binary_QF = Quenched_Stats(GAEA['sSFR'], GAEA['SM'], GAEA['SFR'])

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
GAEA_counts_total, _ = np.histogram(GAEA['SM'], bins=bins)
GAEA_counts_quenched, _ = np.histogram(GAEA['SM'][GAEA_Binary_QF.astype(bool)], bins=bins)
GAEA_quenched_frac = np.zeros_like(bin_centers)
GAEA_mask = GAEA_counts_total > 0
GAEA_quenched_frac[GAEA_mask] = GAEA_counts_quenched[GAEA_mask] / GAEA_counts_total[GAEA_mask]
GAEA_colors = plt.get_cmap('coolwarm')(GAEA_quenched_frac)  

print("Setting up grids and calculating KDE...")

SHARK_x_min, SHARK_x_max = min(SHARK['log_mstar_total'])-0.1, max(SHARK['log_mstar_total'])+0.1
SAGE_x_min, SAGE_x_max = min(SAGE['Total_Stellar_Mass'])-0.1, max(SAGE['Total_Stellar_Mass'])+0.1
GAEA_x_min, GAEA_x_max = min(GAEA['SM'])-0.1, max(GAEA['SM'])+0.1

SHARK_y_min, SHARK_y_max = min(SHARK['mvir_hosthalo'])-0.1, max(SHARK['mvir_hosthalo'])+0.1
SAGE_y_min, SAGE_y_max = min(SAGE['Central_Galaxy_Mvir'])-0.1, max(SAGE['Central_Galaxy_Mvir'])+0.1
GAEA_y_min, GAEA_y_max = min(GAEA['Mhalo'])-0.1, max(GAEA['Mhalo'])+0.1

SHARK_xgrid = np.arange(SHARK_x_min, SHARK_x_max, 0.02)
SAGE_xgrid = np.arange(SAGE_x_min, SAGE_x_max, 0.02)
GAEA_xgrid = np.arange(GAEA_x_min, GAEA_x_max, 0.02)

SHARK_ygrid = np.arange(SHARK_y_min, SHARK_y_max, 0.02)
SAGE_ygrid = np.arange(SAGE_y_min, SAGE_y_max, 0.02)
GAEA_ygrid = np.arange(GAEA_y_min, GAEA_y_max, 0.02)

print("Computing SHARK KDE...")
SHARK_dens2d = kde_2d(SHARK['log_mstar_total'], SHARK['mvir_hosthalo'], SHARK_xgrid, SHARK_ygrid, 0.15, 0.3)
SHARK_dens2d_1 = kde_2d(SHARK['log_mstar_total'], SHARK['mvir_hosthalo'], SHARK_xgrid, SHARK_ygrid, 0.15, 0.3, SHARK_Binary_QF)

print("Computing SAGE KDE...")
SAGE_dens2d = kde_2d(SAGE['Total_Stellar_Mass'], SAGE['Central_Galaxy_Mvir'], SAGE_xgrid, SAGE_ygrid, 0.15, 0.3)
SAGE_dens2d_1 = kde_2d(SAGE['Total_Stellar_Mass'], SAGE['Central_Galaxy_Mvir'], SAGE_xgrid, SAGE_ygrid, 0.15, 0.3, SAGE_Binary_QF)

print("Computing GAEA KDE...")
GAEA_dens2d = kde_2d(GAEA['SM'], GAEA['Mhalo'], GAEA_xgrid, GAEA_ygrid, 0.15, 0.3)
GAEA_dens2d_1 = kde_2d(GAEA['SM'], GAEA['Mhalo'], GAEA_xgrid, GAEA_ygrid, 0.15, 0.3, GAEA_Binary_QF)

print("Computing weighted densities...")
SHARK_weighted_dens2d = SHARK_dens2d_1 / SHARK_dens2d
SAGE_weighted_dens2d = SAGE_dens2d_1 / SAGE_dens2d
GAEA_weighted_dens2d = GAEA_dens2d_1 / GAEA_dens2d

SHARK_masked_weighted_dens2d = np.ma.masked_where(SHARK_dens2d < threshold, SHARK_weighted_dens2d)
SAGE_masked_weighted_dens2d = np.ma.masked_where(SAGE_dens2d < threshold, SAGE_weighted_dens2d)
GAEA_masked_weighted_dens2d = np.ma.masked_where(GAEA_dens2d < threshold, GAEA_weighted_dens2d)

print("Saving processed data...")
# Save all the computed arrays and metadata needed for plotting
data_to_save = {
    # SHARK data
    'SHARK_dens2d': SHARK_dens2d,
    'SHARK_dens2d_1': SHARK_dens2d_1,  
    'SHARK_weighted_dens2d': SHARK_weighted_dens2d,
    'SHARK_masked_weighted_dens2d': SHARK_masked_weighted_dens2d,
    'SHARK_xgrid': SHARK_xgrid,
    'SHARK_ygrid': SHARK_ygrid,
    
    # SAGE data
    'SAGE_dens2d': SAGE_dens2d,
    'SAGE_dens2d_1': SAGE_dens2d_1,
    'SAGE_weighted_dens2d': SAGE_weighted_dens2d,
    'SAGE_masked_weighted_dens2d': SAGE_masked_weighted_dens2d,
    'SAGE_xgrid': SAGE_xgrid,
    'SAGE_ygrid': SAGE_ygrid,
    
    # GAEA data
    'GAEA_dens2d': GAEA_dens2d,
    'GAEA_dens2d_1': GAEA_dens2d_1,
    'GAEA_weighted_dens2d': GAEA_weighted_dens2d,
    'GAEA_masked_weighted_dens2d': GAEA_masked_weighted_dens2d,
    'GAEA_xgrid': GAEA_xgrid,
    'GAEA_ygrid': GAEA_ygrid,
    
    # Plot limits (for consistent plotting)
    'x_min': min([SHARK_x_min, SAGE_x_min, GAEA_x_min]),
    'x_max': max([SHARK_x_max, SAGE_x_max, GAEA_x_max]),
    'y_min': min([SHARK_y_min, SAGE_y_min, GAEA_y_min]),
    'y_max': max([SHARK_y_max, SAGE_y_max, GAEA_y_max]),
    
    # Threshold used
    'threshold': threshold
}

# Save using pickle
with open('/fred/oz004/wvankemp/Halo_Mass_New/Data/Densities/processed_density_data.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("Data processing complete! Saved to 'processed_density_data.pkl'")