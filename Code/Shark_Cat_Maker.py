import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import seaborn as sns
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology.funcs import z_at_value
from astropy.coordinates import SkyCoord
from astropy import units as u 
from astropy.constants import G
import multiprocessing as mp
from astropy.constants import c
from tqdm import tqdm  # Optional: For progress bar

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
G = G.to(u.M_sun**-1 * u.km**2 * u.s**-2 * u.Mpc).value


def fix_h_data(df, h_new=0.7, h_old=0.6751):
    df['mstars_disk'] = df['mstars_disk'] / h_new
    df['mstars_bulge'] = df['mstars_bulge'] / h_new
    # Calculate log_mstar_total after updating disk and bulge
    df['log_mstar_total'] = np.log10(df['mstars_disk'] + df['mstars_bulge'])
    df['mvir_hosthalo'] = np.log10(df['mvir_hosthalo'] / h_new)
    df['mvir_subhalo'] = np.log10(df['mvir_subhalo'] / h_new)
    df['sfr_disk'] = df['sfr_disk'] / h_new
    df['sfr_burst'] = df['sfr_burst'] / h_new
    df['log_sfr_total'] = np.log10((df['sfr_disk'] + df['sfr_burst'])/1e9)
    df['rstar_disk_intrinsic'] = df['rstar_disk_intrinsic'] / h_new
    df['total_ap_dust_FUV_GALEX'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_FUV_GALEX']
    df['total_ap_dust_NUV_GALEX'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_NUV_GALEX']
    df['total_ap_dust_u_SDSS'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_u_SDSS']
    df['total_ap_dust_g_SDSS'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_g_SDSS']
    df['total_ap_dust_r_SDSS'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_r_SDSS']
    df['total_ap_dust_i_SDSS'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_i_SDSS']
    df['total_ap_dust_z_SDSS'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_z_SDSS']
    df['total_ap_dust_u_VST'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_u_VST']
    df['total_ap_dust_g_VST'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_g_VST']
    df['total_ap_dust_r_VST'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_r_VST']
    df['total_ap_dust_i_VST'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_i_VST']
    df['total_ap_dust_Z_VISTA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_Z_VISTA']
    df['total_ap_dust_Y_VISTA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_Y_VISTA']
    df['total_ap_dust_J_VISTA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_J_VISTA']
    df['total_ap_dust_H_VISTA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_H_VISTA']
    df['total_ap_dust_K_VISTA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_K_VISTA']
    df['total_ap_dust_W1_WISE'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_W1_WISE']
    df['total_ap_dust_I1_Spitzer'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_I1_Spitzer']
    df['total_ap_dust_I2_Spitzer'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_I2_Spitzer']
    df['total_ap_dust_W2_WISE'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_W2_WISE']
    df['total_ap_dust_I3_Spitzer'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_I3_Spitzer']
    df['total_ap_dust_I4_Spitzer'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_I4_Spitzer']
    df['total_ap_dust_W3_WISE'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_W3_WISE']
    df['total_ap_dust_W4_WISE'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_W4_WISE']
    df['total_ap_dust_M24_Spitzer'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_M24_Spitzer']
    df['total_ap_dust_M70_Spitzer'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_M70_Spitzer']
    df['total_ap_dust_P70_Herschel'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_P70_Herschel']
    df['total_ap_dust_P100_Herschel'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_P100_Herschel']
    df['total_ap_dust_P160_Herschel'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_P160_Herschel']
    df['total_ap_dust_S250_Herschel'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_S250_Herschel']
    df['total_ap_dust_S350_Herschel'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_S350_Herschel']
    df['total_ap_dust_S450_JCMT'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_S450_JCMT']
    df['total_ap_dust_S500_Herschel'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_S500_Herschel']
    df['total_ap_dust_S850_JCMT'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_S850_JCMT']
    df['total_ap_dust_Band_ionising_photons'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_Band_ionising_photons']
    df['total_ap_dust_Band9_ALMA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_Band9_ALMA']
    df['total_ap_dust_Band8_ALMA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_Band8_ALMA']
    df['total_ap_dust_Band7_ALMA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_Band7_ALMA']
    df['total_ap_dust_Band6_ALMA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_Band6_ALMA']
    df['total_ap_dust_Band5_ALMA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_Band5_ALMA']
    df['total_ap_dust_Band4_ALMA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_Band4_ALMA']
    df['total_ap_dust_Band3_ALMA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_Band3_ALMA']
    df['total_ap_dust_BandX_VLA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_BandX_VLA']
    df['total_ap_dust_BandC_VLA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_BandC_VLA']
    df['total_ap_dust_BandS_VLA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_BandS_VLA']
    df['total_ap_dust_BandL_VLA'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_BandL_VLA']
    df['total_ap_dust_Band_610MHz'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_Band_610MHz']
    df['total_ap_dust_Band_325MHz'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_Band_325MHz']
    df['total_ap_dust_Band_150MHz'] = 5*np.log10(h_new/h_old) + df['total_ap_dust_Band_150MHz']
    return df

def disp_gap(data):
    N = len(data)
    if N < 2:
        return np.nan
    
    data_sort = np.sort(data)
    # compute gaps and weights
    gaps = data_sort[1:] - data_sort[:-1]   
    weights = np.arange(1, N) * np.arange(N - 1, 0, -1)
    
    # Gapper estimate of dispersion
    sigma_gap = np.sqrt(np.pi) / (N * (N - 1)) * np.sum(weights * gaps)
    sigma = np.sqrt((N / (N - 1)) * sigma_gap**2)

    return sigma

def mstar_cen3_func(group):
    top3 = group.nlargest(3, 'log_mstar_total')
    mstar_sum = np.sum(10**top3['log_mstar_total'])
    return np.log10(mstar_sum)

def make_m_halo_cen3(df):
    # A = 31.111
    # M_A = 10.769
    # beta = 0.889
    # gamma = -0.555
    A = 46.944
    M_A = 10.483
    beta = 0.249
    gamma = -0.601
    Mc = 10**df['mstar_cen3']  # Convert log values back to linear scale
    Mh = A * Mc * ((Mc / (10**M_A))**(beta) + (Mc / (10**M_A))**gamma)
    return np.log10(Mh)

def make_m_halo_cvt(df):
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
    Ac = np.where(df['rad'] < rad_lim,
                 beta * ((df['rad'] / rad_lim)**n2 - 1),
                 0)
    A = 5/3 + Ab + Ac
    return np.log10(A * df['vdisp_gap']**2 * df['rad'] / G)

def cat_maker(df, dg, zcut):
    df['vel'] = (df['zobs']  * c.to(u.km/u.s).value) / (1 + df['zobs'])
    # Identify group membership counts
    group_counts = df['id_group_sky'].value_counts()
    dg = dg[dg['zobs'] < zcut]
    # Group galaxies: id_group_sky != -1 AND groups with >=3 members
    valid_groups = set(dg[dg['flag'] == 0]['id_group_sky'])
    group_mask = (
        (df['id_group_sky'] != -1) &
        (df['id_group_sky'].map(group_counts) >= 3) &
        (df['id_group_sky'].isin(valid_groups))
    )
    df_group_gals = df[group_mask].copy().reset_index(drop=True)
    df_group_gals['Nm'] = df_group_gals.groupby('id_group_sky')['id_group_sky'].transform('count')

    # Field galaxies: galaxies whose id_group_sky is not in any valid group
    field_mask = ~df['id_group_sky'].isin(valid_groups)
    df_field_gals = df[field_mask].copy().reset_index(drop=True)

    # Make vdisp_gap from group galaxies
    df_group_gals['vdisp_gap'] = df_group_gals.groupby('id_group_sky')['vel'].transform(disp_gap)

    # Make projected separation from group centre to each galaxy
    # Ensure .values are used to avoid pandas dtype issues
    ra_gal = df_group_gals['ra'].values * u.deg
    dec_gal = df_group_gals['dec'].values * u.deg
    ra_cen = df_group_gals.groupby('id_group_sky')['ra'].transform('median').values * u.deg
    dec_cen = df_group_gals.groupby('id_group_sky')['dec'].transform('median').values * u.deg
    z_cen = df_group_gals.groupby('id_group_sky')['zobs'].transform('median').values
    c1 = SkyCoord(ra=ra_gal, dec=dec_gal, frame='icrs')
    c2 = SkyCoord(ra=ra_cen, dec=dec_cen, frame='icrs')
    sep_angle = c1.separation(c2)  # This is an Angle with units
    comov_dist = cosmo.comoving_distance(z_cen)  # This is an astropy Quantity with units of Mpc
    # If sep_angle and comov_dist are arrays, ensure broadcasting works
    df_group_gals['sep'] = (2 * comov_dist.value * np.sin(sep_angle.to(u.radian).value / 2))


    group_cat = df_group_gals.groupby('id_group_sky').agg({
        'mvir_hosthalo': 'first',  # same for all in group
        'ra': 'median',
        'dec': 'median',
        'zobs': 'median',
        'zcos': 'median',
        'Nm': 'first',  
        'vdisp_gap': 'first',
        'sep': 'max', 
        'log_mstar_total': 'max',  
        'log_sfr_total': 'max'
    }).reset_index()
    group_cat = group_cat.rename(columns={'sep': 'rad'})
    group_cat = group_cat.rename(columns={'log_mstar_total': 'mstar_cen'})
    group_cat = group_cat.rename(columns={'log_sfr_total': 'sfr_cen'})

    # Avoid DeprecationWarning: select grouping columns before apply
    group_cat['mstar_cen3'] = df_group_gals.groupby('id_group_sky')[['log_mstar_total']].apply(mstar_cen3_func).values
    group_cat['Mhalo_P3M'] = make_m_halo_cen3(group_cat)
    group_cat['Mhalo_CVT'] = make_m_halo_cvt(group_cat)
    group_cat['Mhalo_CT'] = np.log10(5/3 * group_cat['vdisp_gap']**2 * group_cat['rad'] / G)
    # Merge group halo masses and rad back to group galaxies via id_group_sky
    cols_to_merge = ['id_group_sky', 'Mhalo_P3M', 'Mhalo_CVT', 'Mhalo_CT', 'rad']
    df_group_gals = df_group_gals.merge(group_cat[cols_to_merge], on='id_group_sky', how='left')
    return group_cat, df_group_gals, df_field_gals



df = Table.read('/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/waves_wide_gals.fits')
df = df.to_pandas()

dg = Table.read('/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/waves_wide_groups.fits')
dg = dg.to_pandas()

df = fix_h_data(df)

df_i_01 = df[(df['total_ap_dust_i_SDSS'] <= 19.2) & (df['total_ap_dust_i_SDSS'] > 0) & (df['zobs'] < 0.115) & (df['log_mstar_total'] > 7.5)].reset_index(drop=True)
df_i_02 = df[(df['total_ap_dust_i_SDSS'] <= 19.2) & (df['total_ap_dust_i_SDSS'] > 0) & (df['zobs'] < 0.215) & (df['log_mstar_total'] > 7.5)].reset_index(drop=True)
df_i_03 = df[(df['total_ap_dust_i_SDSS'] <= 19.2) & (df['total_ap_dust_i_SDSS'] > 0) & (df['zobs'] < 0.315) & (df['log_mstar_total'] > 7.5)].reset_index(drop=True)
df_Z_01 = df[(df['total_ap_dust_Z_VISTA'] <=21.2) & (df['total_ap_dust_Z_VISTA'] > 0) & (df['zobs'] < 0.115) & (df['log_mstar_total'] > 7.5)].reset_index(drop=True)
df_Z_02 = df[(df['total_ap_dust_Z_VISTA'] <= 21.2) & (df['total_ap_dust_Z_VISTA'] > 0) & (df['zobs'] < 0.215) & (df['log_mstar_total'] > 7.5)].reset_index(drop=True)
df_Z_03 = df[(df['total_ap_dust_Z_VISTA'] <= 21.2) & (df['total_ap_dust_Z_VISTA'] > 0) & (df['zobs'] < 0.315) & (df['log_mstar_total'] > 7.5)].reset_index(drop=True)
df_r_01 = df[(df['total_ap_dust_r_SDSS'] <= 19.8) & (df['total_ap_dust_r_SDSS'] > 0) & (df['zobs'] < 0.115) & (df['log_mstar_total'] > 7.5)].reset_index(drop=True)
df_r_02 = df[(df['total_ap_dust_r_SDSS'] <= 19.8) & (df['total_ap_dust_r_SDSS'] > 0) & (df['zobs'] < 0.215) & (df['log_mstar_total'] > 7.5)].reset_index(drop=True)
df_r_03 = df[(df['total_ap_dust_r_SDSS'] <= 19.8) & (df['total_ap_dust_r_SDSS'] > 0) & (df['zobs'] < 0.315) & (df['log_mstar_total'] > 7.5)].reset_index(drop=True)


groups_i_01, df_i_group_gals_01, df_i_field_gals_01 = cat_maker(df_i_01, dg[dg['id_group_sky'].isin(df_i_01['id_group_sky'])].reset_index(drop=True), zcut=0.1)
groups_i_02, df_i_group_gals_02, df_i_field_gals_02 = cat_maker(df_i_02, dg[dg['id_group_sky'].isin(df_i_02['id_group_sky'])].reset_index(drop=True), zcut=0.2)
groups_i_03, df_i_group_gals_03, df_i_field_gals_03 = cat_maker(df_i_03, dg[dg['id_group_sky'].isin(df_i_03['id_group_sky'])].reset_index(drop=True), zcut=0.3)
groups_Z_01, df_Z_group_gals_01, df_Z_field_gals_01 = cat_maker(df_Z_01, dg[dg['id_group_sky'].isin(df_Z_01['id_group_sky'])].reset_index(drop=True), zcut=0.1)
groups_Z_02, df_Z_group_gals_02, df_Z_field_gals_02 = cat_maker(df_Z_02, dg[dg['id_group_sky'].isin(df_Z_02['id_group_sky'])].reset_index(drop=True), zcut=0.2)
groups_Z_03, df_Z_group_gals_03, df_Z_field_gals_03 = cat_maker(df_Z_03, dg[dg['id_group_sky'].isin(df_Z_03['id_group_sky'])].reset_index(drop=True), zcut=0.3)
groups_r_01, df_r_group_gals_01, df_r_field_gals_01 = cat_maker(df_r_01, dg[dg['id_group_sky'].isin(df_r_01['id_group_sky'])].reset_index(drop=True), zcut=0.1)
groups_r_02, df_r_group_gals_02, df_r_field_gals_02 = cat_maker(df_r_02, dg[dg['id_group_sky'].isin(df_r_02['id_group_sky'])].reset_index(drop=True), zcut=0.2)
groups_r_03, df_r_group_gals_03, df_r_field_gals_03 = cat_maker(df_r_03, dg[dg['id_group_sky'].isin(df_r_03['id_group_sky'])].reset_index(drop=True), zcut=0.3)

# Save the catalogs
groups_i_01 = Table.from_pandas(groups_i_01)
groups_i_01.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/i_band/group_catalog_01.fits', format='fits', overwrite=True)

groups_i_02 = Table.from_pandas(groups_i_02)
groups_i_02.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/i_band/group_catalog_02.fits', format='fits', overwrite=True)

groups_i_03 = Table.from_pandas(groups_i_03)
groups_i_03.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/i_band/group_catalog_03.fits', format='fits', overwrite=True)

groups_Z_01 = Table.from_pandas(groups_Z_01)
groups_Z_01.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/Z_band/group_catalog_01.fits', format='fits', overwrite=True)

groups_Z_02 = Table.from_pandas(groups_Z_02)
groups_Z_02.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/Z_band/group_catalog_02.fits', format='fits', overwrite=True)

groups_Z_03 = Table.from_pandas(groups_Z_03)
groups_Z_03.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/Z_band/group_catalog_03.fits', format='fits', overwrite=True)

groups_r_01 = Table.from_pandas(groups_r_01)
groups_r_01.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/r_band/group_catalog_01.fits', format='fits', overwrite=True)

groups_r_02 = Table.from_pandas(groups_r_02)
groups_r_02.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/r_band/group_catalog_02.fits', format='fits', overwrite=True)

groups_r_03 = Table.from_pandas(groups_r_03)
groups_r_03.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/r_band/group_catalog_03.fits', format='fits', overwrite=True)

# Save the group galaxies
df_i_group_gals_01 = Table.from_pandas(df_i_group_gals_01)
df_i_group_gals_01.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/i_band/group_galaxies_01.fits', format='fits', overwrite=True)

df_i_group_gals_02 = Table.from_pandas(df_i_group_gals_02)
df_i_group_gals_02.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/i_band/group_galaxies_02.fits', format='fits', overwrite=True)

df_i_group_gals_03 = Table.from_pandas(df_i_group_gals_03)
df_i_group_gals_03.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/i_band/group_galaxies_03.fits', format='fits', overwrite=True)

df_Z_group_gals_01 = Table.from_pandas(df_Z_group_gals_01)
df_Z_group_gals_01.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/Z_band/group_galaxies_01.fits', format='fits', overwrite=True)

df_Z_group_gals_02 = Table.from_pandas(df_Z_group_gals_02)      
df_Z_group_gals_02.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/Z_band/group_galaxies_02.fits', format='fits', overwrite=True)

df_Z_group_gals_03 = Table.from_pandas(df_Z_group_gals_03)
df_Z_group_gals_03.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/Z_band/group_galaxies_03.fits', format='fits', overwrite=True)

df_r_group_gals_01 = Table.from_pandas(df_r_group_gals_01)
df_r_group_gals_01.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/r_band/group_galaxies_01.fits', format='fits', overwrite=True)

df_r_group_gals_02 = Table.from_pandas(df_r_group_gals_02)
df_r_group_gals_02.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/r_band/group_galaxies_02.fits', format='fits', overwrite=True)

df_r_group_gals_03 = Table.from_pandas(df_r_group_gals_03)
df_r_group_gals_03.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/r_band/group_galaxies_03.fits', format='fits', overwrite=True)

# Save the field galaxies
df_i_field_gals_01 = Table.from_pandas(df_i_field_gals_01)
df_i_field_gals_01.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/i_band/field_galaxies_01.fits', format='fits', overwrite=True)

df_i_field_gals_02 = Table.from_pandas(df_i_field_gals_02)
df_i_field_gals_02.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/i_band/field_galaxies_02.fits', format='fits', overwrite=True)

df_i_field_gals_03 = Table.from_pandas(df_i_field_gals_03)
df_i_field_gals_03.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/i_band/field_galaxies_03.fits', format='fits', overwrite=True)

df_Z_field_gals_01 = Table.from_pandas(df_Z_field_gals_01)
df_Z_field_gals_01.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/Z_band/field_galaxies_01.fits', format='fits', overwrite=True)

df_Z_field_gals_02 = Table.from_pandas(df_Z_field_gals_02)
df_Z_field_gals_02.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/Z_band/field_galaxies_02.fits', format='fits', overwrite=True)

df_Z_field_gals_03 = Table.from_pandas(df_Z_field_gals_03)
df_Z_field_gals_03.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/Z_band/field_galaxies_03.fits', format='fits', overwrite=True)

df_r_field_gals_01 = Table.from_pandas(df_r_field_gals_01)
df_r_field_gals_01.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/r_band/field_galaxies_01.fits', format='fits', overwrite=True)

df_r_field_gals_02 = Table.from_pandas(df_r_field_gals_02)
df_r_field_gals_02.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/r_band/field_galaxies_02.fits', format='fits', overwrite=True)

df_r_field_gals_03 = Table.from_pandas(df_r_field_gals_03)
df_r_field_gals_03.write(f'/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/r_band/field_galaxies_03.fits', format='fits', overwrite=True)
