from tokenize import group
import numpy as np
from astropy.io import fits
from astropy.cosmology.funcs import z_at_value
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u, constants as const
from tqdm import tqdm
from scipy.spatial import cKDTree
from astropy.table import Table
from astropy.coordinates import SkyCoord
import pandas as pd
import time

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
G = const.G.to(u.M_sun**-1 * u.km**2 * u.s**-2 * u.Mpc).value


def disp_gap(velocity):
    N = len(velocity)
    if N < 2:
        return np.nan
    
    data_sort = np.sort(velocity)
    # compute gaps and weights
    gaps = data_sort[1:] - data_sort[:-1]   
    weights = np.arange(1, N) * np.arange(N - 1, 0, -1)
    
    # Gapper estimate of dispersion
    sigma_gap = np.sqrt(np.pi) / (N * (N - 1)) * np.sum(weights * gaps)
    sigma = np.sqrt((N / (N - 1)) * sigma_gap**2)

    return sigma


# Function to calculate circular median for RA values
def circular_median(ra_values):
    """
    Compute the circular median of RA values (in degrees), accounting for wrap-around at 360°.

    Parameters:
        ra_values (list or np.ndarray): List of RA values in degrees.

    Returns:
        float: Circular median RA in degrees.
    """
    ra_values = np.asarray(ra_values)

    # Angular distance function (circular)
    def angular_distance(a, b):
        diff = np.abs(a - b) % 360
        return np.minimum(diff, 360 - diff)

    # Total angular distance for each candidate
    total_distances = np.array([np.sum(angular_distance(ra_values, candidate)) for candidate in ra_values])

    # Find the minimum total distance
    min_total = np.min(total_distances)
    
    # Select all candidates with that distance
    candidates = ra_values[np.isclose(total_distances, min_total)]

    # If there are multiple candidates, compute the circular mean of the candidates
    if len(candidates) > 1:
        circular_mean = np.arctan2(np.mean(np.sin(np.radians(candidates))), np.mean(np.cos(np.radians(candidates))))
        return np.degrees(circular_mean) % 360

    # If there's only one candidate, return it directly
    return candidates[0] % 360


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


def Add_Properties(df):

    print("\n=== Re-Centering COORDINATES ===")
    mid_x = np.median(df['x'])
    mid_y = np.median(df['y'])
    mid_z = np.median(df['z'])
    df['x'] -= mid_x
    df['y'] -= mid_y
    df['z'] -= mid_z
    print(f"\n=== Complete Re-Centering COORDINATES ===")

    print("\n=== Making RA, DEC COORDINATES ===")
    r = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    df['ra'] = np.degrees(np.arctan2(df['y'], df['x'])) % 360
    df['dec'] = np.degrees(np.arcsin(df['z'] / r))  # Avoid division by zero
    print(f"\n=== Complete Making RA, DEC COORDINATES ===")

    print("\n=== zcos COORDINATES ===")
    n = 10000000
    max_r = np.max(r) * 1.1
    r_grid = np.linspace(1e-4, max_r, n)
    z_grid = np.zeros_like(r_grid)
    for i in tqdm(range(len(r_grid)), desc="Building redshift grid"):
        z_grid[i] = z_at_value(cosmo.comoving_distance, r_grid[i] * u.Mpc, method='bounded', zmax=0.18)
    df['zcos'] = np.interp(r, r_grid, z_grid)
    print(f"\n=== Complete zcos COORDINATES ===")

    print("\n=== zobs COORDINATES ===")
    mask = r != 0
    r_hat_x = np.zeros_like(df['x'])
    r_hat_y = np.zeros_like(df['y'])
    r_hat_z = np.zeros_like(df['z'])
    r_hat_x[mask] = df['x'][mask] / r[mask]
    r_hat_y[mask] = df['y'][mask] / r[mask]
    r_hat_z[mask] = df['z'][mask] / r[mask]
    v_radial = df['vx'] * r_hat_x + df['vy'] * r_hat_y + df['vz'] * r_hat_z
    c_km_s = const.c.to(u.km/u.s).value
    z_obs = (1 + df['zcos']) * (1 + v_radial / c_km_s) - 1
    df['zobs'] = np.where(z_obs <= 0, df['zcos'], z_obs)
    print(f"\n=== Complete zobs COORDINATES ===")

    print("\n=== Making apparant magnitudes ===")
    dl = cosmo.luminosity_distance(df['zcos']).value # Mpc
    dl *= 1e6  # Convert Mpc to pc
    df['u_app'] = df['u_abs'] + 5 * np.log10(dl / 10)
    df['g_app'] = df['g_abs'] + 5 * np.log10(dl / 10)
    df['r_app'] = df['r_abs'] + 5 * np.log10(dl / 10)
    df['i_app'] = df['i_abs'] + 5 * np.log10(dl / 10)
    df['z_app'] = df['z_abs'] + 5 * np.log10(dl / 10)
    print(f"\n=== Complete Making apparant magnitudes ===")

    return df


def Make_Catalogs(df):

    print("\n=== LINKING GALAXIES TO GROUPS ===")
    # 1. For each unique Mhalo, find galaxies within a linking length
    group_ids = np.full(len(df['Mhalo']), -1, dtype=int)
    current_group = 0
    group_to_galaxy_map = {}  # Maps group_id to list of galaxy indices

    unique_halos = np.unique(df['Mhalo'])
    print(f"Processing {len(unique_halos)} unique halo masses")

    for halo in tqdm(unique_halos, desc="Processing halos"):
        idx = np.where(df['Mhalo'] == halo)[0]
        if len(idx) == 0:
            continue
        positions = np.vstack([df['x'][idx], df['y'][idx], df['z'][idx]]).T

        # Determine linking length based on Mhalo value
        if halo < 11.5:
            r_link = 0.5
        elif 11.5 <= halo < 12.0:
            r_link = 0.75
        elif 12.0 <= halo < 13.0:
            r_link = 1.0
        elif 13.0 <= halo < 14.0:
            r_link = 3.5
        else:
            r_link = 10.0

        tree = cKDTree(positions)
        clusters = tree.query_ball_tree(tree, r=r_link)

        visited = set()
        for i in range(len(idx)):
            if idx[i] in visited:
                continue
            # Find all members connected within r_link
            group_members = set()
            to_visit = {i}
            while to_visit:
                current = to_visit.pop()
                if current in group_members:
                    continue
                group_members.add(current)
                neighbors = clusters[current]
                for neighbor in neighbors:
                    if neighbor not in group_members:
                        to_visit.add(neighbor)

            # Assign group IDs
            group_idx = [idx[member] for member in group_members]
            for member in group_members:
                group_ids[idx[member]] = current_group
                visited.add(idx[member])
            group_to_galaxy_map[current_group] = group_idx
            current_group += 1

    df['group_id'] = group_ids
    df['galaxy_id'] = np.arange(len(df))
    print(f"\n=== COMPLETED LINKING GALAXIES TO GROUPS with {current_group} groups ===")

    print("\n=== MAKING Field GALAXY CATALOG ===")
    # needs to have less than 3 galaxies to be a field galaxy
    df_field = df.groupby('group_id').filter(lambda x: len(x) < 3).reset_index(drop=True)
    df_field = df_field.drop(columns=['group_id'])
    print(f"\n=== Complete MAKING Field GALAXY CATALOG with {len(df_field)} galaxies ===")

    print("\n=== MAKING GROUP GALAXY CATALOG ===")
    # needs to have 3 or more galaxies to be a group
    df_group_galaxies = df.groupby('group_id').filter(lambda x: len(x) >= 3).reset_index(drop=True)
    print(f"\n=== Complete MAKING GROUP GALAXY CATALOG with {len(df_group_galaxies)} galaxies ===")

    print("\n=== ADDING GROUP PROPERTIES ===")
    # Nm - number of members in each group
    df_group_galaxies['Nm'] = df_group_galaxies.groupby('group_id')['group_id'].transform('count')
    df_group_galaxies['vel'] = (df_group_galaxies['zobs'] * const.c.to(u.km/u.s).value) / (1 + df_group_galaxies['zobs'])
    df_group_galaxies['ra_median'] = df_group_galaxies.groupby('group_id')['ra'].transform(circular_median)
    df_group_galaxies['dec_median'] = df_group_galaxies.groupby('group_id')['dec'].transform('median')
    df_group_galaxies['zobs_median'] = df_group_galaxies.groupby('group_id')['zobs'].transform('median')
    df_group_galaxies['zcos_median'] = df_group_galaxies.groupby('group_id')['zcos'].transform('median')

    c1 = SkyCoord(ra=df_group_galaxies['ra'], dec=df_group_galaxies['dec'], unit='deg', frame='icrs')
    c2 = SkyCoord(ra=df_group_galaxies['ra_median'], dec=df_group_galaxies['dec_median'], unit='deg', frame='icrs')
    df_group_galaxies['sep'] = ((2 * cosmo.comoving_distance(df_group_galaxies['zobs_median']) *
                                np.sin(c1.separation(c2).to(u.radian)/2))).value
    df_group_galaxies['sep_max'] = df_group_galaxies.groupby('group_id')['sep'].transform('max')
    df_group_galaxies['vdisp_gap'] = df_group_galaxies.groupby('group_id')['vel'].transform(disp_gap)
    df_group_galaxies['P3M'] = df_group_galaxies.groupby('group_id')['SM'].transform(lambda x: np.log10((10**x).nlargest(min(3, len(x))).sum()))
    df_group_galaxies['Mhalo_P3M'] = make_Mhalo_P3M(df_group_galaxies)
    df_group_galaxies['Mhalo_CVT'] = make_Mhalo_CVT(df_group_galaxies)
    df_group_galaxies['Mhalo_VT'] = np.log10( (5/3) * df_group_galaxies['vdisp_gap']**2 * df_group_galaxies['sep_max'] / G )
    print(f"\n=== Complete ADDING GROUP PROPERTIES ===")

    print("\n=== MAKING GROUP CATALOG ===")
    # Colums: ra (ra_median of groups), dec (dec_median of groups), zobs (zobs_median of groups), zcos (zcos_median of groups), Mhalo, Nm, sep_max, vdisp_gap, P3M, Mhalo_P3M, Mhalo_CVT, Mhalo_VT
    df_group_catalog = df_group_galaxies[['ra_median', 'dec_median', 'zobs_median', 'zcos_median', 'Mhalo', 'Nm', 'sep_max', 'vdisp_gap', 'P3M', 'Mhalo_P3M', 'Mhalo_CVT', 'Mhalo_VT']].copy()
    df_group_catalog['group_id'] = df_group_galaxies['group_id'].values
    df_group_catalog = df_group_catalog.drop_duplicates(subset=['group_id']).reset_index(drop=True)
    df_group_catalog.rename(columns={'ra_median': 'ra', 'dec_median': 'dec', 'zobs_median': 'zobs', 'zcos_median': 'zcos'}, inplace=True)
    print(f"\n=== Complete MAKING GROUP CATALOG with {len(df_group_catalog)} groups ===")

    print("\n=== Make zlim zobs<0.1 ===")
    # Use croup catalog to filter first zobs<0.1 - then only keep groups with group_id still in group catalog - group galaxies
    df_group_catalog = df_group_catalog[df_group_catalog['zobs'] < 0.1].reset_index(drop=True)
    df_group_galaxies = df_group_galaxies[df_group_galaxies['group_id'].isin(df_group_catalog['group_id'])].reset_index(drop=True)
    df_field = df_field[df_field['zobs'] < 0.1].reset_index(drop=True)
    print(f"\n=== Complete Make zlim zobs<0.1 ===")

    return df_field, df_group_galaxies, df_group_catalog



print("Starting GAEA_Extractor.py script")
start_time = time.time()

print("Loading data from GAEA Original FITS file")

df = Table.read("/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Original_Joined_1e8.fits")
df = df.to_pandas()

#rename columns to match expected names
df = df.rename(columns={
    'STELLAR_MASS': 'SM',
    'X': 'x',
    'Y': 'y',
    'Z': 'z',
    'VX': 'vx',
    'VY': 'vy',
    'VZ': 'vz',
    'M_HALO': 'Mhalo',
    'U_MAG_ABS': 'u_abs',
    'G_MAG_ABS': 'g_abs',
    'R_MAG_ABS': 'r_abs',
    'I_MAG_ABS': 'i_abs',
    'Z_MAG_ABS': 'z_abs'})

print("Data loaded successfully")


print("Adding properties to dataframe")
df = Add_Properties(df)
print("Properties added successfully")

print("Saving dataframe with added properties")
df_write = Table.from_pandas(df)
df_write.write(f"/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Added_Properties_1e8.fits", format='fits', overwrite=True)
print("Dataframe saved successfully")

print("Making Catalogs with i-band selection < 19.2")
df_field, df_group_galaxies, df_group_catalog = Make_Catalogs(df[(df['i_app'] <= 19.2) & (df['i_app'] > 0) & (df['zobs'] < 0.115) & (df['SM'] > 7.5)].reset_index(drop=True))
print("Catalogs made successfully")

print("Saving Catalogs")
df_field_write = Table.from_pandas(df_field)
df_field_write.write(f"/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Field_Catalog_1e8.fits", format='fits', overwrite=True)

df_group_galaxies_write = Table.from_pandas(df_group_galaxies)
df_group_galaxies_write.write(f"/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Group_Galaxies_1e8.fits", format='fits', overwrite=True)

df_group_catalog_write = Table.from_pandas(df_group_catalog)
df_group_catalog_write.write(f"/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Group_Catalog_1e8.fits", format='fits', overwrite=True)
print("Catalogs saved successfully")
