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

print("Starting GAEA_Extractor.py script")
start_time = time.time()

# Function to calculate Gapper dispersion
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

# Function to calculate halo mass from stellar mass
def S2HM(Mc, A, M_A, beta, gamma):
    """
    Calculate halo mass from stellar mass using the given parameters.
    
    Parameters:
        Mc (float or array): Log10 of stellar mass
        A, M_A, beta, gamma: Model parameters
        
    Returns:
        float or array: Log10 of halo mass
    """
    Mc = 10**Mc  # Convert log values back to linear scale
    Mh = A * Mc * ((Mc / (10**M_A))**(beta) + (Mc / (10**M_A))**gamma)
    return np.log10(Mh)

# Function to calculate M200 from velocity dispersion
def M200_Disp_Cor(vd, rad):
    """
    Calculate M200 using velocity dispersion and max_sep corrections.
    
    Parameters:
        vd (float or array): Velocity dispersion
        rad (float or array): Radius (max separation)
        
    Returns:
        float or array: Log10 of M200
    """
    alpha = 1.030
    vd_lim = 244.634
    n1 = -1.989
    beta = 0.213
    rad_lim = 0.369
    n2 = -1.591
    
    # Use np.where for vectorized conditional logic
    Ab = np.where(vd < vd_lim, alpha * ((vd / vd_lim)**n1 - 1), 0)
    Ac = np.where(rad < rad_lim, beta * ((rad / rad_lim)**n2 - 1), 0)
    A = 5 / 3 + Ab + Ac

    return np.log10(A * vd**2 * rad / G)

# --- Load FITS data ---
print("\n=== LOADING DATA ===")
fits_path = "/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Original_Joined_1e8.fits"
print(f"Loading data from: {fits_path}")
with fits.open(fits_path) as hdul:
    data = hdul[1].data
    x = data['x']
    y = data['y']
    z = data['z']
    vx = data['vx']
    vy = data['vy']
    vz = data['vz']
    Mhalo = data['M_Halo']
    
    # Check if I_MAG_ABS exists in the data
    has_magnitudes = 'I_MAG_ABS' in data.names
    if has_magnitudes:
        I_MAG_ABS = data['I_MAG_ABS']
        print(f"Found I_MAG_ABS column in data")
    else:
        print(f"No I_MAG_ABS column found in data")
    
    # Keep all columns in original table
    all_columns = data.names
    print(f"Loaded {len(data)} galaxies with {len(all_columns)} columns")

print("\n=== PROCESSING COORDINATES ===")
mid_x = np.median(x)
mid_y = np.median(y)
mid_z = np.median(z)
x = x - mid_x
y = y - mid_y
z = z - mid_z
data['x'] = x
data['y'] = y
data['z'] = z
print(f"Centered coordinates around the median position: ({mid_x:.2f}, {mid_y:.2f}, {mid_z:.2f})")

# --- Calculate dist, ra, dec ---
print("Calculating distance, RA, and Dec...")
r = np.sqrt(x**2 + y**2 + z**2)

# RA calculation
ra = np.empty_like(x)
ra = np.degrees(np.arctan2(y, x)) % 360

# Dec calculation
dec = np.degrees(np.arcsin(z / r))
print("Finished calculating distance, RA, and Dec")

# --- Calculate z_cos ---
print("\n=== CALCULATING REDSHIFTS ===")
print("Building redshift interpolation grid...")
n = 100000
max_r = np.max(r) * 1.1
r_grid = np.linspace(1e-4, max_r, n)
z_grid = np.zeros_like(r_grid)
for i in tqdm(range(len(r_grid)), desc="Building redshift grid"):
    z_grid[i] = z_at_value(cosmo.comoving_distance, r_grid[i] * u.Mpc, method='bounded', zmax=1.5)
z_cos = np.interp(r, r_grid, z_grid)
print("Finished calculating cosmological redshifts")

# --- Calculate z_obs ---
print("Calculating observed redshifts with peculiar velocities...")
mask = r != 0
r_hat_x = np.zeros_like(x)
r_hat_y = np.zeros_like(y)
r_hat_z = np.zeros_like(z)
r_hat_x[mask] = x[mask] / r[mask]
r_hat_y[mask] = y[mask] / r[mask]
r_hat_z[mask] = z[mask] / r[mask]
v_radial = vx * r_hat_x + vy * r_hat_y + vz * r_hat_z
c_km_s = const.c.to(u.km/u.s).value
z_obs = (1 + z_cos) * (1 + v_radial / c_km_s) - 1
z_obs = np.where(z_obs <= 0, z_cos, z_obs)
print("Finished calculating observed redshifts")

# --- Calculate I-band apparent magnitude if I_MAG_ABS exists ---
i_mag_app = None
if has_magnitudes:
    print("\n=== CALCULATING APPARENT MAGNITUDES ===")
    # Calculate luminosity distance in Mpc
    dl = cosmo.luminosity_distance(z_cos).value
    
    # Distance modulus: DM = 5*log10(dl/10pc)
    # 10pc in Mpc = 1e-5
    dist_mod = 5 * np.log10(dl / 1e-5)
    
    # Calculate apparent magnitude
    i_mag_app = I_MAG_ABS + dist_mod
    print("Finished calculating I-band apparent magnitudes")

# --- Save all galaxies with added properties ---
print("\n=== SAVING DATA WITH ADDED PROPERTIES ===")
t = Table(data)
t['dist'] = r
t['ra'] = ra
t['dec'] = dec
t['z_cos'] = z_cos
t['z_obs'] = z_obs
if has_magnitudes and i_mag_app is not None:
    t['I_MAG_APP'] = i_mag_app

output_path = "/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Added_Properties_1e8.fits"
t.write(output_path, overwrite=True)
print(f"Saved all galaxies with added properties to: {output_path}")

# --- Apply magnitude cut if magnitudes exist ---
print("\n=== APPLYING MAGNITUDE CUT ===")
mask_mag_cut = np.ones(len(r), dtype=bool)  # Default: keep all galaxies
if has_magnitudes and i_mag_app is not None:
    mag_limit = 19.2  # Set your desired apparent magnitude limit here
    mask_mag_cut = i_mag_app < mag_limit
    print(f"Applied I-band magnitude cut at {mag_limit}")
    print(f"Kept {np.sum(mask_mag_cut)} out of {len(mask_mag_cut)} galaxies ({np.sum(mask_mag_cut)/len(mask_mag_cut)*100:.1f}%)")

    # Apply magnitude cut to all arrays
    x = x[mask_mag_cut]
    y = y[mask_mag_cut]
    z = z[mask_mag_cut]
    vx = vx[mask_mag_cut]
    vy = vy[mask_mag_cut]
    vz = vz[mask_mag_cut]
    Mhalo = Mhalo[mask_mag_cut]
    r = r[mask_mag_cut]
    ra = ra[mask_mag_cut]
    dec = dec[mask_mag_cut]
    z_cos = z_cos[mask_mag_cut]
    z_obs = z_obs[mask_mag_cut]
    i_mag_app = i_mag_app[mask_mag_cut]
    
    # For the data object, we'll create a filtered version for grouping
    data_filtered = Table(data[mask_mag_cut])
else:
    print("No magnitude cut applied (no magnitude data available)")
    data_filtered = data

# --- Group catalog creation ---
print("\n=== CREATING GROUP CATALOG ===")
print("Finding galaxies in groups based on halo mass...")
# 1. For each unique Mhalo, find galaxies within a linking length
group_ids = np.full(len(Mhalo), -1, dtype=int)
current_group = 0
group_to_galaxy_map = {}  # Maps group_id to list of galaxy indices

unique_halos = np.unique(Mhalo)
print(f"Processing {len(unique_halos)} unique halo masses")

for halo in tqdm(unique_halos, desc="Processing halos"):
    idx = np.where(Mhalo == halo)[0]
    if len(idx) == 0:
        continue
    positions = np.vstack([x[idx], y[idx], z[idx]]).T

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

print(f"Created {current_group} distinct groups")

# After creating the groups but before calculating group properties
print("\n=== FILTERING GROUPS ===")
print("Ensuring groups have at least 3 members...")

# Count groups more efficiently
print("Counting members in each group...")
unique_groups, group_counts = np.unique(group_ids[group_ids >= 0], return_counts=True)
print(f"Found {len(unique_groups)} total groups before filtering")

# Create mapping of group_id to counts
group_count_map = dict(zip(unique_groups, group_counts))

# Find small groups efficiently
small_groups = [gid for gid, count in group_count_map.items() if count < 3]
print(f"Found {len(small_groups)} groups with fewer than 3 members (will be removed)")

# Create set of small groups for faster lookup
small_groups_set = set(small_groups)

# More efficient way to update group IDs
print("Reassigning galaxies in small groups...")
group_ids = np.array([gid if gid < 0 or gid not in small_groups_set else -1 for gid in group_ids])

# Update the mapping (you can do this in a separate step if it's slow)
print("Updating group mapping...")
for gid in small_groups:
    if gid in group_to_galaxy_map:
        del group_to_galaxy_map[gid]

# Report number of valid groups remaining
valid_groups = len(unique_groups) - len(small_groups)
print(f"After filtering, {valid_groups} groups remain with ≥3 members")

# Create a pandas DataFrame for easier group calculations
print("\n=== CALCULATING GROUP PROPERTIES ===")
if has_magnitudes and i_mag_app is not None:
    # Use the indices from the magnitude-filtered dataset
    galaxy_indices = np.arange(len(Mhalo))
    galaxy_df = pd.DataFrame({
        'galaxy_index': galaxy_indices,
        'group_id': group_ids,
        'ra': ra,
        'dec': dec,
        'z_cos': z_cos,
        'z_obs': z_obs,
        'Mhalo': Mhalo,
        'x': x,
        'y': y,
        'z': z,
        'vx': vx,
        'vy': vy,
        'vz': vz,
        'dist': r,
        'I_MAG_APP': i_mag_app
    })
    
    # Reference to the filtered data for output
    data_for_output = data_filtered
else:
    # No magnitude filtering was applied
    galaxy_indices = np.arange(len(Mhalo))
    galaxy_df = pd.DataFrame({
        'galaxy_index': galaxy_indices,
        'group_id': group_ids,
        'ra': ra,
        'dec': dec,
        'z_cos': z_cos,
        'z_obs': z_obs,
        'Mhalo': Mhalo,
        'x': x,
        'y': y,
        'z': z,
        'vx': vx,
        'vy': vy,
        'vz': vz,
        'dist': r
    })
    
    # Reference to the original data for output
    data_for_output = data_filtered

# Separate galaxies in groups
galaxies_in_groups = galaxy_df[galaxy_df['group_id'] >= 0].copy()  # Add .copy() to avoid the warnings
print(f"Found {len(galaxies_in_groups)} galaxies in {len(galaxies_in_groups['group_id'].unique())} groups")

# Calculate group-level statistics similar to Shark_Extractor
if not galaxies_in_groups.empty:
    print("Calculating group statistics (medians, dispersions, separations)...")
    # Calculate median values for each group
    # Use circular_median for RA to handle the 0/360 boundary
    galaxies_in_groups.loc[:, 'ra_median'] = galaxies_in_groups.groupby('group_id')['ra'].transform(
        lambda x: circular_median(x)
    )
    galaxies_in_groups.loc[:, 'dec_median'] = galaxies_in_groups.groupby('group_id')['dec'].transform('median')
    galaxies_in_groups.loc[:, 'z_obs_median'] = galaxies_in_groups.groupby('group_id')['z_obs'].transform('median')
    galaxies_in_groups.loc[:, 'z_cos_median'] = galaxies_in_groups.groupby('group_id')['z_cos'].transform('median')
    
    # Calculate velocities for velocity dispersion
    galaxies_in_groups.loc[:, 'vel'] = (galaxies_in_groups['z_obs'] * c_km_s) / (1 + galaxies_in_groups['z_obs'])
    
    # Calculate velocity dispersion using Gapper method
    print("Calculating velocity dispersions with Gapper method...")
    galaxies_in_groups.loc[:, 'vdisp_gap'] = galaxies_in_groups.groupby('group_id')['vel'].transform(disp_gap)
    
    # Calculate separations from median position
    print("Calculating separations from group centers...")
    c1 = SkyCoord(ra=galaxies_in_groups['ra'].values * u.degree, dec=galaxies_in_groups['dec'].values * u.degree, frame='icrs')
    c2 = SkyCoord(ra=galaxies_in_groups['ra_median'].values * u.degree, dec=galaxies_in_groups['dec_median'].values * u.degree, frame='icrs')
    
    # Calculate the separation of each galaxy from the group's central position
    galaxies_in_groups.loc[:, 'sep'] = ((2 * cosmo.comoving_distance(galaxies_in_groups['z_obs_median']) * 
                                np.sin(c1.separation(c2).to(u.radian)/2))).value
    
    # Calculate maximum separation within each group
    galaxies_in_groups.loc[:, 'sep_max'] = galaxies_in_groups.groupby('group_id')['sep'].transform('max')
    
    # Count members in each group
    galaxies_in_groups.loc[:, 'n_members'] = galaxies_in_groups.groupby('group_id')['group_id'].transform('count')
    
    # Get Mhalo for each group (using the median for consistency)
    galaxies_in_groups.loc[:, 'Mhalo_group'] = galaxies_in_groups.groupby('group_id')['Mhalo'].transform('median')
    print("Finished calculating all group properties")

    # Create group galaxies output - including all original columns
    print("\n=== SAVING GROUP GALAXY CATALOG ===")
    group_galaxies_table = Table()
    
    # First, add the rows from the original data for galaxies in groups
    original_indices = galaxies_in_groups['galaxy_index'].values
    
    # Add all columns from the original data
    print("Adding original columns to group galaxy catalog...")
    for col_name in data_for_output.colnames:
        group_galaxies_table[col_name] = data_for_output[col_name][original_indices]
    
    # Add the computed group properties
    print("Adding computed group properties to group galaxy catalog...")
    group_galaxies_table['group_id'] = galaxies_in_groups['group_id'].values
    group_galaxies_table['dist'] = galaxies_in_groups['dist'].values
    group_galaxies_table['ra'] = galaxies_in_groups['ra'].values
    group_galaxies_table['dec'] = galaxies_in_groups['dec'].values
    group_galaxies_table['z_cos'] = galaxies_in_groups['z_cos'].values
    group_galaxies_table['z_obs'] = galaxies_in_groups['z_obs'].values
    group_galaxies_table['ra_median'] = galaxies_in_groups['ra_median'].values
    group_galaxies_table['dec_median'] = galaxies_in_groups['dec_median'].values
    group_galaxies_table['z_obs_median'] = galaxies_in_groups['z_obs_median'].values
    group_galaxies_table['z_cos_median'] = galaxies_in_groups['z_cos_median'].values
    group_galaxies_table['vel'] = galaxies_in_groups['vel'].values
    group_galaxies_table['vdisp_gap'] = galaxies_in_groups['vdisp_gap'].values
    group_galaxies_table['sep'] = galaxies_in_groups['sep'].values
    group_galaxies_table['sep_max'] = galaxies_in_groups['sep_max'].values
    group_galaxies_table['n_members'] = galaxies_in_groups['n_members'].values
    
    # Add apparent magnitude if available
    if has_magnitudes and i_mag_app is not None:
        group_galaxies_table['I_MAG_APP'] = galaxies_in_groups['I_MAG_APP'].values
    
    # Save the group galaxies file
    group_galaxies_path = "/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Group_Galaxies_1e8.fits"
    # group_galaxies_table.write(group_galaxies_path, overwrite=True)
    print(f"Saved {len(group_galaxies_table)} galaxies in groups to: {group_galaxies_path}")

    # Create group catalog
    print("\n=== SAVING GROUP CATALOG ===")
    group_summary = galaxies_in_groups.drop_duplicates('group_id')
    group_catalog = pd.DataFrame({
        'group_id': group_summary['group_id'],
        'ra_median': group_summary['ra_median'],
        'dec_median': group_summary['dec_median'],
        'z_obs_median': group_summary['z_obs_median'],
        'z_cos_median': group_summary['z_cos_median'],
        'vdisp_gap': group_summary['vdisp_gap'],
        'sep_max': group_summary['sep_max'],
        'n_members': group_summary['n_members'],
        'Mhalo': group_summary['Mhalo_group']
    })
    
    # Save the group catalog
    group_catalog_path = "/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Group_Catalog_1e8.fits"
    group_table = Table.from_pandas(group_catalog)
    group_table.write(group_catalog_path, overwrite=True)
    print(f"Saved {len(group_table)} groups to: {group_catalog_path}")

print("\n=== CALCULATING STELLAR MASS PROPERTIES ===")
print("Checking for stellar mass column...")

# Check if the stellar mass column exists (may have different names in different simulations)
stellar_mass_col = None
possible_names = ['stellar_mass', 'Stellar_Mass', 'SM', 'M_star', 'Mstar', 'M_Star', 'M_STAR', 'STELLAR_MASS']

for col_name in possible_names:
    if col_name in data_for_output.colnames:
        stellar_mass_col = col_name
        print(f"Found stellar mass column: {col_name}")
        break

if stellar_mass_col is None:
    print("WARNING: No stellar mass column found. SM and SM3 will not be calculated.")
else:
    try:
        print("Reading stellar masses and converting to native byte order...")
        # Convert to native byte order to avoid endian issues
        stellar_masses = np.array(data_for_output[stellar_mass_col][original_indices], dtype=np.float64)
        
        # Add to galaxies_in_groups DataFrame
        print("Adding stellar masses to group galaxies...")
        galaxies_in_groups['SM'] = stellar_masses
        
        # Use vectorized operations for calculating CSM and CSM3 - much faster approach
        print("Calculating CSM (maximum stellar mass in each group)...")
        galaxies_in_groups.loc[:, 'CSM'] = galaxies_in_groups.groupby('group_id')['SM'].transform('max')
        
        print("Calculating CSM3 (sum of top 3 stellar masses in each group)...")
        galaxies_in_groups.loc[:, 'CSM3'] = galaxies_in_groups.groupby('group_id')['SM'].transform(
            lambda x: np.log10((10**x).nlargest(min(3, len(x))).sum())
        )
        
        print("Finished calculating stellar mass properties")
        
        # Add these columns to the group galaxies table
        print("Adding stellar mass properties to group galaxy catalog...")
        group_galaxies_table['SM'] = stellar_masses
        group_galaxies_table['CSM'] = galaxies_in_groups['CSM'].values
        group_galaxies_table['CSM3'] = galaxies_in_groups['CSM3'].values
        
        # Re-save the group galaxies file
        print("Re-saving group galaxies catalog with stellar mass properties...")
        group_galaxies_path = "/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Group_Galaxies_1e8.fits"
        # group_galaxies_table.write(group_galaxies_path, overwrite=True)
        
        # Add these columns to the group catalog using more efficient pandas operations
        print("Adding stellar mass properties to group catalog...")
        
        # Instead of looping through each group ID (which is slow), use pandas groupby
        print("Creating group summary with stellar mass properties...")
        csm_by_group = galaxies_in_groups.groupby('group_id')['CSM'].first().reset_index()
        csm3_by_group = galaxies_in_groups.groupby('group_id')['CSM3'].first().reset_index()
        
        # Merge these with the group catalog using fast dataframe operations
        print("Merging with group catalog...")
        group_catalog = pd.merge(group_catalog, csm_by_group, on='group_id', how='left')
        group_catalog = pd.merge(group_catalog, csm3_by_group, on='group_id', how='left')
        
        # Re-save the group catalog
        print("Re-saving group catalog with stellar mass properties...")
        group_catalog_path = "/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Group_Catalog_1e8.fits"
        group_table = Table.from_pandas(group_catalog)
        group_table.write(group_catalog_path, overwrite=True)
        print(f"Saved {len(group_table)} groups to: {group_catalog_path}")
        
        print("\n=== CALCULATING ADDITIONAL HALO MASS ESTIMATES ===")

        # Calculate Mhalo_P3M using stellar mass-halo mass relation
        print("Calculating Mhalo_P3M from CSM3...")

        # Add Mhalo_P3M to group galaxies dataframe
        galaxies_in_groups.loc[:, 'Mhalo_P3M'] = S2HM(
            galaxies_in_groups['CSM3'], 46.944, 10.483, 0.249, -0.601)
        
        # Add to group galaxies table
        group_galaxies_table['Mhalo_P3M'] = galaxies_in_groups['Mhalo_P3M'].values
        
        # Calculate Mhalo_Disp using velocity dispersion and separation
        print("Calculating Mhalo_Disp from velocity dispersion and separation...")
        valid_mask = (galaxies_in_groups['vdisp_gap'] > 0) & (galaxies_in_groups['sep_max'] > 0)
        galaxies_in_groups.loc[:, 'Mhalo_CVT'] = np.nan
        galaxies_in_groups.loc[valid_mask, 'Mhalo_CVT'] = M200_Disp_Cor(
            galaxies_in_groups.loc[valid_mask, 'vdisp_gap'],
            galaxies_in_groups.loc[valid_mask, 'sep_max']
        )
        galaxies_in_groups.loc[:, 'Mhalo_VT'] = np.nan
        galaxies_in_groups.loc[valid_mask, 'Mhalo_VT'] = np.log10(5/3 * galaxies_in_groups.loc[valid_mask, 'vdisp_gap']**2 * galaxies_in_groups.loc[valid_mask, 'sep_max'] / G)
        
        # Add to group galaxies table
        group_galaxies_table['Mhalo_CVT'] = galaxies_in_groups['Mhalo_CVT'].values
        group_galaxies_table['Mhalo_VT'] = galaxies_in_groups['Mhalo_VT'].values

        # Add these columns to group catalog as well
        mhalo_p3m_by_group = galaxies_in_groups.groupby('group_id')['Mhalo_P3M'].first().reset_index()
        Mhalo_CVT_by_group = galaxies_in_groups.groupby('group_id')['Mhalo_CVT'].first().reset_index()
        Mhalo_VT_by_group = galaxies_in_groups.groupby('group_id')['Mhalo_VT'].first().reset_index()

        # Merge with group catalog
        group_catalog = pd.merge(group_catalog, mhalo_p3m_by_group, on='group_id', how='left')
        group_catalog = pd.merge(group_catalog, Mhalo_CVT_by_group, on='group_id', how='left')
        group_catalog = pd.merge(group_catalog, Mhalo_VT_by_group, on='group_id', how='left')

        merge_cols = ['group_id', 'vdisp_gap', 'sep_max', 'Mhalo_P3M', 'Mhalo_VT', 'Mhalo_CVT']
        group_galaxies_table = group_galaxies_table.merge(group_catalog[merge_cols], on='group_id', how='left')
        print(group_galaxies_table.columns)
        
        # Re-save the files with the new halo mass columns
        print("Re-saving catalogs with additional halo mass estimates...")
        group_galaxies_path = "/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Group_Galaxies_1e8.fits"
        group_galaxies_table.write(group_galaxies_path, overwrite=True)

        group_catalog_path = "/fred/oz004/wvankemp/Halo_Mass_New/Data/GAEA/GAEA_Group_Catalog_1e8.fits"
        group_table = Table.from_pandas(group_catalog)
        group_table.write(group_catalog_path, overwrite=True)
        
        print("Finished calculating additional halo mass estimates")
            
        
    except Exception as e:
        print(f"ERROR calculating stellar mass properties: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Continuing without stellar mass properties...")

# Calculate and print total execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"\n=== SCRIPT COMPLETED ===")
print(f"Total execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
print(f"Files saved:")
print(f"1. GAEA_Added_Properties_1e8.fits - All galaxies with added properties")
if not galaxies_in_groups.empty:
    print(f"2. GAEA_Group_Galaxies_1e8.fits - Galaxies in groups with group properties")
    print(f"3. GAEA_Group_Catalog_1e8.fits - Group catalog with group properties")