import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from astropy.table import Table
from astropy.io import fits
from scipy.special import erf, erfinv
from astropy.constants import c
from astropy import units as u 
from astropy.constants import G
from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import Angle

G = G.to(u.M_sun**-1 * u.km**2 * u.s**-2 * u.Mpc).value
# Cosmological parameters
h = 0.7
cosmo = FlatLambdaCDM(H0=100*h, Om0=0.3121, Ob0=0.0491, Tcmb0=2.725)

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='large')
plt.rc('axes', labelsize='large')
plt.rc('axes', titlesize='large')
plt.rc('ytick', labelsize='large')
plt.rc('legend', fontsize='x-small')



df = Table.read('/fred/oz004/wvankemp/Halo_Mass_Relations/Data/Obs_Data/Original/SGP_WISE_Combined_2dF_GAMA.fits')
df = df.to_pandas()

# Choose a wrap center so the 0°/360° seam is moved away from your data
# For SGP (RA ~340° to 26°), 330° places the seam at 330° and makes the data contiguous.
WRAP_CENTER_DEG = 330.0

# Wrap RA to [0, 360) but with the seam at WRAP_CENTER_DEG rather than 0°.
# This keeps values non-negative while removing the visual break around 0°/360°.
# Example: with center=330°, 340° -> 10°, 26° -> 56° (continuous range 10–56).
ra_plot = pd.Series(
    ((df['ra'].values - WRAP_CENTER_DEG) % 360.0),
    index=df.index,
    name='ra_plot'
)


# Define G23 sky box and samples
ra = df['ra']
dec = df['dec']
g23_area = (ra >= 339) & (ra <= 351) & (dec >= -35) & (dec <= -30)
gama_g23 = g23_area & (~df['MAG_AUTO_I'].isna()) & (df['LogSmass'] < 15)
two_df = (~df['BJG'].isna()) & (df['LogSmass'] < 15)
two_df_g23 = g23_area & (~df['BJG'].isna()) & (df['LogSmass'] < 15)

# Exclusive categories (handy for other plots)
mask_2df_not_g23 = two_df #& ~g23_area                                # red
mask_gama_only = gama_g23 & ~two_df_g23                              # yellow
mask_2df_g23_only = two_df_g23 & ~gama_g23                           # green
mask_both_g23 = two_df_g23 & gama_g23                                # chartreuse
# -------- Multi-panel plot (3 datasets) --------

# Define the RA window in TRUE RA (0–360), displayed decreasing left→right.
RA_LEFT_TRUE = 26.0    # left side label
RA_RIGHT_TRUE = 340.0  # right side label

def format_ra_axis(ax, left_true=RA_LEFT_TRUE, right_true=RA_RIGHT_TRUE,
                   wrap_center=WRAP_CENTER_DEG, tick_step=10.0):
    """Apply wrapped 0–360 RA axis with descending left→right labels.
    Shows the window [left_true → right_true] as contiguous after wrapping.
    """
    x_left_wrapped = ((left_true - wrap_center) % 360.0)
    x_right_wrapped = ((right_true - wrap_center) % 360.0)
    # Limits first (we invert later to show decreasing RA)
    ax.set_xlim(x_right_wrapped, x_left_wrapped)
    # Ticks across the visible wrapped extent
    lo, hi = (x_left_wrapped, x_right_wrapped) if x_left_wrapped <= x_right_wrapped else (x_right_wrapped, x_left_wrapped)
    ticks_wrapped = np.arange(lo, hi + 1e-6, tick_step)
    ax.set_xticks(ticks_wrapped)
    ax.set_xticklabels([f"{((t + wrap_center) % 360):.0f}" for t in ticks_wrapped])
    ax.invert_xaxis()  # RA decreases to the right
    ax.grid(True)

# Build 1x3 panel layout
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
ax_sgp2df, ax_g232df, ax_g23gama = axes
plt.subplots_adjust(wspace=0)

# Panel 1: SGP - 2dF (outside G23)
ax_sgp2df.scatter(ra_plot[mask_2df_not_g23], df.loc[mask_2df_not_g23, 'dec'],
                  s=2, c='red', alpha=0.7)
ax_sgp2df.set_title(f'SGP - 2dF [{mask_2df_not_g23.sum()}]')
format_ra_axis(ax_sgp2df)

# Panel 2: G23 - 2dF (all 2dF in G23, including overlap with GAMA)
ax_g232df.scatter(ra_plot[two_df_g23], df.loc[two_df_g23, 'dec'],
                  s=2, c='green', alpha=0.7)
ax_g232df.set_title(f'G23 - 2dF [{two_df_g23.sum()}]')
format_ra_axis(ax_g232df)

# Panel 3: G23 - GAMA (all GAMA in G23, including overlap with 2dF)
ax_g23gama.scatter(ra_plot[gama_g23], df.loc[gama_g23, 'dec'],
                   s=2, c='orange', alpha=0.7)
ax_g23gama.set_title(f'G23 - GAMA [{gama_g23.sum()}]')
format_ra_axis(ax_g23gama)

# Labels
ax_sgp2df.set_ylabel('Dec [deg]')
for ax in axes:
    ax.set_xlabel('RA [deg]')


plt.savefig('/fred/oz004/wvankemp/Halo_Mass_New/Plots/SGP_Dataset_three_panels.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()