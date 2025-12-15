import numpy as np
import matplotlib.pyplot as plt
import pickle

print("Loading processed data...")
# Load the processed data
with open('/fred/oz004/wvankemp/Halo_Mass_New/Data/Densities/processed_density_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract all the variables
SHARK_dens2d = data['SHARK_dens2d']
SHARK_dens2d_1 = data['SHARK_dens2d_1']
SHARK_weighted_dens2d = data['SHARK_weighted_dens2d']
SHARK_masked_weighted_dens2d = data['SHARK_masked_weighted_dens2d']
SHARK_xgrid = data['SHARK_xgrid']
SHARK_ygrid = data['SHARK_ygrid']

SAGE_dens2d = data['SAGE_dens2d']
SAGE_dens2d_1 = data['SAGE_dens2d_1']
SAGE_weighted_dens2d = data['SAGE_weighted_dens2d']
SAGE_masked_weighted_dens2d = data['SAGE_masked_weighted_dens2d']
SAGE_xgrid = data['SAGE_xgrid']
SAGE_ygrid = data['SAGE_ygrid']

GAEA_dens2d = data['GAEA_dens2d']
GAEA_dens2d_1 = data['GAEA_dens2d_1']
GAEA_weighted_dens2d = data['GAEA_weighted_dens2d']
GAEA_masked_weighted_dens2d = data['GAEA_masked_weighted_dens2d']
GAEA_xgrid = data['GAEA_xgrid']
GAEA_ygrid = data['GAEA_ygrid']

x_min = data['x_min']
x_max = 12.5
y_min = data['y_min']
y_max = data['y_max']
threshold = data['threshold']

print("Creating plots...")

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
ax_SHARK.set_xlim(x_min, x_max)
ax_SHARK.set_ylim(y_min, y_max)
ax_SHARK.set_title('SHARK', fontsize=16)
# ax_SHARK.scatter(SHARK['log_mstar_total'], SHARK['mvir_hosthalo'], c='k', s=1, alpha=0.02)
ax_SHARK.contour(SHARK_xgrid, SHARK_ygrid, SHARK_dens2d.T, colors='k', linewidths=1, levels=12, zorder=-3, linestyles='solid', alpha=0.4)
cf_1 = ax_SHARK.contourf(SHARK_xgrid, SHARK_ygrid, SHARK_masked_weighted_dens2d.T, cmap='coolwarm', zorder=-4, vmin=0, vmax=1, levels=256)

# Density plot for SAGE
ax_SAGE.minorticks_on()
ax_SAGE.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                 bottom=True, top=True, left=True, right=True, direction='in')
ax_SAGE.set_xlabel(r'$\log$ M$_{\star}$ [M$_{\odot}$]')
ax_SAGE.set_xlim(x_min, x_max)
ax_SAGE.set_ylim(y_min, y_max)
ax_SAGE.set_title('SAGE', fontsize=16)
# Plot 1 in 100 points for SAGE
# ax_SAGE.scatter(SAGE['Total_Stellar_Mass'].iloc[scatter_indices], SAGE['Central_Galaxy_Mvir'].iloc[scatter_indices], c='k', s=1, alpha=0.02)
ax_SAGE.contour(SAGE_xgrid, SAGE_ygrid, SAGE_dens2d.T, colors='k', linewidths=1, levels=12, zorder=-3, linestyles='solid', alpha=0.4)
cf_2 = ax_SAGE.contourf(SAGE_xgrid, SAGE_ygrid, SAGE_masked_weighted_dens2d.T, cmap='coolwarm', zorder=-4, vmin=0, vmax=1, levels=256)

# # Density plot for GAEA
ax_GAEA.minorticks_on()
ax_GAEA.tick_params(which='both', labelbottom=True, labeltop=False, labelleft=False, labelright=False,
                 bottom=True, top=True, left=True, right=True, direction='in')
ax_GAEA.set_xlabel(r'$\log$ M$_{\star}$ [M$_{\odot}$]')
ax_GAEA.set_xlim(x_min, x_max)
ax_GAEA.set_ylim(y_min, y_max)
ax_GAEA.set_title('GAEA', fontsize=16)
# ax_GAEA.scatter(GAEA['STELLAR_MASS'].iloc[scatter_indices], GAEA['M_HALO'].iloc[scatter_indices], c='k', s=1, alpha=0.02)
ax_GAEA.contour(GAEA_xgrid, GAEA_ygrid, GAEA_dens2d.T, colors='k', linewidths=1, levels=12, zorder=-3, linestyles='solid', alpha=0.4)
cf_3 = ax_GAEA.contourf(GAEA_xgrid, GAEA_ygrid, GAEA_masked_weighted_dens2d.T, cmap='coolwarm', zorder=-4, vmin=0, vmax=1, levels=256)

# Colorbar for density plots
cbar = fig.colorbar(cf_1, ax=[ax_SHARK, ax_SAGE, ax_GAEA], pad=0.12, location='bottom', aspect=50)
cbar.set_label('Quenched Fraction', rotation=0, labelpad=0)
cbar.set_ticks(np.arange(0, 1.2, 0.2))
 
plt.savefig('/fred/oz004/wvankemp/Halo_Mass_New/Plots/QF_Mstar_Mhalo_Sims.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

print("Plot saved successfully!")