import numpy as np
from astropy.table import Table
import seaborn as sns
import pandas as pd

# Load the parquet files into Pandas dataframe
df = pd.read_parquet('/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/waves_wide_groups.parquet', engine='pyarrow')
df = Table.from_pandas(df)

# Save the table to a FITS file
df.write('/fred/oz004/wvankemp/Halo_Mass_New/Data/SHARK/waves_wide_groups.fits', format='fits', overwrite=True)