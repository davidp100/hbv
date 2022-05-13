"""Configure and run model using Climate class defined in climate.py"""

import os
import sys
import datetime

import numpy as np
import pandas as pd
from sorcery import dict_of

import utils
import hbv

# =============================================================================
# Set/get some basic simulation info and model parameters

# -----------------------------------------------------------------------------
# Define timestep, simulation period, grid and fixed properties

dt = 86400
start_date = datetime.datetime(1998, 9, 1, 0, 0, 0)
end_date = datetime.datetime(2003, 12, 31, 23, 59, 59)

elev_path = 'Z:/DP/Work/HBV/Tests/Astore_DTMF.asc'
mask_path = 'Z:/DP/Work/HBV/Tests/Astore_MASK.asc'
flen_path = 'Z:/DP/Work/HBV/Tests/Astore_FLEN.asc'

elev, nx, ny, xll, yll, dx, dy = utils.read_asc(elev_path)
mask = utils.read_asc(mask_path, data_type=np.int, return_metadata=False)
flen = utils.read_asc(flen_path, return_metadata=False)

# For gravitational redistribution of snow
cell_order_path = 'Z:/DP/Work/HBV/Tests/Astore_DS.csv'
cell_order = pd.read_csv(cell_order_path)

# -----------------------------------------------------------------------------
# Get variables needed for climate inputs

# Read in station details as dataframe
# - assumes headers are: Station, YI, XI, Elevation, Path
stations_path = 'Z:/DP/Work/HBV/Tests/Astore_Stations.csv'
station_details = pd.read_csv(stations_path)

# Dictionary of elevation gradients by variable and by month
# - pr units are as per CRHM [1/100m]
# - tas units are [K/m]
# - pet units initially assumed to follow pr
# - sign convention:
#       positive = increase with elevation
#       negative = decrease with elevation
elevation_gradients = {
    'pr': {
        1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01, 5: 0.01, 6: 0.01,
        7: 0.01, 8: 0.01, 9: 0.01, 10: 0.01, 11: 0.01, 12: 0.01
    },
    'tas': {
        1: -0.0065, 2: -0.0065, 3: -0.0065, 4: -0.0065, 5: -0.0065, 6: -0.0065,
        7: -0.0065, 8: -0.0065, 9: -0.0065, 10: -0.0065, 11: -0.0065, 12: -0.0065
    },
    'pet': {
        1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0,
        7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0
    }
}

# Exponent for inverse-distance weighting interpolation step
idw_exp = 2.0

# -----------------------------------------------------------------------------
# Set parameters (initial and/or fixed values)

# - These parameters could also be set as numpy arrays (i.e. to permit spatially
# varying parameter values)
# - Also possible to do things like use python optimisation routines to
# calibrate parameters...

icf = 2.0
lpf = 0.9
fc = 250.0
ttm = 273.15
cfmax = 3.0
cfr = 0.05
whc = 0.1
beta = 2.0
perc = 3.0
cflux = 2.0
k = 0.03
alpha = 0.7
k1 = 0.01
tau = 1.0 / 86400.0

# For gravitational redistribution of snow
ssm = 25.0
ssc = 20000.0
ssa = -0.08
sshdm = 5.0

# =============================================================================
# Set up and run model

# Make a dictionary of required variables
# - i.e. {'dt': dt, 'start_date': start_date, ...}
setup_dict = dict_of(
    # Timestep, simulation period, grid details, fixed input arrays
    dt, start_date, end_date, nx, ny, dx, mask, elev, flen, cell_order,
    # For climate setup
    station_details, elevation_gradients, idw_exp,
    # Parameters
    icf, lpf, fc, ttm, cfmax, cfr, whc, beta, perc, cflux, k, alpha, k1, tau,
    ssm, ssc, ssa, sshdm,
)

m = hbv.BaseModel(**setup_dict)
m.run_model()

output_path = 'Z:/DP/Work/HBV/Tests/output.csv'
m.df_cat.to_csv(output_path)

# =============================================================================
# Miscellaneous

# -----------------------------------------------------------------------------
# Examples of parameter bounds

# https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015WR018247
# https://www.tandfonline.com/doi/full/10.1080/02626667.2018.1466056
# https://www.cambridge.org/core/journals/annals-of-glaciology/article/runoff-modelling-in-an-arctic-unglaciated-catchment-fuglebekken-spitsbergen/B3E1F5055A9736D0494EB1AF4F3CED26
# par_bnds = [
#     (0.0, 3.0), # icf
#     (0.3, 1.0), # lpf
#     (50.0, 700.0), # fc
#     (272.15, 274.15), # ttm
#     (0.5, 5.0), # cfmax
#     (0, 0.1), # cfr
#     (0.0, 0.2), # whc
#     (1.0, 6.0), # beta
#     (0.5, 6.0), # perc
#     (0.0, 4.0), # cflux
#     (0.01, 0.9), # k
#     (0.1, 3.0), # alpha
#     (0.001, 0.1), # k1
#     (0.1 / 86400.0, 10.0 / 86400.0) # tau
# ]


