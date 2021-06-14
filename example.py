"""Configure and run model"""

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
# Example with climate inputs using a class

# -----------------------------------------------------------------------------
# Set up climate inputs class

# - Making a simple class allows us to (1) calculate relevant constants and read
# any key files (e.g. in __init__()) and (2) define a method to calculate (or 
# read) climate inputs at each timestep in calc_fields()
# - A class keeps all the important data and functionality together and gives
# a lot of flexibility
# - Example of creating spatial fields from one station time series using
# lapse rates etc (just for testing)
# - Could also adapt to read from a file(s) of arrays for each timestep (e.g. a
# netcdf file) or to take a simulated spatial field from elsewhere (e.g. the
# random mixing code)

climate_input_path = 'Z:/DP/Work/HBV/Tests/climate_inputs.csv'

class Climate(object):
    """Read and/or calculate climate inputs.
    
    Attributes:
        orog_fac (ndarray): Precipitation multiplication factor [-]
        tadj (ndarray): Elevation adjustment for temperature [K]
        pr (ndarray): Precipitation [mm timestep-1]
        rf (ndarray): Rainfall [mm timestep-1]
        sf (ndarray): Snowfall [mm timestep-1]
        tas (ndarray): Near-surface air temperature [K]
        pet (ndarray): Potential evapotranspiration [mm timestep-1]
    """
    
    def __init__(self, climate_input_path, ny, nx, elev):
        """Read initial inputs and calculate helper constants/variables."""
        # Station time series
        self.df = pd.read_csv(
            climate_input_path, index_col='datetime', dtype=np.float32, 
            parse_dates=True, dayfirst=True
        )
        
        # Precipitation (orographic) factor array
        zmax = 5000.0
        betasub = 0.35
        ksub = 3.0
        betasuper = 3.0
        ksuper  = 5.0
        rfac = 1.9
        zbase = 2394.0
        self.orog_fac = np.zeros((ny, nx), dtype=np.float32)
        self.orog_fac[elev<=zmax] = betasub * ((elev[elev<=zmax] * (zbase**-1)) ** ksub)
        self.orog_fac[elev>zmax] = betasuper * ((zmax * (elev[elev>zmax]**-1)) ** ksuper)
        self.orog_fac *= rfac
        
        # Temperature elevation adjustment array
        self.tadj = np.zeros((ny, nx), dtype=np.float32)
        self.tadj[:] = (elev - zbase) * -0.0065
        
        # Melting/freezing threshold temperature
        self.tm = 273.15
    
    def calc_fields(self, date):
        """Calculate spatial fields of climate inputs for timestep.
        
        Args:
            date (datetime): Date/time of required climate fields
        """
        self.tas = self.tadj + np.float32(self.df.loc[self.df.index == date, 'tas'].values[0])
        self.pr = self.orog_fac * np.float32(self.df.loc[self.df.index == date, 'pr'].values[0])
        self.rf = np.where(self.tas > self.tm, self.pr, 0.0)
        self.sf = self.pr - self.rf
        # Very crude approximation of PET for testing
        ##self.pet = self.tas * 0.0 # !EDIT
        self.pet = np.where(
            ((self.tas - self.tm) >= -20.0) & ((self.tas - self.tm) < -2.0),
            (self.tas - self.tm) * 0.01 + 0.22,
            0.0 # self.pet
        )
        self.pet = np.where(
            (self.tas - self.tm) >= -2.0,
            (self.tas - self.tm) * 0.3 + 0.8,
            self.pet
        )

# -----------------------------------------------------------------------------
# Define model class

# - This is a class that inherits from hbv.BaseModel, where a couple of methods
# are overridden to allow us to pass climate inputs to the model at each
# timestep (see Climate class above)
# - If we define some conventions we might be able to skip this part in future,
# but it is needed in the initial version of the code

class Model(hbv.BaseModel):
    
    def init_climate_obj(self):
        self.ci = Climate(climate_input_path, ny, nx, elev)
    
    def get_climate_inputs(self):
        self.ci.calc_fields(self.date)
        self.pr[:] = self.ci.pr[:]
        self.rf[:] = self.ci.rf[:]
        self.sf[:] = self.ci.sf[:]
        self.tas[:] = self.ci.tas[:]
        self.pet[:] = self.ci.pet[:]

# -----------------------------------------------------------------------------
# Set up and run model

# Make a dictionary of required variables
# - i.e. {'dt': dt, 'start_date': start_date, ...}
setup_dict = dict_of(
    # Timestep, simulation period, grid details, fixed input arrays
    dt, start_date, end_date, nx, ny, dx, mask, elev, flen, cell_order,
    # Parameters
    icf, lpf, fc, ttm, cfmax, cfr, whc, beta, perc, cflux, k, alpha, k1, tau,
    ssm, ssc, ssa, sshdm,
)

m = Model(**setup_dict)
m.run_model()

output_path = 'Z:/DP/Work/HBV/Tests/out_y1.csv'
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


