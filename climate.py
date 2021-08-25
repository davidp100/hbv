# No accounting for missing data currently

import numpy as np
import pandas as pd

class Climate(object):
    """Read and calculate climate inputs.
    
    Approach follows MicroMet: (1) adjust gauge observations to a reference
    elevation using elevation gradients, (2) interpolate gauge observations on
    reference elevation, (3) (re)introduce elevation signal using elevation
    gradients. Main differences are (a) using IDW interpolation rather than
    objective analysis and (b) using CRHM equation for adjusting precipitation
    for elevation. 
    
    Attributes:
        station_details (pd.DataFrame): Dataframe of station metadata/file paths
        elevation_gradients (dict of dicts): Elevation gradients by variable and month
        elev (ndarray): Cell elevations [m]
        mask (ndarray): Array marking inside (1) and outside (0) catchment
        ny (int): Number of grid cells in north-south direction
        nx (int): Number of grid cells in east-west direction
        idw_exp (float): Exponent for IDW weight calculations
        
        pr (ndarray): Precipitation [mm timestep-1]
        rf (ndarray): Rainfall [mm timestep-1]
        sf (ndarray): Snowfall [mm timestep-1]
        tas (ndarray): Near-surface air temperature [K]
        pet (ndarray): Potential evapotranspiration [mm timestep-1]
        
        tm (float): Melting/freezing temperature [K]
        
        stations (list of str): List of station names/IDs
        station_series (dict of pd.DataFrame): Station climate time series
        station_variables (dict of lists): List of variables available at each station
        station_weights (dict of ndarray): Weights arrays for IDW calculations
        ref_elev (float): Reference elevation for interpolation
    """
    
    def __init__(self, station_details, elevation_gradients, elev, mask, ny, nx, idw_exp):
        """
        Args:
            station_details (pd.DataFrame) Dataframe of station metadata/file paths
            elevation_gradients (dict): Dictionary (of dictionaries) containing elevation gradients
            elev (ndarray): Cell elevations [m]
            mask (ndarray): Array marking inside (1) and outside (0) catchment
            ny (int): Number of grid cells in north-south direction
            nx (int): Number of grid cells in east-west direction
            idw_exp (float): Exponent for IDW weight calculations
        """
        # Assign arguments to attributes
        self.station_details = station_details
        self.elevation_gradients = elevation_gradients
        self.elev = elev
        self.ny = ny
        self.nx = nx
        self.idw_exp = idw_exp
        
        # Initialise (2d) arrays for climate variables (to be filled each timestep)
        self.pr = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.rf = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.sf = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.tas = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.pet = np.zeros((self.ny, self.nx), dtype=np.float32)
        
        # Other attributes
        self.tm = 273.15
        
        # Derived attributes
        self.stations = self.station_details['Station'].tolist()
        
        # Read station time series into a dictionary indexed by station name/ID
        # - also identify variables available for each station
        self.station_series = {}
        self.station_variables = {}
        for index, row in self.station_details.iterrows():
            station = row['Station']
            input_path = row['Path']
            self.station_series[station] = pd.read_csv(
                input_path, index_col='datetime', dtype=np.float32, 
                parse_dates=True, dayfirst=True
            )
            self.station_variables[station] = []
            for variable in ['pr', 'tas', 'pet']:
                if variable in self.station_series[station].columns:
                    self.station_variables[station].append(variable)
        
        # Weights based on distance and a decay function/parameter (i.e. IDW)
        # - based on arrays of distance from each station (cell) to each other cell
        # - one weights array per station
        self.station_weights = {}
        for index, row in self.station_details.iterrows():
            station = row['Station']
            yi = row['YI']
            xi = row['XI']
            dist = distmat_v2(self.pr, (yi, xi))
            dist[dist == 0.0] = 0.0000001 # account for zero distance at station
            self.station_weights[station] = 1.0 / (dist ** self.idw_exp)
        
        # Normalise station weights so sum of weights is one
        # - simplifies IDW calculations
        for station in self.stations:
            if self.stations.index(station) == 0:
                sum_weights = self.station_weights[station].copy()
            else:
                sum_weights += self.station_weights[station]
        for station in self.stations:
            self.station_weights[station] /= sum_weights
        
        # Reference elevation (taken as catchment mean elevation)
        self.ref_elev = np.around(np.mean(elev[mask == 1]))
    
    def calc_fields(self, date):
        """Calculate spatial fields of climate inputs for timestep.
        
        Args:
            date (datetime): Date/time of required climate fields
        """
        # Fill climate arrays with zeros
        self.pr.fill(0.0)
        self.rf.fill(0.0)
        self.sf.fill(0.0)
        self.tas.fill(0.0)
        self.pet.fill(0.0)
        
        # Get station values for timestep
        station_vals = {
            'pr': {}, 'tas': {}, 'pet': {}
        }
        for station in self.stations:
            df = self.station_series[station]
            for variable in ['pr', 'tas', 'pet']:
                if variable in self.station_variables[station]:
                    station_vals[variable][station] = (
                        np.float32(df.loc[df.index == date, variable].values[0])
                    )
        
        # Adjust station values to reference elevation
        station_vals_ref = {
            'pr': {}, 'tas': {}, 'pet': {}
        }
        for index, row in self.station_details.iterrows():
            station = row['Station']
            station_elev = row['Elevation']
            for variable in ['pr', 'tas', 'pet']:
                if variable in self.station_variables[station]:
                    if variable == 'tas':
                        method = 1
                    else:
                        method = 2
                    station_vals_ref[variable][station] = elevation_adjustment(
                        station_vals[variable][station],
                        self.elevation_gradients[variable][date.month], 
                        station_elev, self.ref_elev, method
                    )
        
        # Interpolate adjusted station values
        for station in self.stations:
            if 'pr' in self.station_variables[station]:
                self.pr += (self.station_weights[station] * station_vals_ref['pr'][station])
            if 'tas' in self.station_variables[station]:
                self.tas += (self.station_weights[station] * station_vals_ref['tas'][station])
            if 'pet' in self.station_variables[station]:
                self.pet += (self.station_weights[station] * station_vals_ref['pet'][station])
        
        # Apply elevation gradients (i.e. adjust from reference elevation to
        # actual (DEM) elevations)
        self.pr = elevation_adjustment(
            self.pr, self.elevation_gradients['pr'][date.month], self.ref_elev, 
            self.elev, method=2
        )
        self.tas = elevation_adjustment(
            self.tas, self.elevation_gradients['tas'][date.month], self.ref_elev,
            self.elev, method=1
        )
        self.pet = elevation_adjustment(
            self.pet, self.elevation_gradients['pet'][date.month], self.ref_elev, 
            self.elev, method=2
        )
        
        # Set precipitation below a (low) threshold to zero
        # - could be made a function of timestep
        self.pr[self.pr < 0.01] = 0.0
        
        # Rainfall and snowfall partitioning
        self.rf[:] = np.where(self.tas > self.tm, self.pr, 0.0)
        self.sf[:] = self.pr - self.rf

def distmat_v2(a, index):
    """Calculate distance of a point from all points in 2d array.
    
    https://stackoverflow.com/questions/61628380/calculate-distance-from-all-points-in-numpy-array-to-a-single-point-on-the-basis
    """
    i,j = np.indices(a.shape, sparse=True)
    return np.sqrt((i-index[0])**2 + (j-index[1])**2, dtype=np.float32)

def elevation_adjustment(x, gradient, elevation, target_elevation, method):
    """Adjust a value from its elevation to a target elevation.
    
    Method (1) is for temperature-like variables. Method (2) follows the CRHM
    approach to precipitation adjustment.
    
    Args:
        x (float or ndarray) Value to adjust
        gradient (float) Gradient for adjustment
        elevation (float or ndarray) Elevation associated with value
        target_elevation (float or ndarray) Target elevation to adjust to
        method (int) Flag to indicate form of function to use
    """
    if method == 1:
        x_target = x + gradient * (target_elevation - elevation)
    elif method == 2:
        x_target = x * (1.0 + gradient * (target_elevation - elevation) / 100.0)
        x_target = np.maximum(x_target, 0.0)
    return x_target




