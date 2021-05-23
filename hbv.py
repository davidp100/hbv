"""Minimal spatially distributed HBV model.

This module contains a class for the basic HBV-96 model largely following the
wflow_hbv python implementation.

"""

import datetime

import numpy as np
import pandas as pd

import utils

class BaseModel(object):
    """Minimal spatially distributed HBV model.
    
    Attributes:
        dt (int): Timestep [s]
        start_date (datetime): Simulation start datetime
        end_date (datetime): Simulation end datetime
        nx (int): Number of grid cells in east-west direction
        ny (int): Number of grid cells in north-south direction
        dx (float): Grid cell spacing (same in both directions) [m]
        mask (ndarray): Array marking inside (1) and outside (0) catchment
        elev (ndarray): Cell elevations [m]
        flen (ndarray): Cell distances to (main) outlet [m]
        cat_ncells (float): Number of cells in catchment
        sim_dates (list of datetime): Date/time at each timestep of simulation
        date (datetime): Date/time of current timestep
        
        icf (float): Maximum interception storage [mm]
        lpf (float): Fractional soil moisture limit for aet=pet [-]
        fc (float): Field capacity [mm]
        ttm (float): Melt threshold temperature [K]
        cfmax (float): Snow temperature index melt factor 
            [mm K-1 timestep-1]
        cfr (float): Snow refreezing factor [-]
        whc (float): Fractional snow water holding capacity [-]
        beta (float): Soil seepage exponent [-]
        perc (float): Maximum percolation rate through soil [mm timestep-1]
        cflux (float): Maximum capillary flux [mm timestep-1]
        k (float): Coefficient for upper zone outflow [-]
        alpha (float): Exponent for upper zone outflow [-]
        k1 (float): Coefficient for lower zone outflow [-]
        tau (float): Travel speed (number of timesteps to travel 1 m) 
            [timestep m-1]
        
        lp (float): Soil moisture limit for aet=pet [mm]
        rlag (ndarray): Time lag [timestep(s)]
        
        nlags (int): Total number of lags (from 0 to maximum lag)
        rlag_3d (ndarray): 3D array for fractional lags at each time lag [timestep(s)]
        
        out_vars (list of str): Output variable names
        df_cat (pandas.DataFrame): Catchment output time series
        
        incps (ndarray): Interception storage [mm]
        snws (ndarray): Snowpack solid (ice) water equivalent [mm]
        snwl (ndarray): Snowpack liquid (water) water equivalent [mm]
        sm (ndarray): Soil moisture [mm]
        uz (ndarray): Upper zone storage [mm]
        lz (ndarray): Lower zone storage [mm]
        swe (ndarray): Snowpack total (solid + ice) water equivalent [mm]
        
        pr (ndarray): Precipitation [mm timestep-1]
        rf (ndarray): Rainfall [mm timestep-1]
        sf (ndarray): Snowfall [mm timestep-1]
        tas (ndarray): Near-surface air temperature [K]
        pet (ndarray): Potential evapotranspiration [mm timestep-1]
        pr_sc (ndarray): Precipitation reaching surface (i.e. after 
            interception) [mm timestep-1]
        rf_sc (ndarray): Rainfall reaching surface (i.e. after
            interception) [mm timestep-1]
        sf_sc (ndarray): Snowfall reaching surface (i.e. after 
            interception) [mm timestep-1]
        
        dc_roff (dict): Runoff at catchment outlet by date [mm timestep-1]
        s0 (ndarray): Storage at beginning of timestep
        s1 (ndarray): Storage at end of timestep
        ds (ndarray): Storage change for timestep [mm]
        
        aet (ndarray): Actual evapotranspiration [mm timestep-1]
        melt (ndarray): Melt [mm timestep-1]
        sm_in (ndarray): Inflow to soil moisture storage [mm timestep-1]
        uz_in (ndarray): Inflow to upper zone storage [mm timestep-1]
        
        roff_nr (ndarray): Unrouted runoff for timestep [mm timestep-1]
    """
    
    def __init__(self, **kwargs):
        """Initialise model."""
        self.init_basic(**kwargs)
        self.init_params(**kwargs)
        self.init_outputs(**kwargs)
        self.init_storages() #! Use kwargs and remove self.set_storages()?
        self.set_storages()
        self.init_climate_arrays()
        self.init_helper_vars()
        self.init_climate_obj()
    
    def init_basic(self, **kwargs):
        """Initialise timestep, simulation period, grid and domain definitions.
        
        Keyword Args:
            dt (int): Timestep [s]
            start_date (datetime): Simulation start datetime
            end_date (datetime): Simulation end datetime
            nx (int): Number of grid cells in east-west direction
            ny (int): Number of grid cells in north-south direction
            dx (float): Grid cell spacing (same in both directions) [m]
            mask (ndarray): Array marking inside (1) and outside (0) catchment
            elev (ndarray): Cell elevations [m]
            flen (ndarray): Cell distances to (main) outlet [m]
        """
        self.dt = kwargs['dt']
        self.start_date = kwargs['start_date']
        self.end_date = kwargs['end_date']
        self.nx = kwargs['nx']
        self.ny = kwargs['ny']
        self.dx = kwargs['dx']
        self.elev = kwargs['elev']
        self.mask = kwargs['mask']
        self.flen = kwargs['flen']
        
        self.cat_ncells = np.float32(np.sum(self.mask))
        
        self.sim_dates = []
        d = self.start_date
        while d <= self.end_date:
            self.sim_dates.append(d)
            d += datetime.timedelta(seconds=self.dt)
        
        self.date = self.start_date
    
    def init_params(self, **kwargs):
        """Set parameter values.
        
        If not updated these parameter values will be fixed throughout the
        simulation. The parameters can be specified as single values or arrays.
        Several derived parameters are calculated from the input (argument) 
        parameters (related to routing, as well as soil ET).
        
        Keyword Args:
            icf (float): Maximum interception storage [mm]
            lpf (float): Fractional soil moisture limit for aet=pet [-]
            fc (float): Field capacity [mm]
            ttm (float): Melt threshold temperature [K]
            cfmax (float): Snow temperature index melt factor 
                [mm K-1 timestep-1]
            cfr (float): Snow refreezing factor [-]
            whc (float): Fractional snow water holding capacity [-]
            beta (float): Soil seepage exponent [-]
            perc (float): Maximum percolation rate through soil [mm timestep-1]
            cflux (float): Maximum capillary flux [mm timestep-1]
            k (float): Coefficient for upper zone outflow [-]
            alpha (float): Exponent for upper zone outflow [-]
            k1 (float): Coefficient for lower zone outflow [-]
            tau (float): Travel speed (number of timesteps to travel 1 m) 
                [timestep m-1]
        """
        # Basic parameters
        self.icf = kwargs['icf']
        self.lpf = kwargs['lpf']
        self.fc = kwargs['fc']
        self.ttm = kwargs['ttm']
        self.cfmax = kwargs['cfmax']
        self.cfr = kwargs['cfr']
        self.whc = kwargs['whc']
        self.beta = kwargs['beta']
        self.perc = kwargs['perc']
        self.cflux = kwargs['cflux']
        self.k = kwargs['k']
        self.alpha = kwargs['alpha']
        self.k1 = kwargs['k1']
        self.tau = kwargs['tau']
        
        # Derived parameters
        self.lp = self.fc * self.lpf
        self.rlag = (np.max(self.flen) - self.flen) * self.tau
        self.rlag[self.mask == 0] = 0.0
    
    def init_outputs(self, **kwargs):
        """Initialise variables controlling and recording simulation outputs.
        
        Precipitation, ET and runoff will be added to the list if not present.
        
        Keyword Args:
            out_vars (list of str): Output variable names
        """
        if 'out_vars' in kwargs.keys():
            out_vars = kwargs['out_vars']
            if 'pr' not in out_vars:
                out_vars.append('pr')
            if 'aet' not in out_vars:
                out_vars.append('aet')
            if 'roff' not in out_vars:
                out_vars.append('roff')
        else:
            out_vars = [
                'pr', 'aet', 'melt', 'roff', 'swe', 'sm', 'uz', 'lz', 'incps', 
                'sca', 'ds', 'mb'
            ]
        self.out_vars = out_vars
        self.df_cat = pd.DataFrame(
            data = 0.0,
            index = self.sim_dates,
            columns = self.out_vars
        )
    
    def init_storages(self):
        """Initialise storage (state variable) arrays.
        
        SWE is provided for convenience, although it is redundant as both solid
        and liquid components of total snowpack storage are tracked. Initial 
        non-zero values for selected storages are assigned here, but these can 
        be overriden using self.set_storages().
        """
        self.incps = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.snws = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.snwl = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.sm = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.uz = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.lz = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.swe = np.zeros((self.ny, self.nx), dtype=np.float32)
        
        # Initial storage values where non-zero
        self.sm[:].fill(self.fc)
        self.uz.fill(0.2 * self.fc)
        self.lz.fill(0.33 * self.k1)
    
    def init_climate_arrays(self):
        """Initialise arrays for climate input fields."""
        self.pr = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.rf = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.sf = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.tas = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.pet = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.pr_sc = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.rf_sc = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.sf_sc = np.zeros((self.ny, self.nx), dtype=np.float32)
    
    def init_helper_vars(self):
        """Initialise helper variables."""
        # Runoff helper variables
        self.dc_roff = {}  # outflow_date: runoff
        max_lag = int(np.max(np.ceil(self.rlag).astype(np.int)))
        self.nlags = max_lag + 1
        self.rlag_3d = np.zeros((self.nlags, self.ny, self.nx), dtype=np.float32)
        self.rlag_3d -= 999.0
        for lag in range(self.nlags):
            if lag < max_lag:
                self.rlag_3d[lag,:,:] = np.where(
                    (self.rlag >= np.float32(lag)) & (self.rlag < np.float32(lag+1.0)),
                    1.0 - (self.rlag - np.floor(self.rlag)),
                    self.rlag_3d[lag,:,:]
                )
                self.rlag_3d[lag+1,:,:] = np.where(
                    (self.rlag >= np.float32(lag)) & (self.rlag < np.float32(lag+1.0)),
                    self.rlag - np.floor(self.rlag),
                    self.rlag_3d[lag+1,:,:]
                )
        self.rlag_3d = np.maximum(self.rlag_3d, 0.0)
        self.rlag_3d[:,self.mask == 0] = 0.0
        
        # Storage and mass balance check helper variables
        self.ds = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.s0 = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.s1 = np.zeros((self.ny, self.nx), dtype=np.float32)
    
    def init_climate_obj(self):
        """Initialise user-defined climate input object (optional).
        
        Optional to help pass climate input fields to model at each timestep in
        conjunction with self.get_climate_inputs() method.
        """
        pass
    
    def set_storages(self):
        """Override initial storage values if needed."""
        pass
    
    def run_model(self):
        """Simulate all timesteps."""
        while self.date <= self.end_date:
            self.s0 = (
                self.incps + self.snws + self.snwl + self.sm + self.uz + self.lz
            )
            self.simulate_timestep()
            self.date += datetime.timedelta(seconds=self.dt)
    
    def simulate_timestep(self):
        """Simulate one timestep."""
        self.get_climate_inputs()
        
        # Initialise sub-canopy precipitation (i.e. reaching surface after 
        # interception) - will be modified
        self.pr_sc[:] = self.pr[:]
        self.rf_sc[:] = self.rf[:]
        self.sf_sc[:] = self.sf[:]
        
        self.update_params()
        
        self.simulate_interception()
        self.simulate_evapotranspiration()
        self.simulate_snowpack()
        self.simulate_soil_moisture()
        self.simulate_runoff()
        self.simulate_routing()
        
        self.get_outputs()
    
    def get_climate_inputs(self):
        """Get climate input arrays.
        
        Required as user-defined method to populate self.pr, self.rf, self.sf,
        self.tas, self.pet. May use object initialised in 
        self.init_climate_obj() if required. 
        """
        pass
    
    def update_params(self):
        """Update parameter values for timestep if needed."""
        pass
    
    def simulate_interception(self):
        """Simulate canopy interception of precipitation.
        
        Fill interception storage up to maximum defined by self.icf parameter.
        Use snowfall fraction to partition between snowfall and rainfall.
        """
        if np.max(self.pr) > 0.0:
            sf_frac = self.sf / self.pr
            sf_frac = np.minimum(sf_frac, 1.0)
            sf_frac = np.maximum(sf_frac, 0.0)
            incp = np.minimum(self.pr, self.icf - self.incps)
            self.incps += incp
            self.pr_sc -= incp
            self.sf_sc -= (incp * sf_frac)
            self.rf_sc = self.pr_sc - self.sf_sc
    
    def simulate_evapotranspiration(self):
        """Calculate actual evapotranspiration."""
        # Interception evapotranspiration
        incp_et = np.minimum(self.incps, self.pet)
        self.incps -= incp_et
        
        # Soil evapotranspiration
        soil_pet = (self.pet - incp_et)
        soil_aet = np.where(
            self.sm >= self.lp, 
            np.minimum(self.sm, soil_pet),
            np.minimum(soil_pet, self.pet * self.sm / self.lp) #! Why not soil_pet in second argument?
        )
        self.sm -= soil_aet
        
        self.aet = incp_et + soil_aet
    
    def simulate_snowpack(self):
        """Simulate snowpack accumulation, melt and refreezing.
        
        Rainfall entering the soil moisture component is added to snow melt 
        here too.
        """
        # Potential melt and refreezing (i.e. if unlimited snowpack)
        self.melt = np.where(
            self.tas > self.ttm,
            self.cfmax * (self.tas - self.ttm),
            0.0
        )
        refr = np.where(
            self.tas < self.ttm,
            self.cfmax * self.cfr * (self.ttm - self.tas),
            0.0
        )
        
        # Limiting to snowpack solid/liquid mass and water-holding capacity
        self.melt = np.minimum(self.melt, self.snws)
        refr = np.minimum(refr, self.snwl)
        self.snws += self.sf_sc + refr - self.melt
        max_snwl = self.snws * self.whc
        self.snwl += self.melt + self.rf_sc - refr
        
        # Inflow to soil moisture storage
        self.sm_in = np.maximum(self.snwl - max_snwl, 0.0)
        self.snwl -= self.sm_in
        
        self.swe = self.snws + self.snwl
    
    def simulate_soil_moisture(self):
        """Simulate soil moisture.
        
        If field capacity is reached, excess water becomes direct runoff, which
        means that it is passed as an input to the upper zone storage, along 
        with seepage through the soil. An adjustment is applied that ensures 
        soil moisture is filled to capacity before direct runoff occurs.
        """
        # First estimate of direct runoff
        dir_roff = np.maximum(self.sm + self.sm_in - self.fc, 0.0)
        self.sm += self.sm_in
        self.sm -= dir_roff
        self.sm_in -= dir_roff
        
        # Seepage to upper zone storage
        seep = np.minimum(self.sm / self.fc, 1.0) ** self.beta * self.sm_in
        self.sm -= seep
        sm_fill = np.minimum(self.fc - self.sm, dir_roff)
        
        # Update direct runoff and soil moisture balance
        dir_roff -= sm_fill
        self.sm += sm_fill
        self.uz_in = dir_roff + seep
    
    def simulate_runoff(self):
        """Simulate runoff generation.
        
        Inflow from soil enters upper zone, from which some percolation to the
        lower zone occurs. Capillary flux returns some water to soil moisture. 
        Upper zone outflow calculated using HBV-96 approach (rather than more
        complicated additional option in wflow_hbv). 
        """
        # Percolation to lower zone
        self.uz += self.uz_in
        perc = np.minimum(self.perc, self.uz - (self.uz_in / 2.0)) #! /2.0 
        self.uz -= perc
        
        # Capillary flux
        cap_flux = self.cflux * ((self.fc - self.sm) / self.fc)
        cap_flux = np.minimum(self.uz, cap_flux)
        cap_flux = np.minimum(self.fc - self.sm, cap_flux)
        self.uz -= cap_flux
        self.sm += cap_flux
        
        # Upper zone outflow
        uz_out = np.minimum(
            np.where(
                perc < self.perc, #! no quickflow generated if percolation is below max
                0.0,
                self.k * (self.uz - np.minimum(self.uz_in / 2.0, self.uz)) 
                    ** (1.0 + self.alpha) #! /2.0
            ), 
            self.uz
        )
        self.uz -= uz_out #! Check vs wflow
        
        # Lower zone inflow/outflow
        self.lz += perc
        lz_out = np.minimum(self.lz, self.k1 * self.lz)
        self.lz -= lz_out
        
        # Unrouted runoff
        self.roff_nr = uz_out + lz_out
    
    def simulate_routing(self):
        """Simulate runoff routing.
        
        Adapted from: https://gmd.copernicus.org/articles/13/6093/2020/ . The 
        method is based on calculating a time lag for runoff to reach the
        catchment outlet depending on the distance of a cell from the outlet.
        So for each time lag, add unrouted runoff to the total runoff on the 
        relevant outflow date, which is stored in the self.dc_roff dictionary.
        In this implementation, the time lag is split between the lower and
        upper bounding integers according to its fractional component (e.g.
        for a time lag of 1.25 timesteps, 75% of the unrouted runoff is 
        assigned to lag=1 and 25% to lag=2.
        """
        roff_r = self.rlag_3d * self.roff_nr
        for lag in range(self.nlags):
            lag = int(lag)
            roff_date = self.date + datetime.timedelta(seconds=(lag * self.dt))
            roff_t = np.sum(roff_r[lag,:,:]) / self.cat_ncells
            if roff_date not in self.dc_roff.keys():
                self.dc_roff[roff_date] = roff_t
            else:
                self.dc_roff[roff_date] += roff_t
    
    def check_mb(self, pr, aet, roff):
        """Check catchment mass balance.
        
        Note that catchment runoff needs to be subtracted from self.ds, because
        unrouted runoff is added to storage at end of time step (i.e. self.s1)
        as routing/channel storage effectively, but it is not removed there 
        when outflow occurs.
        """
        self.s1 = (
                self.incps + self.snws + self.snwl + self.sm + self.uz + self.lz
                + self.roff_nr
            )
        self.ds = self.s1 - self.s0
        ds_cat = utils.spatial_mean(self.ds, self.mask) - roff
        self.mb = pr - aet - roff - ds_cat
        return(ds_cat)
    
    def get_outputs(self):
        """Calculate and store catchment outputs in dataframe.
        
        Mass balance check is called here to avoid duplicate calculations of
        spatial means.
        """
        out_vals = []
        
        pr = utils.spatial_mean(self.pr, self.mask)
        aet = utils.spatial_mean(self.aet, self.mask)
        roff = self.dc_roff[self.date]
        
        ds_cat = self.check_mb(pr, aet, roff)
        
        for var in self.out_vars:
            if var == 'pr':
                out_vals.append(pr)
            elif var == 'aet':
                out_vals.append(aet)
            elif var == 'roff':
                out_vals.append(roff)
            elif var == 'ds':
                out_vals.append(ds_cat)
            elif var == 'sca':
                #! Hardcoded threshold (assuming swe in mm)
                sca = (
                    np.sum(self.mask[np.logical_and(self.mask==1, self.swe>0.1)])
                    / self.cat_ncells
                )
                out_vals.append(sca)
            elif var == 'mb':
                out_vals.append(self.mb)
            else:
                out_vals.append(utils.spatial_mean(getattr(self, var), self.mask))
        idx = self.sim_dates.index(self.date)
        self.df_cat.iloc[idx] = out_vals




