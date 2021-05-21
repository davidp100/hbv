"""Utility functions.

"""

import numpy as np
import numpy.ma as ma

def read_asc(file_path, data_type=np.float32, return_metadata=True):
    """Read ascii raster and return array (optionally with metadata)."""
    # Headers
    dc = {}
    with open(file_path, 'r') as fh:
        for i in range(6):
            line = fh.readline()
            key, val = line.rstrip().split()
            dc[key] = val
    nx = int(dc['ncols'])
    ny = int(dc['nrows'])
    xll = float(dc['xllcorner'])
    yll = float(dc['yllcorner'])
    dx = float(dc['cellsize'])
    dy = float(dc['cellsize'])
    nodata = float(dc['NODATA_value'])
    
    # Values array
    arr = np.loadtxt(file_path, dtype=data_type, skiprows=6)
    arr = ma.masked_values(arr, nodata)
    
    if return_metadata:
        return(arr, nx, ny, xll, yll, dx, dy)
    else:
        return(arr)

def spatial_mean(val_arr, mask_arr):
    """Calculate spatial mean of values where mask == 1."""
    sm = np.mean(val_arr[mask_arr==1])
    return(sm)





