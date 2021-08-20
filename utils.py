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

def write_asc(array, file_path, format, nx, ny, xll, yll, dx, nodata_value):
    """Write ascii raster to file."""
    headers = (
        'ncols         ' + str(nx) + '\n' +
        'nrows         ' + str(ny) + '\n' +
        'xllcorner     ' + str(xll) + '\n' +
        'yllcorner     ' + str(yll) + '\n' +
        'cellsize      ' + str(dx) + '\n' +
        'NODATA_value  ' + str(nodata_value)
    )
    if ma.isMaskedArray(array):
        output_array = array.filled(nodata_value)
    else:
        output_array = array
    np.savetxt(file_path, output_array, fmt=format, header=headers, comments='')

def spatial_mean(val_arr, mask_arr):
    """Calculate spatial mean of values where mask == 1."""
    sm = np.mean(val_arr[mask_arr==1])
    return(sm)





