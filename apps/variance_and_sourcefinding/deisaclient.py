import dask.array as da
import numpy
import yaml
import os
import json
import sys
import time

from radioimaging.util import util
from radioimaging.evaluation import evaluation
from radioimaging.deconvolution import deconvolve
import sep

from deisa.dask import Deisa, get_connection_info
import dask
from dask.distributed import get_worker, Client

#calculates the variance for each pixel of a residual image using a sliding window
def calculate_ppvariance(residual, window_size):
    output = numpy.zeros(residual.shape)

    for i in range(output.shape[0]):
        output[i,:,:] = deconvolve.compute_windowed_var(residual[i,:,:], window_size)

    return output

#uses sep to find a list of sources in some sky image, giving their positions, sizes, orientation, and flux
def find_sources(model, sthresh, max_sources):
    output = numpy.full((model.shape[0], max_sources, 6), -1, dtype=numpy.float32)

    for i in range(model.shape[0]):
        currmodel = model[i,:,:].astype(model.dtype.newbyteorder('='))
        bkg = sep.Background(currmodel)
        data_sub = currmodel - bkg
        objects = sep.extract(data_sub, sthresh, err=bkg.globalrms)

        for j in range(len(objects)):
            if j >= max_sources:
                break

            x = objects['x'][j]
            y = objects['y'][j]
            a = objects['a'][j]
            b = objects['b'][j]
            theta = objects['theta'][j]
            flux = objects['flux'][j]

            output[i,j,:] = [x, y, a, b, theta, flux]

    return output

def write_sources(output_file, objects):
    for i in range(objects.shape[0]):
        source_data = []
        curr_output_file = output_file + str(i)

        for j in range(objects.shape[1]):
            if objects[i,j,0] < 0:
                break

            util.write_to_csv([objects[i,j,0], objects[i,j,1], objects[i,j,2], objects[i,j,3], objects[i,j,4], objects[i,j,5]], curr_output_file)

def main():
    pdicfg_filename = sys.argv[1]
    pdicfg = None
    with open(pdicfg_filename, 'r') as file:
        pdicfg = yaml.safe_load(file)

    mscfg_filename = sys.argv[2]
    mscfg_data = None
    with open(mscfg_filename) as f: 
        mscfg_data = f.read()
    mscfg = json.loads(mscfg_data)

    nmaj = mscfg["nmajcycmsc"] + 1
    npixels = mscfg["npixels"]
    window_size = mscfg["visvar_window"]

    max_sources = mscfg["max_sources"]
    sthresh = mscfg["source_threshold"]


    deisa = Deisa(get_connection_info=lambda: get_connection_info(pdicfg['dask_addr']))

    for i in range(nmaj):
        dresidual, it = deisa.get_array("residual")
        dresidual.rechunk((1, npixels, npixels))
        #recuperates all residuals and writes them into one big fits, could also display/send this information some other way
        resid_variances = dresidual.map_blocks(calculate_ppvariance, window_size=window_size, dtype=dresidual.dtype).compute()

        util.tofits(resid_variances, "results/residual_variances_" + str(i) + ".fits")
        drecon, it = deisa.get_array("reconstruction")
        drecon.rechunk((1, npixels, npixels))
        source_arr = drecon.map_blocks(find_sources, sthresh=sthresh, max_sources=max_sources, dtype=float).compute()
        
        write_sources("results/sources_" + str(i) + "_", source_arr)

    time.sleep(10)

    deisa.close()

if __name__ == "__main__":
    main()