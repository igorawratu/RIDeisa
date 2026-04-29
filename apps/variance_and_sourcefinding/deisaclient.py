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

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

#calculates the variance for each pixel of a residual image using a sliding window
def calculate_ppvariance(residual, window_size):
    output = numpy.zeros(residual.shape)

    for i in range(output.shape[0]):
        output[i,:,:] = deconvolve.compute_windowed_var(residual[i,:,:], window_size)

    return output

#uses sep to find a list of sources in some sky image, giving their positions, sizes, orientation, and flux
def find_sources(model, sthresh, max_sources):
    output = numpy.full((1, max_sources, 6), -1, dtype=numpy.float32)

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

def write_sources(sources, output_file):
    for i in range(sources.shape[0]):
        source_data = []
        curr_output_file = output_file

        for j in range(sources.shape[1]):
            if sources[i,j,0] < 0:
                break

            util.write_to_csv([sources[i,j,0], sources[i,j,1], sources[i,j,2], sources[i,j,3], sources[i,j,4], sources[i,j,5]], curr_output_file)

def plot_sources(sources, model, filename):
    max_flux = -1
    for i in range(sources.shape[0]):
        for j in range(sources.shape[1]):
            max_flux = max(sources[i,j,5], max_flux)

    for i in range(sources.shape[0]):
        fig, ax = plt.subplots()
        m, s = numpy.mean(model[i]), numpy.std(model[i])
        im = ax.imshow(model[i], interpolation='nearest', cmap='turbo', origin='lower')

        for j in range(sources.shape[1]):
            if sources[i,j,0] < 0:
                break

            if sources[i,j,5] < max_flux*1e-2:
                continue

            e = Ellipse(xy=(sources[i,j,0], sources[i,j,1]),
                        width=6*sources[i,j,2],
                        height=6*sources[i,j,3],
                        angle=sources[i,j,4] * 180. / numpy.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            ax.add_artist(e)

        plt.savefig(filename + ".png", bbox_inches='tight', dpi=600)
        plt.clf()

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

    nmaj = mscfg["nmajcycl1"]
    npixels = mscfg["npixels"]
    window_size = mscfg["visvar_window"]

    max_sources = mscfg["max_sources"]
    sthresh = mscfg["source_threshold"]


    deisa = Deisa(get_connection_info=lambda: get_connection_info(pdicfg['dask_addr']))
    variance_futures = []
    source_futures = []
    models = []

    for i in range(nmaj + 1):
        dresidual, it = deisa.get_array("residual")
        variance_futures.append(deisa.client.compute(dresidual.map_blocks(calculate_ppvariance, window_size=3, dtype=dresidual.dtype)))

        drecon, it = deisa.get_array("reconstruction")

        #we only have a blank reconstruction if i is 0
        if i > 0:
            recon_persisted = drecon.persist()
            source_futures.append(deisa.client.compute(drecon.map_blocks(find_sources, sthresh, max_sources, dtype=float)))
            models.append(recon_persisted.compute())

    for i, source_future in enumerate(source_futures):
        sources = source_future.result()
        write_sources(sources, "results/sources_" + str(i+1))
        plot_sources(sources, models[i], "results/sources_" + str(i+1))

    for i, vf in enumerate(variance_futures):
        variances = vf.result()
        util.tofits(variances, "results/var_" + str(i) + ".fits")

    deisa.close()

if __name__ == "__main__":
    main()