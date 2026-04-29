import dask.array as da
import numpy
import yaml
import os
import json
import sys
import time
import copy
import gc

from radioimaging.util import util

from astropy.coordinates import SkyCoord
import astropy.units as u

import dask
from deisa.dask import Deisa, get_connection_info
from dask.distributed import get_worker, Client

import ducc0.wgridder as ng 

from ska_sdp_datamodels.science_data_model.polarisation_model import PolarisationFrame
from ska_sdp_datamodels.science_data_model.polarisation_functions import convert_pol_frame
from ska_sdp_func_python.util.coordinate_support import skycoord_to_lmn

import matplotlib.pyplot as plt

def plot_stats(jackknifed_images, residuals, filename):
    _, n, nx, ny = jackknifed_images.shape
    jackknife1d = jackknifed_images.reshape((n*nx*ny))
    x_one = numpy.arange(nx*ny)
    x = numpy.tile(x_one, n)

    plt.scatter(x, jackknife1d, s=0.01, marker=".", alpha=1, label="jknife")

    residuals = residuals.reshape(residuals.shape[0], nx*ny)
    for i in range(residuals.shape[0]):
        plt.scatter(x_one, residuals[i], s=0.01, marker="D", label="recon " + str(i), alpha=0.2)

    plt.xlabel("x + y * width")
    plt.ylabel("Jy/beam")

    plt.legend(markerscale=50)
    plt.savefig(filename, bbox_inches='tight', dpi=600)
    plt.clf()

#Modified and simplified version of the RASCIL invert_ng function. This is done because there is certain functionality not necessary for this demo and to avoid reconstructing
#the RASCIL visibility structure, which requires much more data than necessary than for a simple invert. Also assumes only stokes I, no phase shifting, and that everything gets gridded to 1 channel
def invert(viso, mul, uvw_coords, freq, flags, weights, npixdirty, pixsize):
    nthreads = 4
    epsilon = 1e-12
    do_wstacking = True

    vis = viso
    nrows, nbaselines, vnchan, vnpol = vis.shape

    ms = vis * (1 - flags) * mul
    ms = ms.reshape([nrows * nbaselines, vnchan, vnpol])

    opf = tpf = PolarisationFrame("stokesI")

    ms = convert_pol_frame(ms, opf, tpf, polaxis=2).astype("c16")
    fuvw = copy.deepcopy(uvw_coords)
    fuvw = fuvw.reshape([nrows * nbaselines, 3])
    # We need to flip the u and w axes.
    fuvw[:, 0] *= -1.0
    fuvw[:, 2] *= -1.0

    wgt = (weights * (1 - flags)).astype("f8")
    wgt = wgt.reshape([nrows * nbaselines, vnchan, vnpol])

    img = numpy.zeros((npixdirty, npixdirty))
    sumwt = 0

    # Nifty gridder likes to receive contiguous arrays
    # so we transpose at the beginning
    mst = ms.T
    wgtt = wgt.T

    lms = numpy.ascontiguousarray(mst[0, :, :].T)
    if numpy.max(numpy.abs(lms)) > 0.0:
        lwt = numpy.ascontiguousarray(wgtt[0, :, :].T)
        dirty = ng.ms2dirty(
            fuvw,
            freq,
            lms,
            lwt,
            npixdirty,
            npixdirty,
            pixsize,
            pixsize,
            0,
            0,
            epsilon,
            do_wstacking,
            nthreads=nthreads,
            double_precision_accumulation=True,
        )
        img = dirty.T

    return img

#gets a reference residual using jackknife resampling
def jackknife_vis(vis, uvw_coords, flags, weights, freqs, npix, pixsize_rad, num, sumwt):
    vis = vis.view(numpy.complex128)

    jackknifes = numpy.zeros((1, 1, num, npix, npix))

    for i in range(num):
        mul = numpy.random.choice([-1, 1], size=vis.shape)
        jackknifes[0,0,i,:,:] = invert(vis[0], mul[0], uvw_coords[0,0,...], freqs[0,0,0,0,:], flags[0], weights[0], npix, pixsize_rad) / sumwt
        gc.collect()

    return jackknifes

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

    nmaj = mscfg["nmajcycl1"] + 1
    npix = mscfg["npixels"]
    pixsize_rad = mscfg["cellsize"]
    num_jackknifes = mscfg["num_jackknifes"]

    deisa = Deisa(get_connection_info=lambda: get_connection_info(pdicfg['dask_addr']))

    sumwt = deisa.get_array("sumwts")[0].sum().compute()

    vis, _ = deisa.get_array("vis")
    uvw_coords, _ = deisa.get_array("uvw_coords")
    flags, _ = deisa.get_array("flags")
    weights, _ = deisa.get_array("weights")
    freqs, _ = deisa.get_array("freqs")

    jackknifes_per_channel = dask.array.map_blocks(jackknife_vis, vis, uvw_coords, flags, weights, freqs, npix, pixsize_rad, num_jackknifes, sumwt, dtype=numpy.float64)
    jackknifes = jackknifes_per_channel.sum(axis=0)

    jackknifes_persisted = deisa.client.persist(jackknifes)

    residual_futures = []
    for i in range(nmaj):
        dresidual, _ = deisa.get_array("residual")
        residual_futures.append(deisa.client.compute(dresidual.sum(axis=0)))

    residuals = numpy.zeros((len(residual_futures), npix, npix))
    for i, future in enumerate(residual_futures):
        residuals[i,:,:] = future.result().reshape((npix, npix))

    jackknifed_images = deisa.client.compute(jackknifes_persisted).result()

    util.tofits(jackknifed_images, "jackknifes.fits")
    util.tofits(residuals, "residuals.fits")

    deisa.close()

    for i in range(residuals.shape[0]):
        curr_resid = residuals[i].reshape(1, residuals.shape[1], residuals.shape[2])
        plot_stats(jackknifed_images, curr_resid, "results/plot_mc" + str(i) + ".png")

    plot_stats(jackknifed_images, residuals, "results/plot_mcall.png")

if __name__ == "__main__":
    main()