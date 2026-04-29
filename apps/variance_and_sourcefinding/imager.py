#!/usr/bin/env python

"""\
Simple msclean imager to test pdi
""" 

__author__ = "Sunrise Wang"
__email__ = "sunrise.wang@oca.eu, sunrisewng@gmail.com"

import pdi, yaml

import json
import numpy
import time
import sys
import gc

from pathlib import Path
from mpi4py import MPI

from ska_sdp_func_python.image.cleaners import msclean
from radioimaging.util import util
from radioimaging.visibility import residual, weights, ingest
from radioimaging.images import images, filters
from radioimaging.deconvolution import deconvolve
from casacore.tables import table

import astropy.units as u

pdicfg_filename = sys.argv[1]
mscfg_filename = sys.argv[2]

def deconv_node(comm, mscfg, pdicfg):
    pdi.init(yaml.dump(pdicfg['pdi']))

    dummwt = 0
    comm.allgather(dummwt)

    fista_iter = int(mscfg["nfistaiter"])
    init_lambda_full = mscfg["init_lambda_full"]
    nmaj = int(mscfg["nmajcycl1"])
    wavelet_dict = mscfg["wavelet_dict"]
    lambda_growth_steepness = mscfg["lambda_growth_steepness"]
    npixels = mscfg["npixels"]

    #convert strings needed to byte arrays for pdi
    dask_addr = pdicfg['dask_addr']
    bdask_addr = bytearray()
    bdask_addr.extend(map(ord, dask_addr))

    pdi.expose("img_size", npixels, pdi.OUT)
    pdi.expose("dask_address", bdask_addr, pdi.OUT)

    pdi.event("precompute")

    psfs = comm.gather(None, root=0)

    full_psf = None
    for curr_psf in psfs[1:]:
        if full_psf is None:
            full_psf = curr_psf
        else:
            full_psf += curr_psf

    recon = numpy.zeros((1, 1, full_psf.shape[0], full_psf.shape[1]))

    for i in range(nmaj):
        residuals = comm.gather(None, root=0)

        full_resid = None
        for curr_resid in residuals[1:]:
            if full_resid is None:
                full_resid = curr_resid
            else:
                full_resid += curr_resid

        pdi.multi_expose("majcyc", [("iteration", i, pdi.OUT),
            ("tmp_recon", recon[0, 0, :, :], pdi.OUT),
            ("tmp_resid", full_resid, pdi.OUT)])

        t = float(i) / (float(nmaj))
        curr_lambda_mul = init_lambda_full + (1 - init_lambda_full) * ((numpy.exp(lambda_growth_steepness * t) - 1) / (numpy.exp(lambda_growth_steepness) - 1))

        deconvolved = deconvolve.deconvolve_multipartition_single(full_resid, full_psf, fista_iter, curr_lambda_mul)

        recon[0, 0, :, :] += deconvolved

        comm.Bcast(recon, root=0)

    residuals = comm.gather(None, root=0)

    full_resid = None
    for curr_resid in residuals[1:]:
        if full_resid is None:
            full_resid = curr_resid
        else:
            full_resid += curr_resid

    pdi.multi_expose("majcyc", [("iteration", nmaj, pdi.OUT),
        ("tmp_recon", recon[0, 0, :, :], pdi.OUT),
        ("tmp_resid", full_resid, pdi.OUT)])

    #to wait for analytics to finish, need to find a better way to do this
    time.sleep(60)

    pdi.finalize()

def grid_node(comm, mscfg, pdicfg):
    #-1 because technically deconvolution is the 0th node but is not taken into account for pdi as this example only sends residuals and visibilities
    partition = comm.Get_rank() - 1
    nnodes = comm.Get_size() - 1
    
    #dataset ingestion
    all_datasets = mscfg["datasets"]
    assert(len(all_datasets) >= (nnodes))
    ms_name = all_datasets[partition]
    channel_start = int(mscfg["channel_start"])
    channel_end = int(mscfg["channel_end"])
    data_descriptors = range(int(mscfg["data_descriptor_start"]), int(mscfg["data_descriptor_end"]) + 1)

    #imaging
    cellsize = mscfg["cellsize"]
    weighting = mscfg["weighting"]
    robustness = mscfg["robustness"]
    nmaj = int(mscfg["nmajcycl1"])
    npixels = mscfg["npixels"]

    [vis], _ = ingest.create_visibility_from_ms(ms_name, start_chan = channel_start, end_chan = channel_end, selected_dds = data_descriptors, use_weight_spec=True, flatten=False)
    ntime, nbaseline, nfreq, npol = vis["vis"].data.shape

    vis = weights.compute_weights(vis, npixels, cellsize, weighting, robustness=robustness)

    sumwt = numpy.sum(vis.visibility_acc.flagged_imaging_weight.data)

    psf, model, sumwt = residual.compute_psf(vis, npixels, cellsize, include_weight_and_model=True)

    resid = residual.compute_residual(model, vis, npixels, cellsize)

    allwts = comm.allgather(sumwt)[1:]
    total_wt = numpy.sum(allwts)
    corrected_wt = sumwt.item() / total_wt.item()

    nppsf = psf.pixels.data[0, 0, :, :] * corrected_wt
    npresid = resid.pixels.data[0, 0, :, :] * corrected_wt

    iteration = 0

    comm.gather(nppsf, root=0)
    comm.gather(npresid, root=0)

    for i in range(nmaj):
        comm.Bcast(model.pixels.data, root=0)
        resid = residual.compute_residual(model, vis, npixels, cellsize)
        gc.collect()

        iteration = i+1
        npresid = resid.pixels.data[0, 0, :, :]  * corrected_wt

        comm.gather(npresid, root=0)

comm = MPI.COMM_WORLD

#read configs
with open(pdicfg_filename, 'r') as pdicfgfile:
    try:    
        pdicfg = yaml.safe_load(pdicfgfile)
    except yaml.YAMLError as exc:
        exit(exc)

with open(mscfg_filename) as f: 
    mscfg_data = f.read()

mscfg = json.loads(mscfg_data)

rank = comm.Get_rank()

#run nodes, 0 for deconv, the rest for de/gridding
if rank == 0:
    deconv_node(comm, mscfg, pdicfg)
else:
    grid_node(comm, mscfg, pdicfg)