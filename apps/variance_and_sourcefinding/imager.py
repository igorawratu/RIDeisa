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
from radioimaging.visibility import residual, weights
from radioimaging.images import images, filters
from casacore.tables import table

pdicfg_filename = sys.argv[1]
mscfg_filename = sys.argv[2]

def recon(comm):
    partition = comm.Get_rank()
    nnodes = comm.Get_size()

    #read pdi config
    with open(pdicfg_filename, 'r') as pdicfgfile:
        try:    
            pdicfg = yaml.safe_load(pdicfgfile)
        except yaml.YAMLError as exc:
            exit(exc)

    pdi.init(yaml.dump(pdicfg['pdi']))

    #read reconstruction config
    with open(mscfg_filename) as f: 
        mscfg_data = f.read()

    mscfg = json.loads(mscfg_data)
    
    #dataset ingestion
    all_datasets = mscfg["datasets"]
    ms_name = all_datasets[partition]
    channel_start = int(mscfg["channel_start"])
    channel_end = int(mscfg["channel_end"])
    data_descriptors = range(int(mscfg["data_descriptor_start"]), int(mscfg["data_descriptor_end"]) + 1)
    bda = mscfg["bda"] if mscfg["bda"] is not None else False

    #imaging
    npixels = mscfg["npixels"]
    cellsize = mscfg["cellsize"]
    weighting = mscfg["weighting"]
    robustness = mscfg["robustness"]
    ells = mscfg["ells"]
    ells.sort()
    delta = mscfg["delta"]

    #deconvolution
    msc_niter = int(mscfg["msclean_iter"])
    thresh = mscfg["clean_thresh"]
    scales = mscfg["fullres_clean_scales"]
    nmaj = mscfg["nmajcycmsc"] + 1
    fracthresh = mscfg["clean_fracthresh"]
    first_mc_scales = mscfg["first_mc_clean_scales"][partition]
    first_mc_thresholds = mscfg.get("fracthreshs", None)
    fmct = fracthresh if first_mc_thresholds is None else first_mc_thresholds[partition]
    sens = None
    gain = 0.1

    #output
    output_dir = mscfg["output_dir"] + "_pmsclean/"

    #convert strings needed to byte arrays for pdi
    dask_addr = pdicfg['dask_addr']
    bdask_addr = bytearray()
    bdask_addr.extend(map(ord, dask_addr))

    boutput_dir = bytearray()
    boutput_dir.extend(map(ord, output_dir))

    pdi.expose("img_size", npixels, pdi.OUT)
    pdi.expose("dask_address", bdask_addr, pdi.OUT)
    pdi.expose("output_dir", boutput_dir, pdi.OUT)
    pdi.expose("rank", partition, pdi.OUT)
    pdi.expose("nodes", nnodes, pdi.OUT)

    pdi.event("precompute")

    weight_grid, weight_timings, num_vis = weights.compute_weights_griddata_from_ms(ms_name, channel_start, channel_end, data_descriptors, npixels, cellsize, bda=bda)
    psf, estimate, psf_timings, weight = residual.compute_psf_by_channel(ms_name, channel_start, channel_end, data_descriptors, npixels, cellsize, weighting, robustness=robustness, weight_grid=weight_grid, bda=bda)
    np_psf = psf["pixels"].data[0, 0, :, :]

    #create filters
    sigma2s = [1] * (len(ells) + 1)
    frs, _, _ = filters.create_filters_mstep(np_psf.shape[-1] // 2, [delta]*len(ells), ells, sigma2s)
    frns = [numpy.array(fr) for fr in frs]
    fs2ds = [filters.freq1d_to_radial2d(f1d, np_psf.shape[-1])[1] for f1d in frns]
    wgts = numpy.zeros(len(sigma2s))

    #create full psf
    psfs = comm.allgather(np_psf)
    wgts = comm.allgather(weight)
    total_weight = sum(wgts)

    wgts = [weight / total_weight for weight in wgts]

    joint_psf = util.convolve2d(psfs[0] * wgts[0], fs2ds[0])
    for i, currpsf in enumerate(psfs[1:]):
        joint_psf += util.convolve2d(currpsf * wgts[i+1], fs2ds[i+1])

    joint_psf = numpy.ascontiguousarray(joint_psf, dtype=numpy.float64)

    barrier_start = time.time()
    comm.Barrier()
    barrier_end = time.time()

    for i in range(nmaj):
        gc.collect()

        resid, resid_timings = residual.compute_residual_from_ms(estimate, ms_name, channel_start, channel_end, data_descriptors, npixels, cellsize, weighting, robustness=robustness, weight_grid=weight_grid)

        prev_estimates = None
        if i > 0:
            prev_estimates = comm.allgather(estimate.pixels.data[0,0,:,:])

            full_residual = util.convolve2d(wgts[partition] * resid["pixels"].data[0, 0, :, :], fs2ds[partition])
            for j, sigma2 in enumerate(sigma2s):
                if j != partition:
                    constraint = prev_estimates[j] - prev_estimates[partition]
                    constraint = util.convolve2d(constraint, wgts[j] * psfs[j])
                    constraint = util.convolve2d(constraint, fs2ds[j])

                    full_residual += constraint

            curr_psf = joint_psf
        else:
            full_residual = resid.pixels.data[0,0,:,:]
            curr_psf = psf["pixels"].data[0, 0, :, :]

        gc.collect()

        deconvolved, _ = msclean(full_residual, joint_psf, None, sens, gain, thresh, msc_niter, first_mc_scales if i == 0 else scales, fmct if i == 0 else fracthresh)
        estimate = images.add_to_image(estimate, deconvolved)

        np_resid = resid["pixels"].data[0, 0, :, :]
        np_estimate = estimate["pixels"].data[0, 0, :, :]
        
        pdi.multi_expose("majcyc", 
            [("iteration", i, pdi.OUT),
            ("tmp_resid", np_resid, pdi.OUT),
            ("tmp_recon", np_estimate, pdi.OUT)])

    time.sleep(5)

    pdi.finalize()

comm = MPI.COMM_WORLD

recon(comm)