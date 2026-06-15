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

#matches reconstructed sources to ground truth sources so that comparisons can be done wrt flux density, position, etc. Matches any sources that fall within some position threshold 
#by greedily selecting the closest source (so can't have 1 gt source match to 2 reconstructed sources)
#Technically shape, size, orientation etc. can also be used but omitting it for now
#Returns the matches (recon id, gt id), but also false sources (reconstructed source with no matching ground truth source) and missed sources (ground truth source with no matching reconstructed source)
def match_sources(recon_sources, gt_sources, match_threshold=2):
    matches = []
    matched_gt = set()

    for i in range(recon_sources.shape[1]):
        if recon_sources[0,i,0] < 0:
            break

        matching_id = -1
        closest_distance = 9999999

        for j, gt_source in enumerate(gt_sources):
            if j in matched_gt:
                continue

            x = recon_sources[0,i,0] - gt_source[0]
            y = recon_sources[0,i,1] - gt_source[1]
            d = (x**2 + y**2)**0.5
            if d <= match_threshold and d < closest_distance:
                closest_distance = d
                matching_id = j

        if matching_id >= 0:
            matched_gt.add(matching_id)
            matches.append((i, matching_id))

    output = numpy.zeros((1, len(matches), 2), dtype=numpy.int32)
    for i, match in enumerate(matches):
        output[0,i,0] = match[0]
        output[0,i,1] = match[1]

    return output

#functions for different metrics of matching sources

#integrated flux ratio
def flux_ratio(matches, recon_sources, gt_sources):
    flux_ratios = numpy.zeros((1, matches.shape[1]), dtype=numpy.float32)

    for i in range(matches.shape[1]):
        flux_ratios[0,i] = recon_sources[0,matches[0,i,0],5] / gt_sources[matches[0,i,1]][5]

    return flux_ratios

#peak flux ratio
def pflux_ratio(matches, recon_sources, gt_sources):
    pflux_ratios = numpy.zeros((1, matches.shape[1]), dtype=numpy.float32)

    for i in range(matches.shape[1]):
        pflux_ratios[0,i] = recon_sources[0, matches[0,i,0], 6] / gt_sources[matches[0,i,1]][6]

    return pflux_ratios

#positional offsets
def pos_offset_y(matches, recon_sources, gt_sources):
    yoffsets = numpy.zeros((1, matches.shape[1]), dtype=numpy.float32)

    for i in range(matches.shape[1]):
        yoffsets[0,i] = recon_sources[0, matches[0,i,0], 1] - gt_sources[matches[0,i,1]][1]

    return yoffsets

def pos_offset_x(matches, recon_sources, gt_sources):
    xoffsets = numpy.zeros((1, matches.shape[1]), dtype=numpy.float32)

    for i in range(matches.shape[1]):
        xoffsets[0,i] = recon_sources[0, matches[0,i,0], 0] - gt_sources[matches[0,i,1]][0]

    return xoffsets


#distance offsets
def dist_offset(matches, recon_sources, gt_sources):
    distances = numpy.zeros((1, matches.shape[1]), dtype=numpy.float32)

    for i in range(matches.shape[1]):
        x = recon_sources[0, matches[0,i,0], 0] - gt_sources[matches[0,i,1]][0]
        y = recon_sources[0, matches[0,i,0], 1] - gt_sources[matches[0,i,1]][1]
        
        distances[0,i] = (x**2 + y**2)**0.5

    return distances

#peak to integrated ratios
def peak_integrated_flux_ratios(matches, recon_sources, gt_sources):
    peak_integrated_flux_ratios = numpy.zeros((1, matches.shape[1]), dtype=numpy.float32)

    for i in range(matches.shape[1]):
        recon = recon_sources[0, matches[0,i,0], 6] / recon_sources[0, matches[0,i,0], 5]
        gt = gt_sources[matches[0,i,1]][6] / gt_sources[matches[0,i,1]][5]
        
        peak_integrated_flux_ratios[0,i] = recon/gt

    return peak_integrated_flux_ratios

def plot_comparison_metrics(matches, recon_sources, gt_sources, match_stats, y_axis, title, xax_title, yax_title, output_filename):
    ymin = 9999999
    ymax = -9999999

    yvals = []

    for i in range(matches.shape[1]):
        ymax = max(ymax, gt_sources[matches[0,i,1]][y_axis])
        ymin = min(ymin, gt_sources[matches[0,i,1]][y_axis])

        yvals.append(gt_sources[matches[0,i,1]][y_axis])

    padding = (ymax - ymin) * 0.1

    plt.title(title)
    plt.xlabel(xax_title)
    plt.ylabel(yax_title)
    plt.scatter(match_stats[0,:], yvals)
    plt.ylim(ymin - padding, ymax + padding)

    plt.savefig(output_filename + ".png", bbox_inches='tight', dpi=600)
    plt.clf()

def find_sources(model, sthresh, max_sources, consolidate=True, consolidate_thresh=(1, 2.5)):
    output = numpy.full((1, max_sources, 8), -1, dtype=numpy.float32)

    currmodel = model[0,...].astype(model.dtype.newbyteorder('='))
    bkg = sep.Background(currmodel)
    data_sub = currmodel - bkg
    objects = sep.extract(data_sub, sthresh, err=bkg.globalrms)

    num_sources = 0

    for j in range(len(objects)):
        if num_sources >= max_sources:
            break

        x = objects['x'][j]
        y = objects['y'][j]
        a = objects['a'][j]
        b = objects['b'][j]
        theta = objects['theta'][j]
        flux = objects['flux'][j]
        pflux = objects['peak'][j]
        cdist = numpy.sqrt(x**2 + y**2)

        deg = theta * 180. / numpy.pi

        matching_id = -1
        for k in range(num_sources):
            if abs(x - output[0,k,0]) <= consolidate_thresh[0] and \
                abs(y - output[0,k,1]) <= consolidate_thresh[0] and \
                abs(a - output[0,k,2]) <= consolidate_thresh[0] and \
                abs(b - output[0,k,3]) <= consolidate_thresh[0] and \
                abs(deg - output[0,k,4]) <= consolidate_thresh[1]:
                    matching_id = k
                    break 

        if matching_id < 0:
            output[0,num_sources,:] = [x, y, a, b, deg, flux, pflux, cdist]
            num_sources += 1
        else:
            output[0,matching_id,5] += flux
            output[0,matching_id,6] += pflux

    return output

def plot_sources(sources, gt_sources, matches, model, filename):
    recon_matches = set()
    gt_matches = set()

    for i in range(matches.shape[1]):
        recon_matches.add(matches[0,i,0])
        gt_matches.add(matches[0,i,1])

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

            e = Ellipse(xy=(sources[i,j,0], sources[i,j,1]),
                        width=6*sources[i,j,2],
                        height=6*sources[i,j,3],
                        angle=sources[i,j,4],
                        linewidth=0.5)
            e.set_facecolor('none')

            if j in recon_matches:
                e.set_edgecolor('green')
            else:
                e.set_edgecolor('red')
            ax.add_artist(e)

        for j, gt_source in enumerate(gt_sources):
            e = Ellipse(xy=(gt_source[0], gt_source[1]),
                        width=6*gt_source[2],
                        height=6*gt_source[3],
                        angle=gt_source[4],
                        linewidth=0.5)
            e.set_facecolor('none')

            if j in gt_matches:
                e.set_edgecolor('cyan')
            else:
                e.set_edgecolor('orange')
            ax.add_artist(e)

        plt.savefig(filename + ".png", bbox_inches='tight', dpi=600)
        plt.clf()

def write_sources(sources, output_file):
    for i in range(sources.shape[0]):
        source_data = []
        curr_output_file = output_file

        for j in range(sources.shape[1]):
            if sources[i,j,0] < 0:
                break

            util.write_to_csv([sources[i,j,0], sources[i,j,1], sources[i,j,2], sources[i,j,3], sources[i,j,4], sources[i,j,5], sources[i,j,6], sources[i,j,7]], curr_output_file)

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

    gt_sources_fname = mscfg["gt_sources_fname"]
    match_thresh = mscfg["match_threshold"]

    gt_sources = util.read_csv(gt_sources_fname, separate_rows=True)[1:]
    gt_sources = [[float(x) for x in row] for row in gt_sources]

    deisa_client = Deisa(get_connection_info=lambda: get_connection_info(pdicfg['dask_addr']))
    source_futures = []
    models = []

    for i in range(nmaj+1):
        drecon, it = deisa_client.get_array("reconstruction")
        dresid, it = deisa_client.get_array("residual")

        #we only have a blank reconstruction if i is 0, just ignore it
        if i > 0:
            recon_persisted = drecon.persist()
            drecon_sources = drecon.map_blocks(find_sources, sthresh, max_sources, dtype=numpy.float32)
            dmatches = drecon_sources.map_blocks(match_sources, gt_sources, match_thresh, dtype=numpy.int32)
            dflux_ratios = dask.array.map_blocks(flux_ratio, dmatches, drecon_sources, gt_sources, dtype=numpy.float32)
            dpflux_ratios = dask.array.map_blocks(pflux_ratio, dmatches, drecon_sources, gt_sources, dtype=numpy.float32)
            dpoffset_ys = dask.array.map_blocks(pos_offset_y, dmatches, drecon_sources, gt_sources, dtype=numpy.float32)
            dpoffset_xs = dask.array.map_blocks(pos_offset_x, dmatches, drecon_sources, gt_sources, dtype=numpy.float32)
            dpoffset_dists = dask.array.map_blocks(dist_offset, dmatches, drecon_sources, gt_sources, dtype=numpy.float32)
            dpiflux_ratios = dask.array.map_blocks(peak_integrated_flux_ratios, dmatches, drecon_sources, gt_sources, dtype=numpy.float32)

            model = recon_persisted.compute()
            recon_sources, matches, flux_ratios, pflux_ratios, poffset_ys, poffset_xs, poffset_dists, piflux_ratios = \
                dask.compute(drecon_sources, dmatches, dflux_ratios, dpflux_ratios, dpoffset_ys, dpoffset_xs, dpoffset_dists, dpiflux_ratios)

            write_sources(recon_sources, "results/recon_sources_" + str(i))
            plot_sources(recon_sources, gt_sources, matches, model, "results/recon_sources_" + str(i))
            plot_comparison_metrics(matches, recon_sources, gt_sources, flux_ratios, 5, "Flux Ratio", "flux_recon/flux_gt", "Jy_gt", "results/fluxratio_" + str(i))
            plot_comparison_metrics(matches, recon_sources, gt_sources, pflux_ratios, 5, "PFlux Ratio", "pflux_recon/pflux_gt", "Jy_gt", "results/pfluxratio_" + str(i))
            plot_comparison_metrics(matches, recon_sources, gt_sources, poffset_dists, 5, "Distance Offset", "|recon_xy - gt_xy|_2", "Jy_gt", "results/distance_offset_" + str(i))
            plot_comparison_metrics(matches, recon_sources, gt_sources, piflux_ratios, 5, "Peak to integrated flux", "piflux_ratio_recon/piflux_ratio_gt", "Jy_gt", "results/piflux_ratio_flux_" + str(i))
            plot_comparison_metrics(matches, recon_sources, gt_sources, piflux_ratios, 7, "Peak to integrated flux", "piflux_ratio_recon/piflux_ratio_gt", "dist_to_phasecenter", "results/piflux_ratio__dist_" + str(i))

            plt.title("X Y offsets")
            plt.scatter(poffset_xs[0], poffset_ys[0])
            plt.savefig("results/xyoffset_" + str(i) + ".png", bbox_inches='tight', dpi=600)
            plt.clf()

    deisa_client.close()

if __name__ == "__main__":
    main()