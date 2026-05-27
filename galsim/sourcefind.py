import sep
import numpy
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from radioimaging.util import util


#uses SEP to find sources, and optionally consolidate them, as the same source can be found multiple times. The threshold in the case is in pixels and degrees
def find_sources(model, sthresh, max_sources, consolidate=True, consolidate_thresh=(1, 2.5)):
    output = numpy.full((max_sources, 6), -1, dtype=numpy.float32)

    currmodel = model.astype(model.dtype.newbyteorder('='))
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
        deg = theta * 180. / numpy.pi

        matching_id = -1
        for k in range(num_sources):
            if abs(x - output[k,0]) <= consolidate_thresh[0] and \
                abs(y - output[k,1]) <= consolidate_thresh[0] and \
                abs(a - output[k,2]) <= consolidate_thresh[0] and \
                abs(b - output[k,3]) <= consolidate_thresh[0] and \
                abs(deg - output[k,4]) <= consolidate_thresh[2]:
                    matching_id = k
                    break 

        if matching_id < 0:
            output[num_sources,:] = [x, y, a, b, deg, flux]
            num_sources += 1
        else:
            output[i, matching_id,5] += flux

    return output, num_sources


def plot_sources(sources, model, filename):
    max_flux = -1
    for j in range(sources.shape[0]):
        max_flux = max(sources[j,5], max_flux)

    fig, ax = plt.subplots()
    m, s = numpy.mean(model), numpy.std(model)
    im = ax.imshow(model, interpolation='nearest', cmap='turbo', origin='lower')

    for j in range(sources.shape[0]):
        if sources[j,0] < 0:
            break

        e = Ellipse(xy=(sources[j,0], sources[j,1]),
                    width=6*sources[j,2],
                    height=6*sources[j,3],
                    angle=sources[j,4],
                    linewidth=0.5)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)

    plt.savefig(filename + ".png", bbox_inches='tight', dpi=600)
    plt.clf()

def write_sources(sources, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

    util.write_to_csv(["x", "y", "a", "b", "deg", "flux"], output_file)

    for j in range(sources.shape[0]):
        if sources[j,0] < 0:
            break

        util.write_to_csv([sources[j,0], sources[j,1], sources[j,2], sources[j,3], sources[j,4], sources[j,5]], output_file)


model = util.fromfits("galfield_gt.fits")
sources, num_sources = find_sources(model, 10, 1000)
plot_sources(sources, model, "sources")

write_sources(sources, "sources.csv")