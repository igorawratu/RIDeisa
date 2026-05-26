import sys
import os
import math
import logging
import galsim
import numpy
from scipy.stats import qmc

num_gal = 64
npix = 1024
pix_scale = 1
image_border = 50

rng = numpy.random.Generator(numpy.random.Philox())

fluxes_pow = rng.uniform(low=0, high=2, size=num_gal)
fluxes_base = rng.uniform(low=0, high=1, size=num_gal)

hlrs = rng.uniform(low=2, high=5, size=num_gal)
ns = rng.uniform(low=1, high=3, size=num_gal)
shear_positions = rng.uniform(low=-1, high=1, size=(num_gal, 2))
shear_magnitudes = rng.uniform(low=0, high=0.4, size=num_gal)

sampler = qmc.Sobol(d=2, scramble=True)
pos_minmax = (image_border, npix - image_border)
positions = sampler.random(num_gal) * (pos_minmax[1] - pos_minmax[0]) + pos_minmax[0] 

image = galsim.ImageF(npix, npix, scale=pix_scale)

for i in range(num_gal):
    flux = fluxes_base[i] * 10**fluxes_pow[i]
    gal = galsim.Sersic(n=ns[i], half_light_radius=hlrs[i], flux=flux)
    curr_mag = numpy.sqrt(shear_positions[i,0]**2 + shear_positions[i,1]**2)
    e1 = shear_positions[i,0]/curr_mag*shear_magnitudes[i]
    e2 = shear_positions[i,1]/curr_mag*shear_magnitudes[i]
    gal = gal.shear(g1=e1, g2=e2)

    stamp = gal.drawImage()
    
    stamp.setCenter(positions[i,0], positions[i,1])
    print(stamp.bounds)

    bounds = stamp.bounds & image.bounds
    image[bounds] += stamp[bounds]

file_name = os.path.join('galfield_gt.fits')
image.write(file_name)