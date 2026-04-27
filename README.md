# RIDeisa
Prototype on using the in situ tool Deisa with a radio-interferometric imaging pipeline

Contains a bunch of example applications on using Deisa together with a radio-interferometric pipeline. These are currently:
1. A multi-node simulation using the SMURFIT framework (paper under review) using deisa to both estimate variance and source find on reconstructed images
2. A multi-node simulation parallelizing de/gridding by EM frequency with deconvolution done on a separate node in serial. This uses deisa to perform jackknife resampling in order to reconstruct a population of reference residuals. It then plots the residuals of the reconstructions of different major cycles against this in a scatter plot.

Datasets used currently are some test ones. The repository will be updated with a download script for some nicer ones once I get around to simulating and uploading them.