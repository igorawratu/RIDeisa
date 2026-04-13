#!/bin/bash

#assumes spack with pdi repos are already installed, if not, follow the install instructions at https://github.com/pdidev/spack

#This script sets up everything in a python venv and uses some symlinks to get around things. This is not recommended but we are currently missing some spack packages
#Will be updated once packages become available

rm -r spackenv

spack env create ./spackenv
spack env activate -p ./spackenv/

spack add pdi pdiplugin-decl-hdf5 pdiplugin-decl-netcdf pdiplugin-mpi pdiplugin-pycall pdiplugin-serialize pdiplugin-set-value pdiplugin-trace pdiplugin-user-code
spack add python@3.10

spack concretize
spack install

python -m venv create pythonenv

#assumes python 3.10, needed to have pdi in the venv, update accordingly
SPACK_SITE="/home/sunwang/RIDeisa/spackenv/.spack-env/view/lib/python3.10/site-packages"
echo $SPACK_SITE > pythonenv/lib/python3.10/site-packages/spack-pdi.pth

source pythonenv/bin/activate

rm -r radio-imaging

git clone https://github.com/simon-prunet/radio-imaging
cd radio-imaging
rm -r external_dependencies
mkdir external_dependencies

cd external_dependencies
git clone https://gitlab.com/ska-telescope/external/rascil-main
cd rascil-main
git checkout tags/1.1.0

cp ../../hotfixes/rascil-main/Makefile .
cp ../../hotfixes/rascil-main/requirements.in .
cp ../../hotfixes/rascil-main/simulation_helpers.py rascil/processing_components/simulation/
cp ../../hotfixes/rascil-main/msv2.py rascil/processing_components/visibility/

make requirements
make install_requirements

cd ../../
pip install -r requirements.txt
pip install -e .

pip install deisa

#need to use spack version of mpi
pip uninstall mpi4py
spack add openmpi py-mpi4py
spack concretize
spack install