#!/bin/bash
gpu=1
module load gcc/8.2.0
module load intel/2020
module load cuda/11.5.2
module load cmake/3.19.2/gnu-6.4.0
module load openmpi/4.0.3-intel2020-cuda11.0
module load gsl/2.6/gnu-6.4.0

if [ $gpu -eq 1 ]; then
module load cuda/11.5.2
fi

#export PKG_CONFIG_PATH=/ibex/ai/home/omairyrm/gwas-parsec-dev/stars-h/build/install/lib/pkgconfig/:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/ibex/ai/home/omairyrm/hicma-mp-cholesky/hicma-x-dev/stars-h/build/installdir/lib/pkgconfig/:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/ibex/ai/home/omairyrm/hicma-mp-cholesky/hicma-x-dev/hcore/build/install/lib/pkgconfig/:$PKG_CONFIG_PATH
