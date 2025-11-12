#!/bin/bash
gpu=1
#module load gcc/6.4.0
module load gcc/12.2.0 
module load intel/2022.3 
module load cmake/3.28.4/gnu-11.3.1
module load openmpi/4.1.4/gnu11.2.1
module load cuda/12.2  
module load mkl/2022.2.1

#if [ $gpu -eq 1 ]; then
   #module load cuda/9.0
#  module load cuda/11.2.2
  #export PKG_CONFIG_PATH=$HOME/install-mpi/lib/pkgconfig:$PKG_CONFIG_PATH
  #export LD_LIBRARY_PATH=$HOME/install-mpi/lib:$LD_LIBRARY_PATH
  #export PKG_CONFIG_PATH=/ibex/ai/home/omairyrm/l2-dev/hwloc-1.11.13/install/lib/pkgconfig:$PKG_CONFIG_PATH
  export PKG_CONFIG_PATH=/ibex/ai/home/omairyrm/gbgwas/hicma-x-dev/stars-h-dev/build/install/lib/pkgconfig/:$PKG_CONFIG_PATH
  export PKG_CONFIG_PATH=/ibex/ai/home/omairyrm/gbgwas/hicma-x-dev/hcore/build/install/lib/pkgconfig/:$PKG_CONFIG_PATH
  #export PKG_CONFIG_PATH=/ibex/ai/home/omairyrm/l2-dev/modified-chameleon/chameleon/build-nompi/install/lib/pkgconfig:$PKG_CONFIG_PATH
  #export PKG_CONFIG_PATH=$HOME/magma-2.5.2/install/lib/pkgconfig:$PKG_CONFIG_PATH
  #export LD_LIBRARY_PATH=$HOME/magma-2.5.2/install/lib:$LD_LIBRARY_PATH
#else
#  export PKG_CONFIG_PATH=$HOME/install-mpi-nogpu/lib/pkgconfig:$PKG_CONFIG_PATH
#  export LD_LIBRARY_PATH=$HOME/install-mpi-nogpu/lib:$LD_LIBRARY_PATH
#fi
