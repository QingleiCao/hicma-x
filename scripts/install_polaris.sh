#/bin/bash

module list -l

home=$PWD

echo "update submodules"
#git submodule update --init --recursive

echo "Install starsh"
cd $home/stars-h && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DMPI=OFF -DSTARPU=OFF -DCUDA=OFF -DOPENMP=OFF -DGSL=OFF -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran -DEXAMPLES=OFF
make -j 8 install
echo "export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/env_hicma_parsec.sh
export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Install hcore"
cd $home/hcore
git apply $home/tools/hcore.patch
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran
make -j 8 install
echo "export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/env_hicma_parsec.sh
export export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Install hicma_parsec"
cd  $home && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE="Release" -DDPLASMA_PRECISIONS="s;d" -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DPARSEC_MAX_DEP_OUT_COUNT=16 -DCMAKE_PREFIX_PATH=/home/cql0536/software/lapack/build/installdir/lib64 -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_EXE_LINKER_FLAGS="-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/lib64" -DHICMA_PARSEC_HAVE_CUDA=ON
#cmake .. -DCMAKE_BUILD_TYPE="Release" -DDPLASMA_PRECISIONS="s;d" -DBUILD_SHARED_LIBS=ON -DPARSEC_GPU_WITH_CUDA=ON -DPARSEC_HAVE_DEV_CUDA_SUPPORT=ON -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DPARSEC_GPU_WITH_HIP=OFF -DPARSEC_HAVE_DEV_HIP_SUPPORT=OFF  -DPARSEC_MAX_DEP_OUT_COUNT=16 -DCMAKE_PREFIX_PATH=/home/cql0536/software/lapack/build/installdir/lib64 -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DCMAKE_Fortran_COMPILER=gfortran
#cmake .. -DCMAKE_BUILD_TYPE="Release" -DDPLASMA_PRECISIONS="s;d" -DBUILD_SHARED_LIBS=ON -DPARSEC_GPU_WITH_CUDA=ON -DPARSEC_HAVE_DEV_CUDA_SUPPORT=ON -DBLA_VENDOR=Intel10_64lp_seq -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -G Ninja -DPARSEC_PROF_TRACE=ON -DPARSEC_PROF_GRAPHER=ON
make -j 10
