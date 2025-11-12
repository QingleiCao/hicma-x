#/bin/bash

# env
export OMP_NUM_THREADS=1
#alias cdshared="cd /lustre/orion/csc312/proj-shared/lei"
#alias cdproj="cd /ccs/proj/csc312/lei"

module purge

module load PrgEnv-amd
module load craype-accel-amd-gfx90a
module load rocm
module load Core
module load ninja
module load cmake
module load gsl
module load hwloc/2.5.0
module load perftools-base
module load perftools
module load valgrind4hpc
#module load papi
module load gdb4hpc
module load openblas/0.3.17-pthreads

export HIP_DIR=$HIP_PATH
export PATH=$PATH:$HIP_PATH/bin:/ccs/home/cql0536/frontier/opt/bin

ulimit -s 8388608

module list

#cd hicma-x-dev
export home=$PWD

echo "update submodules"
#git submodule upda"te --init --recursive

# Clean build directory to avoid ninja configuration issues
echo "Cleaning build directory..."
rm -rf build
mkdir -p build
cd build


cmake ../ \
  -DCMAKE_INSTALL_PREFIX=`pwd`/installdir \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpic++ \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DHICMA_PARSEC_HAVE_HIP=ON \
  -DCMAKE_HIP_FLAGS="-I$CRAY_MPICH_DIR/include -I$ROCM_PATH/include -D__HIP_PLATFORM_AMD__" \
  -DCMAKE_C_FLAGS="-I$ROCM_PATH/include -D__HIP_PLATFORM_AMD__" \
  -DCMAKE_CXX_FLAGS="-I$ROCM_PATH/include -D__HIP_PLATFORM_AMD__" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$ROCM_PATH/lib" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$ROCM_PATH/lib" \
  -DROCM_PATH="$ROCM_PATH"


####################################################################################
echo "Install starsh"
cd stars-h && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cc -DCMAKE_INSTALL_PREFIX=$PWD/installdir -DMPI=OFF -DOPENMP=OFF -DSTARPU=ON -DEXAMPLES=ON -DTESTING=ON -DBUILD_SHARED_LIBS=ON -DBLAS_LIBRARIES="-L ${OLCF_OPENBLAS_ROOT}/lib -lopenblas" -DCBLAS_LIBRARIES="-L ${OLCF_OPENBLAS_ROOT}/lib -lopenblas" -DGSL=ON
make -j 8 install
echo "export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/env_hicma_parsec.sh
export PKG_CONFIG_PATH=$home/stars-h/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Install hcore"
cd $home/hcore 
git apply $home/tools/hcore.patch
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DCMAKE_C_COMPILER=cc -DBLAS_LIBRARIES="-L ${OLCF_OPENBLAS_ROOT}/lib -lopenblas" -DCBLAS_LIBRARIES="-L ${OLCF_OPENBLAS_ROOT}/lib -lopenblas" -DLAPACKE_DIR="${OLCF_OPENBLAS_ROOT}" -DLAPACKE_LIBRARIES="-L ${OLCF_OPENBLAS_ROOT}/lib -lopenblas" -DLAPACK_LIBRARIES="-L ${OLCF_OPENBLAS_ROOT}/lib -lopenblas" -DCMAKE_EXE_LINKER_FLAGS="-L ${OLCF_OPENBLAS_ROOT}/lib" -DHCORE_TESTING=OFF -DCMAKE_C_FLAGS="-Wno-implicit-function-declaration"
make -j 8 install
echo "export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:\$PKG_CONFIG_PATH" >> $home/env_hicma_parsec.sh
export PKG_CONFIG_PATH=$home/hcore/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH

echo "Install hicma_parsec"
cd $home
# Clean and recreate build directory to avoid ninja issues
rm -rf build_hicma
mkdir -p build_hicma && cd build_hicma
# Configure hicma_parsec with timeout protection
echo "Configuring hicma_parsec cmake with timeout protection..."
timeout 300 cmake ../ -DCMAKE_INSTALL_PREFIX=`pwd`/installdir \
  -DCMAKE_C_COMPILER=mpicc \
  -DCMAKE_CXX_COMPILER=mpic++ \
  -DCMAKE_Fortran_COMPILER=mpif90 \
  -DCMAKE_C_FLAGS="" \
  -DCMAKE_BUILD_TYPE="Release" \
  -DCMAKE_C_FLAGS_RELEASE="-DNDEBUG" \
  -G Ninja \
  -DDPLASMA_PRECISIONS="s;d" \
  -DLAPACKE_DIR="${OLCF_OPENBLAS_ROOT}" \
  -DLAPACKE_LIBRARIES="-L ${OLCF_OPENBLAS_ROOT}/lib -lopenblas" \
  -DBLAS_LIBRARIES="-L ${OLCF_OPENBLAS_ROOT}/lib -lopenblas" \
  -DBLAS_INCLUDE_DIRS="$OLCF_OPENBLAS_ROOT/include" \
  -DCMAKE_C_FLAGS="-Wno-implicit-function-declaration" \
  -DPARSEC_MAX_DEP_OUT_COUNT=16 \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DHICMA_PARSEC_HAVE_HIP=ON \
  -DCMAKE_INCLUDE_PATH="$CRAY_MPICH_DIR/include" \
  -DCMAKE_LIBRARY_PATH="$CRAY_MPICH_DIR/lib" \
  -DMPI_C_INCLUDE_PATH="$CRAY_MPICH_DIR/include" \
  -DMPI_C_LIBRARIES="$CRAY_MPICH_DIR/lib/libmpi_amd.so" \
  -DMPI_CXX_INCLUDE_PATH="$CRAY_MPICH_DIR/include" \
  -DMPI_CXX_LIBRARIES="$CRAY_MPICH_DIR/lib/libmpi_amd.so" \
  -DMPI_Fortran_INCLUDE_PATH="$CRAY_MPICH_DIR/include" \
  -DMPI_Fortran_LIBRARIES="$CRAY_MPICH_DIR/lib/libmpi_amd.so" \
  -DCMAKE_HIP_FLAGS="-I$CRAY_MPICH_DIR/include"

if [ $? -ne 0 ]; then
    echo "CMake configuration failed or timed out!"
    echo "Trying alternative configuration..."
    timeout 300 cmake ../ -DCMAKE_INSTALL_PREFIX=`pwd`/installdir \
      -DCMAKE_C_COMPILER=mpicc \
      -DCMAKE_CXX_COMPILER=mpic++ \
      -DCMAKE_Fortran_COMPILER=mpif90 \
      -DCMAKE_BUILD_TYPE="Release" \
      -G Ninja \
      -DCMAKE_HIP_ARCHITECTURES=gfx90a \
      -DHICMA_PARSEC_HAVE_HIP=ON \
      -DCMAKE_C_FLAGS="-Wno-implicit-function-declaration"
fi 
#cmake ../ -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpic++ -DCMAKE_Fortran_COMPILER=mpif90 -DCMAKE_C_FLAGS="" -DCMAKE_BUILD_TYPE="Debug" -DCMAKE_C_FLAGS_RELEASE="-DNDEBUG" -G Ninja -DPARSEC_GPU_WITH_HIP=ON -DPARSEC_GPU_WITH_CUDA=OFF -DDPLASMA_PRECISIONS="s;d" -DLAPACKE_DIR="${OLCF_OPENBLAS_ROOT}" -DLAPACKE_LIBRARIES="-L ${OLCF_OPENBLAS_ROOT}/lib -lopenblas" -DBLAS_LIBRARIES="-L ${OLCF_OPENBLAS_ROOT}/lib -lopenblas" -DBLAS_INCLUDE_DIRS="$OLCF_OPENBLAS_ROOT/include" -G Ninja -DCMAKE_C_FLAGS="-Wno-implicit-function-declaration" -DPARSEC_MAX_DEP_OUT_COUNT=16 -DBUILD_SHARED_LIBS=OFF -DCMAKE_HIP_ARCHITECTURES=gfx90a -DDPLASMA_GPU_WITH_HIP=ON 
#cmake ../ -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_Fortran_COMPILER=ftn -DCMAKE_C_FLAGS="" -DCMAKE_BUILD_TYPE="Release" -DCMAKE_C_FLAGS_RELEASE="-DNDEBUG" -G Ninja -DPARSEC_GPU_WITH_HIP=ON -DPARSEC_GPU_WITH_CUDA=OFF -DDPLASMA_PRECISIONS="s;d" -DLAPACKE_DIR="${OLCF_OPENBLAS_ROOT}" -DLAPACKE_LIBRARIES="-L ${OLCF_OPENBLAS_ROOT}/lib -lopenblas" -DBLAS_LIBRARIES="-L ${OLCF_OPENBLAS_ROOT}/lib -lopenblas" -DBLAS_INCLUDE_DIRS="$OLCF_OPENBLAS_ROOT/include" -G Ninja -DCMAKE_C_FLAGS="-Wno-implicit-function-declaration" -DPARSEC_MAX_DEP_OUT_COUNT=16 -DBUILD_SHARED_LIBS=OFF -DCMAKE_HIP_ARCHITECTURES=gfx90a
#cmake ../ -DCMAKE_INSTALL_PREFIX=`pwd`/installdir -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_Fortran_COMPILER=ftn -DCMAKE_C_FLAGS="" -DCMAKE_BUILD_TYPE="Release" -DCMAKE_C_FLAGS_RELEASE="-DNDEBUG" -G Ninja -DPARSEC_GPU_WITH_HIP=ON -DPARSEC_GPU_WITH_CUDA=OFF -DDPLASMA_PRECISIONS="s;d" -DLAPACKE_DIR="${OLCF_OPENBLAS_ROOT}" -DLAPACKE_LIBRARIES="-L ${OLCF_OPENBLAS_ROOT}/lib -lopenblas" -DBLAS_LIBRARIES="-L ${OLCF_OPENBLAS_ROOT}/lib -lopenblas" -DBLAS_INCLUDE_DIRS="$OLCF_OPENBLAS_ROOT/include" -G Ninja -DCMAKE_C_FLAGS="-Wno-implicit-function-declaration" -DPARSEC_MAX_DEP_OUT_COUNT=16 -DBUILD_SHARED_LIBS=OFF
# Build with timeout protection and limited parallel jobs
echo "Building hicma_parsec with ninja (limited to 2 parallel jobs)..."
timeout 3600 ninja -j2

if [ $? -ne 0 ]; then
    echo "Ninja build failed or timed out!"
    echo "Trying with single job..."
    timeout 3600 ninja -j1
fi
#make -j 8
