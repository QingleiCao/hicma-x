#!/bin/bash

# Script to build LAPACK first, then configure and build the main project
# This ensures the external LAPACK library is available before STARS-H tries to find it

set -e

# Store the original directory
ORIGINAL_DIR="$(pwd)"

echo "=== Building HICMA with External LAPACK ==="
echo "This script will:"
echo "1. Build LAPACK first"
echo "2. Configure the main project with the correct LAPACK paths"
echo "3. Build the main project"
echo ""

# Create a separate directory for LAPACK
LAPACK_DIR="$ORIGINAL_DIR/lapack_build"
echo "Building LAPACK in: $LAPACK_DIR"

# Create LAPACK directory
mkdir -p "$LAPACK_DIR"
cd "$LAPACK_DIR"

# Clone LAPACK if not already present
if [ ! -d "lapack" ]; then
    echo "Cloning LAPACK..."
    git clone https://github.com/Reference-LAPACK/lapack.git
fi

cd lapack

# Create build directory
mkdir -p build
cd build

# Configure LAPACK
echo "Configuring LAPACK..."
cmake .. \
    -DCMAKE_INSTALL_PREFIX="$LAPACK_DIR/installdir" \
    -DBUILD_SHARED_LIBS=ON \
    -DCBLAS=ON \
    -DLAPACKE=ON \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DLAPACKE_WITH_TMG=ON

# Build LAPACK
echo "Building LAPACK..."
make -j$(nproc)

# Install LAPACK
echo "Installing LAPACK..."
make install

echo "LAPACK build completed successfully!"
echo "LAPACK installed to: $LAPACK_DIR/installdir"

# Add LAPACK libraries to LD_LIBRARY_PATH for runtime linking
echo "Adding LAPACK libraries to LD_LIBRARY_PATH..."
export LD_LIBRARY_PATH="$LAPACK_DIR/installdir/lib64:$LD_LIBRARY_PATH"
echo "LD_LIBRARY_PATH updated: $LD_LIBRARY_PATH"

# Now configure and build the main project
echo ""
echo "=== Configuring Main Project ==="

# Go to the main project build directory
cd "$ORIGINAL_DIR/build"

# Clean the build directory
echo "Cleaning build directory..."
rm -rf *

# Configure the main project with the correct LAPACK paths
echo "Configuring main project with LAPACK paths..."
cmake .. \
    -DCMAKE_BUILD_TYPE="Debug" \
    -G Ninja \
    -DHICMA_PARSEC_HAVE_CUDA=ON \
    -DDPLASMA_WITH_SCALAPACK_WRAPPER=OFF \
    -DBLAS_DIR="$LAPACK_DIR/installdir" \
    -DBLAS_LIBDIR="$LAPACK_DIR/installdir/lib64" \
    -DCBLAS_DIR="$LAPACK_DIR/installdir" \
    -DCBLAS_LIBDIR="$LAPACK_DIR/installdir/lib64" \
    -DLAPACKE_DIR="$LAPACK_DIR/installdir" \
    -DLAPACKE_LIBDIR="$LAPACK_DIR/installdir/lib64" \
    -DLAPACK_DIR="$LAPACK_DIR/installdir" \
    -DLAPACK_LIBDIR="$LAPACK_DIR/installdir/lib64" \
    -DBLA_VENDOR=Generic

echo "Configuration completed successfully!"

# Build the main project
echo ""
echo "=== Building Main Project ==="
echo "Building with ninja..."
ninja

echo ""
echo "=== Build Completed Successfully! ==="
echo "All targets built successfully with external LAPACK support."
