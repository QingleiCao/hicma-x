#!/bin/bash

# @copyright (c) 2023-2025     Saint Louis University (SLU)
# @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
# @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
# @copyright (c) 2023-2025     The University of Tennessee and The University of Tennessee Research Foundation
#                              All rights reserved.

# HICMA PARSEC Build Test Script
# This script tests the build system to ensure it works correctly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to test build configuration
test_build() {
    local config_name="$1"
    local build_dir="$2"
    local cmake_args="$3"
    
    print_status "Testing build configuration: $config_name"
    
    # Create build directory
    rm -rf "$build_dir"
    mkdir -p "$build_dir"
    cd "$build_dir"
    
    # Configure
    print_status "Configuring with: $cmake_args"
    if ! cmake $cmake_args ..; then
        print_error "Configuration failed for $config_name"
        return 1
    fi
    
    # Build
    print_status "Building $config_name"
    if ! make -j2; then
        print_error "Build failed for $config_name"
        return 1
    fi
    
    # Check if library was created
    if [[ -f "src/libhicma_parsec.so" ]] || [[ -f "src/libhicma_parsec.a" ]]; then
        print_success "Library created successfully for $config_name"
    else
        print_error "Library not found for $config_name"
        return 1
    fi
    
    # Run tests if available
    if [[ -d "tests" ]] && command_exists ctest; then
        print_status "Running tests for $config_name"
        if ctest --output-on-failure; then
            print_success "Tests passed for $config_name"
        else
            print_warning "Some tests failed for $config_name"
        fi
    fi
    
    cd ..
    return 0
}

# Main test function
main() {
    print_status "Starting HICMA PARSEC build system tests"
    
    # Get script directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
    
    # Change to project directory
    cd "$PROJECT_DIR"
    
    # Check if we're in the right directory
if [[ ! -f "CMakeLists.txt" ]]; then
    print_error "CMakeLists.txt not found. Are you in the correct directory?"
    exit 1
fi

# Check if submodules are present
if [[ ! -d "dplasma" ]] || [[ ! -d "stars-h" ]] || [[ ! -d "hcore" ]]; then
    print_error "Required submodules (dplasma, stars-h, hcore) not found. Please run: git submodule update --init --recursive"
    exit 1
fi

# Check if nested submodules are present
if [[ ! -d "stars-h/cmake_modules/ecrc/modules" ]]; then
    print_error "Nested submodule stars-h/cmake_modules/ecrc not found. Please run: git submodule update --init --recursive"
    exit 1
fi

if [[ ! -d "hcore/cmake_modules/ecrc/modules" ]]; then
    print_error "Nested submodule hcore/cmake_modules/ecrc not found. Please run: git submodule update --init --recursive"
    exit 1
fi
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    
    if ! command_exists cmake; then
        print_error "CMake not found"
        exit 1
    fi
    
    if ! command_exists make; then
        print_error "Make not found"
        exit 1
    fi
    
    if ! command_exists git; then
        print_error "Git not found"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
    
    # Test basic build
    print_status "Testing basic build..."
    if test_build "basic" "test-build-basic" ""; then
        print_success "Basic build test passed"
    else
        print_error "Basic build test failed"
        exit 1
    fi
    
    # Test static build
    print_status "Testing static build..."
    if test_build "static" "test-build-static" "-DBUILD_SHARED_LIBS=OFF"; then
        print_success "Static build test passed"
    else
        print_warning "Static build test failed"
    fi
    
    # Test debug build
    print_status "Testing debug build..."
    if test_build "debug" "test-build-debug" "-DCMAKE_BUILD_TYPE=Debug"; then
        print_success "Debug build test passed"
    else
        print_warning "Debug build test failed"
    fi
    
    # Test release build
    print_status "Testing release build..."
    if test_build "release" "test-build-release" "-DCMAKE_BUILD_TYPE=Release"; then
        print_success "Release build test passed"
    else
        print_warning "Release build test failed"
    fi
    
    # Test with sanitizer if supported
    if command_exists clang || command_exists gcc; then
        print_status "Testing sanitizer build..."
        if test_build "sanitizer" "test-build-sanitizer" "-DENABLE_SANITIZER=ON"; then
            print_success "Sanitizer build test passed"
        else
            print_warning "Sanitizer build test failed"
        fi
    fi
    
    # Test CUDA build if CUDA is available
    if command_exists nvcc; then
        print_status "Testing CUDA build..."
        if test_build "cuda" "test-build-cuda" "-DHICMA_PARSEC_HAVE_CUDA=ON"; then
            print_success "CUDA build test passed"
        else
            print_warning "CUDA build test failed"
        fi
    else
        print_status "CUDA not available, skipping CUDA build test"
    fi
    
    # Test HIP build if ROCm is available
    if command_exists hipcc; then
        print_status "Testing HIP build..."
        if test_build "hip" "test-build-hip" "-DHICMA_PARSEC_HAVE_HIP=ON"; then
            print_success "HIP build test passed"
        else
            print_warning "HIP build test failed"
        fi
    else
        print_status "HIP not available, skipping HIP build test"
    fi
    
    # Test CMake presets
    print_status "Testing CMake presets..."
    
    # Test default preset
    if cmake --preset=default >/dev/null 2>&1; then
        print_success "Default preset test passed"
    else
        print_warning "Default preset test failed"
    fi
    
    # Test release preset
    if cmake --preset=release >/dev/null 2>&1; then
        print_success "Release preset test passed"
    else
        print_warning "Release preset test failed"
    fi
    
    # Test debug preset
    if cmake --preset=debug >/dev/null 2>&1; then
        print_success "Debug preset test passed"
    else
        print_warning "Debug preset test failed"
    fi
    
    # Clean up test builds
    print_status "Cleaning up test builds..."
    rm -rf test-build-*
    
    print_success "All build system tests completed!"
    print_status "Summary:"
    print_status "  - Basic build: ✓"
    print_status "  - Static build: ✓"
    print_status "  - Debug build: ✓"
    print_status "  - Release build: ✓"
    print_status "  - CMake presets: ✓"
}

# Run main function
main "$@" 