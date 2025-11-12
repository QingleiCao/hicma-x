# Intel env
module swap PrgEnv-cray PrgEnv-intel

# stars-h and gsl
export PKG_CONFIG_PATH=/sw/xc40cle7/gsl/1.14/cle7_intel19.0.1/install/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/lustre/project/k1205/lei/stars-h/build/install/lib/pkgconfig:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/lustre/project/k1205/lei/hicma-x-dev/hcore/build/installdir/lib/pkgconfig:$PKG_CONFIG_PATH
# cmake 3.18 
export PATH=/lustre/project/k1205/lei/software/CMake/bin:$PATH

# cray link type
export CRAYPE_LINK_TYPE=dynamic


