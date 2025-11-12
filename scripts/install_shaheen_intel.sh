#intel
module swap PrgEnv-cray PrgEnv-intel
module load cmake
export PKG_CONFIG_PATH=/project/k1205/omairyrm/gbgwas/hicma-x-dev/stars-h-dev/build-intel/install/lib/pkgconfig/:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/project/k1205/omairyrm/gbgwas/hicma-x-dev/hcore/build-intel/install/lib/pkgconfig/:$PKG_CONFIG_PATH
