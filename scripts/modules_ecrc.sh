module purge
module load ecrc-extras
module load cmake-3.22.1-gcc-7.5.0-4se4k5d 
module load gcc/10.2.0 
module load mkl/2019-update-5
module load cmake/3.19.2
module load openmpi/4.1.0-gcc-10.2.0
module load cuda/11.6 
#module load  cuda/11.6
export PKG_CONFIG_PATH=/home/omairyrm/gbgwas/hicma-x-dev/stars-h-dev/build/install/lib/pkgconfig/:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=/home/omairyrm/gbgwas/hicma-x-dev/hcore/build/install/lib/pkgconfig/:$PKG_CONFIG_PATH
#
#export LD_LIBRARY_PATH=/home/omairyrm/intel/oneapi/mkl/2021.2.0/lib/intel64:$LD_LIBRARY_PATH
#export MKLROOT=/home/omairyrm/intel/oneapi/mkl/2021.2.0
 . "/home/omairyrm/anaconda3/etc/profile.d/conda.sh"
 conda activate
