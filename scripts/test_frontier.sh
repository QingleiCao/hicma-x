#!/bin/bash

MASK_0="0xfefe000000000000"
MASK_1="0x00000000fefe0000"
MASK_2="0x000000000000fefe"
MASK_3="0x0000fefe00000000"

CPU_MASK="--cpu-bind=mask_cpu:${MASK_0},${MASK_1},${MASK_2},${MASK_3}"

#cmd="srun -N 1 -n 1 --ntasks-per-node=1 --cpus-per-task=14 --gpus-per-task=1 --threads-per-core=1 --cpu_bind=mask_cpu:0xfefe000000000000 --mem-bind=map_mem:3 /ccs/home/cql0536/frontier/hello_jobstep/hello_jobstep"
cmd="srun -N 1 -n 1 --ntasks-per-node=1 --cpus-per-task=7 --gpus-per-task=1 --threads-per-core=1 --cpu_bind=mask_cpu:0x00fe000000000000 --mem-bind=map_mem:3 /ccs/home/cql0536/frontier/hello_jobstep/hello_jobstep"
echo $cmd
eval $cmd
echo ""

# where 2 ranks share a GPU ???
cmd="srun -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=14 --gpus-per-task=2 --threads-per-core=1 ${CPU_MASK} --mem-bind=map_mem:3,1,0,2 /ccs/home/cql0536/frontier/hello_jobstep/hello_jobstep"
echo $cmd
eval $cmd
echo ""

'
MPI 002 - OMP 000 - HWT 015 - Node frontier03840 - RT_GPU_ID 0,1 - GPU_ID 4,5 - Bus_ID d1,d6
MPI 003 - OMP 000 - HWT 047 - Node frontier03840 - RT_GPU_ID 0,1 - GPU_ID 6,7 - Bus_ID d9,de
MPI 001 - OMP 000 - HWT 031 - Node frontier03840 - RT_GPU_ID 0,1 - GPU_ID 2,3 - Bus_ID c9,ce
MPI 000 - OMP 000 - HWT 063 - Node frontier03840 - RT_GPU_ID 0,1 - GPU_ID 0,1 - Bus_ID c1,c6

cql0536@frontier03840:~/frontier/hello_jobstep> rocm-smi --showbus

======================= ROCm System Management Interface =======================
================================== PCI Bus ID ==================================
GPU[0]		: PCI Bus: 0000:C1:00.0
GPU[1]		: PCI Bus: 0000:C6:00.0
GPU[2]		: PCI Bus: 0000:C9:00.0
GPU[3]		: PCI Bus: 0000:CE:00.0
GPU[4]		: PCI Bus: 0000:D1:00.0
GPU[5]		: PCI Bus: 0000:D6:00.0
GPU[6]		: PCI Bus: 0000:D9:00.0
GPU[7]		: PCI Bus: 0000:DE:00.0
'

cmd="srun -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=14 --gpu-bind=per_task:2 --gpus-per-node=8 --threads-per-core=1 ${CPU_MASK} --mem-bind=map_mem:3,1,0,2 /ccs/home/cql0536/frontier/hello_jobstep/hello_jobstep"
echo $cmd
eval $cmd
echo ""

'
MPI 002 - OMP 000 - HWT 015 - Node frontier03840 - RT_GPU_ID 0,1 - GPU_ID 4,5 - Bus_ID d1,d6
MPI 003 - OMP 000 - HWT 047 - Node frontier03840 - RT_GPU_ID 0,1 - GPU_ID 6,7 - Bus_ID d9,de
MPI 001 - OMP 000 - HWT 031 - Node frontier03840 - RT_GPU_ID 0,1 - GPU_ID 2,3 - Bus_ID c9,ce
MPI 000 - OMP 000 - HWT 063 - Node frontier03840 - RT_GPU_ID 0,1 - GPU_ID 0,1 - Bus_ID c1,c6
'

cmd="srun -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=14 --gpus-per-node=8 --gpu-bind=per_task:2 --threads-per-core=1 ${CPU_MASK} --mem-bind=map_mem:3,1,0,2 /ccs/home/cql0536/frontier/hello_jobstep/hello_jobstep"
echo $cmd
eval $cmd
echo "" 


export MPICH_OFI_NIC_POLICY=USER
export MPICH_OFI_NIC_MAPPING="0:0; 1:1; 2:2; 3:3"
cmd="srun -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=14 --gpus-per-node=8 --gpu-bind=per_task:2 --threads-per-core=1 ${CPU_MASK} --mem-bind=map_mem:3,1,0,2 /ccs/home/cql0536/frontier/hello_jobstep/hello_jobstep"
echo $cmd
eval $cmd
echo "" 


export OMP_NUM_THREADS=1

my_home=/ccs/home/cql0536/frontier/hicma-x-dev/build/hicma_parsec/testing_dpotrf_tlr

'
cmd="$my_home -N 102400 -t 2048 -e 1.0e-8 -u 270 -D 2 -P 1 -v -a 0 -E 0 -c 19 -I 10 -Z 150 -i 50 -W 7 -I 10000 -V 3 -g 8"
echo $cmd
eval $cmd
echo ""
# 75 Tflop/s

cmd="$my_home -N 102400 -t 2048 -e 1.0e-8 -u 270 -D 2 -P 1 -v -a 0 -E 0 -c 19 -I 10 -Z 150 -i 50 -W 7 -I 10000 -V 3 -g 2"
echo $cmd
eval $cmd
echo ""
# 35 Tflop/s

cmd="srun -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=14 --gpus-per-task=2 --threads-per-core=1 ${CPU_MASK} --mem-bind=map_mem:3,1,0,2 $my_home -N 102400 -t 2048 -e 1.0e-8 -u 270 -D 2 -P 1 -v -a 0 -E 0 -c 19 -I 10 -Z 150 -i 50 -W 7 -I 10000 -V 3 -g 2 -c 13 -- --mca runtime_num_cores 13 --mca device hip --mca device_hip_enabled 2 --mca device_show_statistics 1"
echo $cmd
eval $cmd
echo ""
# 22 Tflop/s

cmd="srun -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=14 --gpus-per-node=8 --gpu-bind=per_task:2 --threads-per-core=1 ${CPU_MASK} --mem-bind=map_mem:3,1,0,2 $my_home -N 102400 -t 2048 -e 1.0e-8 -u 270 -D 2 -P 1 -v -a 0 -E 0 -c 19 -I 10 -Z 150 -i 50 -W 7 -I 10000 -V 3 -g 2 -c 13 -- --mca runtime_num_cores 13 --mca device hip --mca device_hip_enabled 2 --mca device_show_statistics 1"
echo $cmd
eval $cmd
echo ""
# 23 Tflop/s
'

#srun -N 1 -n 1 --ntasks-per-node=1 --cpus-per-task=56 --gpus-per-node=8 --threads-per-core=1 --cpu_bind=mask_cpu:0xfefefefefefefefe /lustre/orion/csc312/proj-shared/lei/hicma-x-dev/build/hicma_parsec/testing_dpotrf_tlr -N 102400 -t 2048 -e 1.0e-8 -u 270 -D 2 -P 1 -v -a 0 -E 0 -c 19 -I 10 -Z 150 -i 50 -W 7 -I 10000 -V 3 -g 8 -c 55
## 75 Tflop/s


#srun -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=14 --gpus-per-node=8 --gpu-bind=per_task:2 --threads-per-core=1 --cpu-bind=mask_cpu:0xfefe000000000000,0x00000000fefe0000,0x000000000000fefe,0x0000fefe00000000 --mem-bind=map_mem:3,1,0,2 /lustre/orion/csc312/proj-shared/lei/hicma-x-dev/build/hicma_parsec/testing_dpotrf_tlr -N 102400 -t 2048 -e 1.0e-8 -u 270 -D 2 -P 1 -v -a 0 -E 0 -c 19 -I 10 -Z 150 -i 50 -W 7 -I 10000 -V 3 -g 2 -c 13 -- --mca runtime_num_cores 13 --mca device hip --mca device_hip_enabled 2 --mca device_show_statistics 1 --mca runtime_bind_threads 0
#srun -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=14 --gpus-per-node=8 --gpu-bind=per_task:2 --threads-per-core=1 --cpu-bind=mask_cpu:0xfefe000000000000,0x00000000fefe0000,0x000000000000fefe,0x0000fefe00000000 --mem-bind=map_mem:3,1,0,2 /lustre/orion/csc312/proj-shared/lei/hicma-x-dev/build/hicma_parsec/testing_dpotrf_tlr -N 102400 -t 2048 -e 1.0e-8 -u 270 -D 2 -P 1 -v -a 0 -E 0 -c 19 -I 10 -Z 150 -i 50 -W 7 -I 10000 -V 3 -g 2 -c 13 -- --mca runtime_bind_threads 0
# 23 Tflop/s
