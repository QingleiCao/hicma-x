#!/bin/bash

# DENSE_TLR_MP: MP+Dense 
cmd="mpirun -np 4 --mca btl_openib_allow_ib 1 --map-by ppr:1:node ./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1e-8 -u 130 -D 2 -P 2 -E 0 -s 0 -W 1 -z 30 -a 1 -Z 1000 -v 2"
echo $cmd
eval $cmd

cmd="mpirun -np 4 --mca btl_openib_allow_ib 1 --map-by ppr:1:node ./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1e-8 -u 130 -D 2 -P 2 -E 0 -s 0 -W 1 -z 30 -a 1 -Z 1000 -x -v 2"
echo $cmd
eval $cmd

echo ""
echo ""
echo ""

# DENSE_TLR_MP: MP+Dense/TLR 
cmd="mpirun -np 4 --mca btl_openib_allow_ib 1 --map-by ppr:1:node ./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1e-8 -u 130 -D 2 -P 2 -E 1 -s 0 -W 1 -z 30 -a 1 -v 2"
echo $cmd
eval $cmd

cmd="mpirun -np 4 --mca btl_openib_allow_ib 1 --map-by ppr:1:node ./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1e-8 -u 130 -D 2 -P 2 -E 1 -s 0 -W 1 -z 30 -a 1 -x -v 2"
echo $cmd
eval $cmd

echo ""
echo ""
echo ""

# DENSE_TLR_DP 
cmd="mpirun -np 4 --mca btl_openib_allow_ib 1 --map-by ppr:1:node ./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1e-8 -u 130 -D 2 -P 2 -E 1 -s 0 -W 2 -z 30 -v 2"
echo $cmd
eval $cmd

cmd="mpirun -np 4 --mca btl_openib_allow_ib 1 --map-by ppr:1:node ./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1e-8 -u 130 -D 2 -P 2 -E 1 -s 0 -W 2 -z 30 -x -v 2"
echo $cmd
eval $cmd

echo ""
echo ""
echo ""

# Sparse
cmd="mpirun -np 4 --mca btl_openib_allow_ib 1 --map-by ppr:1:node ./hicma_parsec/testing_potrf_tlr -N 10370 -t 1037  -e 1e-4 -u 50 -j 0.1 -D 6 -Z 1 -F 0 -Y 1 -z 100 -E 0 -M ../stars-h/SARS-CoV-2-meshes/singleviursdata/SortVirus10370.txt -K 1 -B 0 -R 4.63e-04 -O 2 -S -1 -P 2 -s 1 -d 2"
echo $cmd
eval $cmd

cmd="mpirun -np 4 --mca btl_openib_allow_ib 1 --map-by ppr:1:node ./hicma_parsec/testing_potrf_tlr -N 10370 -t 1037  -e 1e-4 -u 50 -j 0.1 -D 6 -Z 1 -F 0 -Y 1 -z 100 -E 0 -M ../stars-h/SARS-CoV-2-meshes/singleviursdata/SortVirus10370.txt -K 1 -B 0 -R 4.63e-04 -O 2 -S -1 -P 2 -s 1 -d 2 -x"
echo $cmd
eval $cmd

echo ""
echo ""
echo ""

# sparse balance workload 
cmd="mpirun -np 4 --mca btl_openib_allow_ib 1 --map-by ppr:1:node ./hicma_parsec/testing_potrf_tlr -N 10370 -t 1037  -e 1e-4 -u 50 -j 0.1 -D 6 -Z 1 -F 0 -Y 1 -z 100 -E 0 -M ../stars-h/SARS-CoV-2-meshes/singleviursdata/SortVirus10370.txt -K 1 -B 0 -R 4.63e-04 -O 2 -S -1 -P 2 -s 2 -d 2"
echo $cmd
eval $cmd

cmd="mpirun -np 4 --mca btl_openib_allow_ib 1 --map-by ppr:1:node ./hicma_parsec/testing_potrf_tlr -N 10370 -t 1037  -e 1e-4 -u 50 -j 0.1 -D 6 -Z 1 -F 0 -Y 1 -z 100 -E 0 -M ../stars-h/SARS-CoV-2-meshes/singleviursdata/SortVirus10370.txt -K 1 -B 0 -R 4.63e-04 -O 2 -S -1 -P 2 -s 2 -d 2 -x"
echo $cmd
eval $cmd

echo ""
echo ""
echo ""

# DENSE_MP_BAND 
cmd="mpirun -np 4 --mca btl_openib_allow_ib 1 --map-by ppr:1:node ./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1.0e-8 -u 270 -D 2 -P 2 -a 0 -E 0 -I 5 -y 10 -Z 50 -W 3 -g 0 -v 2"
echo $cmd
eval $cmd

cmd="mpirun -np 4 --mca btl_openib_allow_ib 1 --map-by ppr:1:node ./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1.0e-8 -u 270 -D 2 -P 2 -a 0 -E 0 -I 5 -y 10 -Z 50 -W 3 -g 0 -x -v 2"
echo $cmd
eval $cmd

echo ""
echo ""
echo ""

# DENSE_SP_HP_BAND 
cmd="mpirun -np 4 --mca btl_openib_allow_ib 1 --map-by ppr:1:node ./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1.0e-8 -u 270 -D 2 -P 0 -a 0 -E 0 -I 0 -y 10 -Z 50 -i 50 -W 4 -g 0 -v 2"
echo $cmd
eval $cmd

cmd="mpirun -np 4 --mca btl_openib_allow_ib 1 --map-by ppr:1:node ./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1.0e-8 -u 270 -D 2 -P 0 -a 0 -E 0 -I 0 -y 10 -Z 50 -i 50 -W 4 -g 0 -x -v 2"
echo $cmd
eval $cmd
