#!/bin/bash

# DENSE_TLR_DP 
cmd="./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1.0e-8 -u 270 -D 2 -P 1 -a 0 -E 0 -I 10 -Z 10 -i 50 -W 2 -g 1 -v 2"
echo $cmd
eval $cmd

cmd="./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1.0e-8 -u 270 -D 2 -P 1 -a 0 -E 0 -I 10 -Z 10 -i 50 -W 2 -g 1 -v 2 -x"
echo $cmd
eval $cmd

echo ""
echo ""
echo ""

# DENSE_TLR_MP 
cmd="./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1.0e-8 -u 270 -D 2 -P 1 -a 1 -E 0 -I 10 -Z 10 -i 50 -W 1 -g 1 -v 2"
echo $cmd
eval $cmd

cmd="./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1.0e-8 -u 270 -D 2 -P 1 -a 1 -E 0 -I 10 -Z 10 -i 50 -W 1 -g 1 -v 2 -x"
echo $cmd
eval $cmd

echo ""
echo ""
echo ""

# DENSE_SP_HP_BAND 
cmd="./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1.0e-8 -u 270 -D 2 -P 0 -a 0 -E 0 -I 0 -y 10 -Z 50 -i 50 -W 4 -g 1 -v 2"
echo $cmd
eval $cmd

cmd="./hicma_parsec/testing_potrf_tlr -N 5400 -t 270 -e 1.0e-8 -u 270 -D 2 -P 0 -a 0 -E 0 -I 0 -y 10 -Z 50 -i 50 -W 4 -g 1 -x -v 2"
echo $cmd
eval $cmd
