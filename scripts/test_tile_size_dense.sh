#!/bin/bash

for N in `seq 19200 9600 48000`; do
    for NB in 800 960 1200 1600; do 
        ./tests/testing_potrf_tlr --N $N --t $NB --v 0 --I 10000 | tee -a log_tile_size.txt 
    done
done

grep hicma_parsec_cholesky log_tile_size.txt | tee -a log_tile_size.csv 
