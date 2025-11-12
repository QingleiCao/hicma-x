# HiCMA-PaRSEC Testing Guide

## Getting Started

### Running Tests
Go to build folder:

```bash
cd build
../scripts/test_run.sh
```

### Statistics-2d-sqexp

```bash
./tests/testing_potrf_tlr --N 2700 --NB 270 --fixedacc 1e-8 --maxrank 130 --kind_of_problem 2
```

### Statistics-3d-exp

```bash
./tests/testing_potrf_tlr --N 27000 --NB 540 --fixedacc 1e-8 --maxrank 540 --kind_of_problem 4 --auto_band 1 -- -mca runtime_comm_coll_bcast 0
```

### 3D-RBF-Mesh-Coordinates-Virus 

```bash
./tests/testing_potrf_tlr --N 10370 --NB 1037 --fixedacc 1e-4 --maxrank 50 --adddiag 0.1 --kind_of_problem 6 --band_dense 1 --send_full_tile 0 --lookahead 1 --auto_band 0 --mesh_file ../../stars-h/SARS-CoV-2-meshes/singleviursdata/SortVirus10370.txt --numobj 1 --rbf_kernel 0 --radius 4.63e-04 --order 2 --density -1 --grid_rows 2 --sparse 1 --verbose 2
```
**Note:** The input data is generated so that the matrix size is determined by the data, which means the matrix size needs to match the size in the data.

Some simple input data samples are under `$ROOT/stars-h/SARS-CoV-2-meshes/singleviursdata`, and you could change it by `--mesh_file` option.

### Hamming Distance

```bash
./tests/testing_hamming --N 10240 --nsnp=10240 --NB 1024 --gpus 1
```

### Genomics 

It needs to compile with "GENOMICS" enabled.

```bash
./tests/testing_KRR --N 6000 --nsnp 6000 --NB 200 --band_dense_dp 1000 --band_dense_sp 1000 --band_dense 10000 --gpus 1 --kind_of_problem 15 --radius=1.0e-7 --adddiag=1000 --adaptive_decision 1 --kind_of_cholesky 9
```

### Climate-emulator 

```bash
./tests/testing_climate_emulator --latitude 120 --NB 1440 --N 14400 --verbose 1 --gpus 1 --verbose 2 --adaptive_decision 1 --mesh_file "/home/qcao3/data"
```

Note: the input data is needed, which can be downloaded at: https://drive.google.com/drive/u/0/folders/1RnnVJPcoeokI8J0Q5pcytIWqU2muU10t

## Program Parameters

The following options are available (use `--help` to see the complete list):

### HiCMA Options
- `--cores`: Number of CPU cores per process 
- `--gpus`: Number of GPUs
- `--grid_rows`: Grid rows
- `--grid_cols`: Grid columns
- `--N`: Matrix size N
- `--NB`: Tile size, N % NB == 0
- `--check`: Enable correctness check
- `--verbose`: Verbosity level

- `--fixedrank`: Fixed rank
- `--fixedacc`: Fixed accuracy
- `--maxrank`: Maximum rank
- `--genmaxrank`: General maximum rank
- `--compmaxrank`: Compression maximum rank
- `--adddiag`: Add diagonal
- `--band_dense`: Band size dense
- `--band_dist`: Band size distributed
- `--band_p`: Band process grid
- `--lookahead`: Lookahead
- `--kind_of_problem`: Kind of problem
- `--send_full_tile`: Send full tile
- `--auto_band`: Auto band
- `--sparse`: Sparse mode
- `--band_dense_dp`: Band size dense double precision
- `--band_dense_sp`: Band size dense single precision
- `--band_dense_fp8`: Band size dense fp8
- `--band_low_rank_dp`: Band size low rank double precision
- `--adaptive_decision`: Adaptive decision for tile format selection
- `--adaptive_memory`: Adaptive memory allocation (0: memory allocated once; 1: memory reallocated per tile after precision decision)
- `--adaptive_maxrank`: Adaptive maximum rank adjustment
- `--kind_of_cholesky`: Kind of Cholesky
- `--mesh_file`: Mesh file
- `--rbf_kernel`: RBF kernel
- `--radius`: Radius
- `--order`: Order
- `--density`: Density
- `--tensor_gemm`: Tensor GEMM
- `--datatype_convert`: Datatype conversion
- `--band_size_termination`: Band size termination
- `--left_looking`: Left looking
- `--nruns`: Number of runs

- `--time_slots`: Time slots
- `--sigma`: Sigma
- `--beta`: Beta
- `--nu`: Nu
- `--beta_time`: Beta time
- `--nu_time`: Nu time
- `--nonsep_param`: Non-separable parameter
- `--noise`: Noise
- `--latitude`: Latitude use in climate emulator, where N == latitude * latitude

- `--HNB`: HNB
- `--nsnp`: Number of SNPs
- `--numobj`: Number of objects
- `--wavek`: Wave k

### Getting Help

More information:

```bash
./testing_potrf_tlr --help
```

This will display the complete list of available options with descriptions.

Additional PaRSEC flags:
```bash
./testing_potrf_tlr -- --parsec-help
```

## Testing Tips

1. **Performance Optimization**: If the problem is a little dense, i.e., band_size > 1 after auto-tuning (e.g., in statistics-3d-sqexp application with accuracy threshold `--fixedacc 1.0e-8`), `-- -mca runtime_comm_coll_bcast 0` is needed for better performance.

2. **Core Configuration**: Set argument `--cores` to `number_of_cores - 1`.

3. **Process Grid**: Choose the process grid to be as square as possible with P < Q recommended.

4. **Tile Size Optimization**: For TLR, it needs to be tuned for the special problem and the architecture setting (refer to the paper below). However, regarding dense testings, it can be selected by a simple testing. 

5. **Combined Adaptive Features**: Combine `--adaptive_decision` and `--adaptive_memory` for optimal performance-accuracy trade-offs.

6. **Recommended Maxrank Settings**:
   - For `--kind_of_problem 2` (statistics-2d-sqexp): set `--maxrank= 150`
   - For `--kind_of_problem 3` (statistics-3d-sqexp): set `--maxrank= 500`
   - For `--kind_of_problem 4` (statistics-3d-exp): set `--maxrank= tile_size / 2`
   - For `--kind_of_problem 6` (statistics-3d-exp): set `--maxrank= 200` if `--radius 3.7e-03` 

7. **Number of Processes Each Node**: For better performance, one process per node; if one process per NUMA node or GPU, specific setting is needed, e.g., through Slurm on Frontier supercomputer: `srun -N 1 -n 4 --ntasks-per-node=4 --cpus-per-task=14 --gpus-per-node=8 --gpu-bind=per_task:2 --threads-per-core=1 --cpu_bind=mask_cpu:0xfefe000000000000,0x00000000fefe0000,0x000000000000fefe,0x0000fefe00000000 --mem-bind=map_mem:3,1,0,2`. 
