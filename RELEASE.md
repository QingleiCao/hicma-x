# HiCMA-X release notes

## 0.2.0

HiCMA-X (formerly HiCMA-PaRSEC) is a library of tile low-rank and
mixed-precision linear algebra on the PaRSEC runtime: Cholesky factorizations
with dense, banded, low-rank and adaptively chosen tile precisions (FP64, FP32,
FP16, FP8), a mixed-precision TRMM, and the kernels behind the climate emulator
and the genomics (kernel ridge regression) applications.

### What is new since 0.1.0

- Runtime precision decisions for the dense mixed-precision Cholesky
  (`--adaptive_decision_runtime 1`), FP8 GEMMs with per-tile scaling, and a
  mixed-precision TRMM with stochastic rounding.
- A release-quality pass over the existing code: the correctness check is
  binding and drives the exit code, the diagonal shift is applied once, several
  memory-safety bugs in initialization and the file readers are fixed, the
  installed package is usable, and the ten Cholesky wrappers share one
  implementation. The full list is in `CHANGELOG.md`; the review that led to it
  is in `docs/code-review.md`.

### Supported configurations

| Configuration | Status in 0.2.0 |
|---|---|
| Linux x86_64, CPU only (OpenBLAS or MKL, LAPACKE, Open MPI or MPICH) | tested (CI, 1 and 2 ranks) |
| Linux x86_64 + CUDA 12 (A100; H100 for FP8) | tested on one A100 (1 and 2 ranks sharing it), one 8-A100 node (1 to 8 ranks) and one H100 (FP8; 1 and 2 ranks) |
| Linux + ROCm 6 (HIP, gfx90a) | compiles; the CPU cases of a HIP build ran on Frontier; not run on AMD GPUs |
| Fugaku (`ON_FUGAKU`), macOS, Windows | not supported |

Requirements: CMake 3.21, a C11/C++17 compiler, a Fortran compiler (BLAS/LAPACK
linkage), MPI with `MPI_THREAD_MULTIPLE`, BLAS, LAPACKE; CUDA 12.0 or later for
GPU builds (`90a` architecture needs 12.0); GSL optional (Matérn kernels).

### Known limitations

- The Cholesky drivers require `N` to be a multiple of `NB` (rejected at
  start-up). The TRMM supports partial tiles.
- Cholesky kinds 1 to 4 have CUDA bodies only; on HIP they run on the CPU.
  Kind 3 with a half-precision band runs on GPUs only (the CPU bodies have no
  half path; without GPUs the single-precision band is widened to the matrix).
- `--enable_stochastic_rounding` is implemented for kinds 1 to 5 and the TRMM,
  not for kinds 8 to 10 and not together with FP8 GEMMs.
- `--tensor_gemm` variants other than 0x8 are not implemented on every path
  (`TESTS.md`, "GPU GEMM variants"); `--datatype_convert 2` needs GPUs.
- A static FP8 band near the diagonal can make the matrix indefinite (the
  factorization reports `info` > 0; with low-rank tiles HCORE aborts on the
  NaNs). Keep it away from the diagonal or use `--adaptive_decision`.
- In runtime-decision mode the CPU GEMM keeps FP16 updates into an FP32 tile in
  FP64 while the GPU GEMM keeps FP32; CPU-only and GPU runs of kind 10 therefore
  differ in their storage precisions (results are within tolerance in both).
- The residual of mixed-precision runs is bounded by the precisions used, not by
  `--fixedacc`; pass `--check_tol` with the bound you expect (see `TESTS.md`).
- The genomics drivers need `-DHICMA_PARSEC_GENOMICS=ON` and a CUDA build; the
  climate emulator needs its external data set. Neither is covered by CI.
- Several ranks sharing one GPU need `-- --mca device_cuda_memory_use <percent>`
  (`device_hip_memory_use` on AMD GPUs).

### Validation

`docs/release-readiness.md` records what was built and run for this release:
CPU, CUDA (A100) and HIP (compile only) configurations, the CTest suite, the
validation matrix (`scripts/validation_matrix.sh`), the install consumer and a
before/after timing comparison.

### Upgrading from 0.1.0

- New: `hicma_parsec_matrix_sum` (mixed-precision checksum). The build-time
  switches of `hicma_parsec_internal.h` (`PRINT_RANK`, `DEBUG_INFO`,
  `WORKLOAD_BALANCE`, ...) are set with `-DNAME=value` in `CMAKE_C_FLAGS`
  instead of editing the header. `GPU_BUFFER_ONCE` is gone: the GPU
  workspace is always allocated once.
- Removed from the installed headers because nothing used them: the macros
  `HICMA_PARSEC_FP64` ... `HICMA_PARSEC_ONE_BIT`, `HICMA_PARSEC_DENSE`,
  `HICMA_PARSEC_LOW_RANK`, `HICMA_PARSEC_SPARSE`, `DATATYPE_USE`,
  `NOT_SPARSE`, `SPARSE_NORMAL`, `SPARSE_BALANCE`, `EPS_DP`,
  `THRESHOLD_MEMORY_PER_NODE`, `str`, `xstr`, and the
  `hicma_parsec_params_t` fields `HMB`, `start_time_kernel_sqr_sum_vec`,
  `str_problem`.
- CUDA builds without `CMAKE_CUDA_ARCHITECTURES` (or `CUDAARCHS`) compile for
  70 80 90 90a (80 90 90a with CUDA 13). 0.1.0 documented this default but
  compiled for the compiler's default, sm_52 with CUDA 12.

- `hicma_parsec_lapack_tile_convert` replaces the eight tile/LAPACK
  conversion taskpools; the named `hicma_parsec_{Tile_to_Lapack,Lapack_to_Tile}*`
  functions remain and now return -1 (on every rank, nothing copied) when the
  descriptor is not 2D block-cyclic over the given grid. Their
  `*_New`/`*_Destruct` constructors, never declared in a header, are gone.

- `libhicma_parsec` has SOVERSION 0 (it was 2, from a stale fallback version).
- BLAS/LAPACKE come from DPLASMA's `find_package(LAPACKE)` for the whole build
  (`LAPACKE_ROOT`, `BLA_VENDOR`); the separate `BLAS_LIBRARIES` discovery and
  the hard-coded MKL link line are gone.
- Results change for problem kinds 0, 1, 5, 6 and 7 when `--adddiag` is non-zero
  (the shift is no longer applied twice), and for TLR kinds 2, 6, 7 only in
  runs that silently overflowed their rank cap before (they now abort with
  HCORE's message, as kind 1 did).
- `--check 1` can fail now. Runs that previously "passed" with a residual
  above the tolerance fail; use `--check_tol` to state the accepted bound.
- The genomics drivers (`testing_syrk`, `testing_KRR*`, `testing_Kmatrix`,
  `testing_igemm_summa`) are built only with `-DHICMA_PARSEC_GENOMICS=ON`.
- Scripts moved: machine recipes to `scripts/machines/`, genomics data tools
  to `scripts/genomics/`; `scripts/test_run*.sh` are replaced by
  `scripts/test_CI.sh` and CTest. The machine scripts configure the whole tree
  once and take `[build-dir] [extra CMake options]`; they no longer build
  STARS-H and HCORE separately or write `env_hicma_parsec.sh`. The scripts of
  retired machines and the ExaGeoStat builds (`scripts/machines/legacy/`) are
  removed.
- `testing_potrf_tlr --nruns n` repeats the factorization only where A can be
  regenerated (the dense layout without `--check`, `--auto_band`,
  `--adaptive_memory` or `--sparse`); elsewhere it does one run and warns.
  0.1.0 factorized the previous run's factor, which timed a different problem
  and crashed in TLR runs.
- One run prints one `R-LO` line (0.1.0 printed a second, incomplete one
  first); with `--nruns n`, one per run plus the final one. Its `GITHASH`
  column holds `git describe` of the source at configure time, and the
  parameter summary starts with the version.
- The set-up notes (band sizes, lookahead, memory layout, FP16 emulation,
  tile-size advice) print only with `--verbose 1`. New notes: GPUs visible
  but `--gpus 0`; a TLR kind (1, 2) with every tile dense; a launcher binding
  that leaves the processes of a node half of its cores or fewer.
- `--left_looking` < 0 and BF16 GEMMs with low-rank tiles exit with status 1
  (they exited 0).
- New `--tensor_gemm_hp` (GPU GEMM variant of the FP16 tiles) and
  `--tensor_gemm_sp` (FP32 or TF32). `--tensor_gemm` sets both: one FP16
  variant (0x8, 0x10, 0x20, 0x40 or 0x80), plus 0x4 for TF32; 0x1 and 0x2
  are accepted and ignored, so the values 0.1.0 handled correctly keep their
  meaning, except with `--adaptive_decision` (below). Refused
  at start-up: no variant or several (0.1.0 gave wrong results in kind 3 and
  aborted mid-run in kind 5), BF16 with low-rank tiles or in kind 3 (wrong
  results), a variant other than 0x8 with `--adaptive_decision_runtime`
  (ignored before), 0x80
  in the Cholesky (exited 0 without factorizing), BF16 in the TRMM.
  `--adaptive_decision` uses the chosen variant and TF32; it always used 0x8
  without TF32, and with 0x10 its dense and low-rank GEMMs disagreed on the
  type of C (NaNs, then a crash). The TRMM drivers now read
  `--tensor_gemm_hp` and check the value too. Also refused: BF16 with
  `--datatype_convert 2` (wrong results), and 0x10 when a low-rank tile has
  rank 0.
- GPU runs of kinds 1 and 5 with a static FP16 band and low-rank tiles
  (`--band_dense_sp` < `--band_dense` < NT) return correct results; 0.1.0
  gave wrong results (residuals 10 to 59) or crashed. So do kind-8 runs with
  an FP8 band and low-rank tiles on GPUs with FP8 (residual 1.5e-2 before).
  Results of such runs change.

## 0.1.0

First public release (2025-11-12).
