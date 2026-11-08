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
| Linux x86_64 + CUDA 12 (A100; H100 for FP8) | tested on one A100 (1 and 2 ranks sharing it); H100 FP8 results from the development of this release are in `PR` history, not re-run here |
| Linux + ROCm 6 (HIP, gfx90a) | compiles; not run on AMD hardware for this release |
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
- `--tensor_gemm 0x80` is a TRMM-only mode; `--datatype_convert 2` needs GPUs.
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

## 0.1.0

First public release (2025-11-12).
