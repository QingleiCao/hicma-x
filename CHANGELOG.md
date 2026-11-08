# Changelog

## 0.2.0 (unreleased)

Release engineering of the 0.1.0 code base plus the features merged on `main`
since November 2025. `docs/code-review.md` records the review this release is
based on, `docs/release-improvement-plan.md` the task list and its status,
`docs/release-readiness.md` the validation results.

### Features (merged before this release cycle)

- Runtime precision decision for the dense mixed-precision Cholesky
  (`--adaptive_decision_runtime 1`, kind 10): every GEMM picks its compute
  precision from the norms of its operands while the factorization runs.
- FP8 (E4M3) GEMMs in every CUDA build, with per-tile power-of-two scaling
  (`--fp8`, `--fp8_scaling`); used by kinds 8, 9 and 10 on GPUs with compute
  capability 8.9 or higher.
- Mixed-precision TRMM (`testing_trmm_mp`), with stochastic rounding
  (`--enable_stochastic_rounding`) and FP32 accumulation (`--accumulation_fp32`).
- DPLASMA and PaRSEC moved to the ICLDisco master branches.

### Correctness

- The correctness check is binding: `check_dpotrf2` and `check_diff` returned
  0 whatever they measured, so `--check 1` could not fail. They now return their
  verdict, the reference dense Cholesky's `info` is inspected, and every driver
  exits non-zero on a failed check or a failed factorization. The verdict is
  the residual ||LL^T-A||_F/||A||_F; the forward error against the dense
  reference factor, ||L-L0||_F/||L||_F, is printed but does not decide (an
  overwrite of `params->info` had always discarded it, and for mixed-precision
  runs it exceeds the residual 10 to 30 times, so one bound cannot serve both).
- `--check_tol` sets the residual bound of `--check` (default: `--fixedacc`);
  a failed check prints the measured residual and the accepted bound.
- `--adddiag` was applied twice for problem kinds 0, 1, 5, 6 and 7 (by the
  STARS-H kernel and again by the generation task) while the reference matrix
  of the check had it once. The shift is now applied exactly once and the
  reference matrix carries it too. Results of these kinds change when
  `--adddiag` is non-zero (the virus-mesh example now reports a residual of
  3e-6 instead of 9e-2).
- `N % NB != 0` is rejected by the Cholesky drivers: the band generation, the
  check and the low-rank kernels assume square tiles and produced wrong results
  (dense FP64 residual of 3e+3) or aborted in HCORE. The TRMM keeps supporting
  partial tiles.
- Runtime-decision GEMM (kind 10): a tile promoted from FP32 to FP64 keeps a
  correctly sized device buffer whatever `--adaptive_decision`/`--adaptive_memory`
  are, instead of only when one of them is set.
- HCORE's storage maxrank was passed as `2*maxrank` in kinds 2, 6 and 7,
  disabling the rank cap that every buffer is sized for; `compmaxrank` is kept
  at least `maxrank`; the per-stream GPU buffers and the CPU full-tile pools
  hold an `mb x 2*maxrank` operand.
- The dense FP16 GPU GEMM aborts on an unsupported `--tensor_gemm` mask (0x80)
  instead of skipping the update; the Cholesky path rejects 0x80 up front
  (it is a TRMM-only mode).
- CPU low-rank TRSM solves with the size of the diagonal block, not the row
  count of the C tile; CPU TRSM treats FP8-decided tiles like SP/HP ones.
- Memory leak: `parsec_private_memory_fini` destructs pooled items without
  freeing their storage, so every workspace a taskpool's pools handed out was
  lost when the taskpool was destroyed: 167 MB per kind-2 factorization at
  N=2700 with 36 threads, about 100 MB per kind-10 factorization at N=8192,
  again at every call (`--nruns`, applications that factorize repeatedly).
  `hicma_parsec_memory_pool_fini` frees the storage first and replaces the 22
  calls in HiCMA-X; peak memory of eight kind-10 runs went from 1238 MB to
  605 MB. Found with AddressSanitizer's leak checker.
- Kind 3 (DENSE_MP_BAND) on GPUs: every run with a half-precision band died
  with `MPI_ERR_TRUNCATE` (a `[type = HALF]` dependency made PaRSEC reshape the
  FP32 tile into a buffer half its size), and `--datatype_convert 1`, a
  kind-3-only mode, crashed on CPU and GPU (the send-decision step rewrote it
  to 2, which the kind-3 graph does not implement consistently). Both work;
  kinds 3 and 4 with a half band and the sender mode have GPU tests now.
- `--tensor_gemm 0x10` (FP16 accumulation) computed garbage on GPUs in kind 3
  and in the low-rank GEMMs of kinds 1, 2, 5, 8, 9 and 10: cuBLAS reads alpha
  and beta in the compute precision, and the float values read as half are 0,
  so every update vanished (kind 3 residual 5e+1, now 2.3e-5). The dense GEMM
  and the TRMM already converted them.
- The sparse kinds (6, 7) crashed when the dense band covers the matrix (for
  example a single tile row): the data is then stored in the contiguous
  descriptor, which they did not use. They select it as kind 2 does.
- An invalid `--kind_of_cholesky` printed an error and exited 0; it exits 1.
  Negative sizes are refused; a `--P`/`--Q` that does not cover the
  processes is reported before the grid is chosen.
- Refused instead of silently miscomputed: `--enable_stochastic_rounding` with
  kinds 8, 9, 10; `--datatype_convert 2` without GPUs; kinds 1 to 4 with GPUs
  on HIP; a half-precision band on CPU-only runs of kind 3. Kind 3 with
  `--datatype_convert 2` runs in the receiver mode: it does not implement the
  sender conversion, and with a one-tile FP64 band its GPU TRSM read the FP64
  diagonal factor as FP32.
- HIP builds register DPLASMA's hipBLAS handles (`DPLASMA::HIP::HANDLES`); the
  DPLASMA routines that run on the GPU with `--gpus > 0`, such as the TRMM in
  `--check`, dereferenced a NULL handle.
- Memory safety: write through `NULL+1` in `hicma_parsec_params_init` when
  `argv[0]` has no `/`; 8-byte allocation for a 24-byte struct in
  `readLocsFile`; `info_gpu` allocated with the requested and freed with the
  detected GPU count; `pheno_file` and other parameter fields read or freed
  uninitialized (the parameter struct is now zeroed first); processes that
  see different numbers of GPUs are refused (they took different branches
  around collectives, and PaRSEC keeps the devices it enabled at start-up);
  the Hamming GPU kernel prototypes match their definitions.
- Resource leaks: memory-pool structs of every Cholesky taskpool, GPU device
  index arrays, the cuBLASLt/cuSOLVER setup failures now reported.

### Architecture and maintenance

- The eight tile/LAPACK conversion JDFs (`Lapack_to_Tile{,_Single}`,
  `Tile_to_Lapack{,_single,_sym,_sym_single,_int,_int8}`), which differed
  only in element type, direction and lower-triangle-only, are one JDF with
  one entry point, `hicma_parsec_lapack_tile_convert(parsec, dcA, buffer, P,
  Q, type, lower, into_tiles)`. The named functions remain as one-line
  wrappers; `hicma_parsec_Lapack_to_Tile_sym`, declared in 0.1.0 without a
  definition, now exists. The buffer offset uses the buffer's element size
  (it used the descriptor's), and a P x Q that does not match the
  descriptor's distribution (P and Q swapped, a band distribution) is refused
  on every rank instead of overwriting memory. `tests/testing_lapack_tile.c`
  covers every type, both directions, partial tiles and every grid of the
  run (CTest `lapack_tile_convert`, `lapack_tile_convert_np2`).
- The ten Cholesky wrappers share `src/kernels/potrf_common.{c,h}` (timing
  hooks, chore selection by device type, memory pools, GPU discovery, driver);
  7250 lines of wrapper code became 2500. The JDF bodies are untouched.
- Per-core and per-GPU statistics arrays are allocated once, after device
  discovery.
- Dead code removed: 14 unreferenced JDFs and sources (`copy2double`,
  `isyrkmp_LT`, `gemm_TN_batched`, `gemm_TN_summa`, `gemm_f16_wrapper`,
  `matrix_dsum/ssum`, `cast_double_diag`, `check_read_data`, `sqrt_kernel`,
  `Lapack_to_Tile_sym`, `convert_i2s`, `diagonal_correction_calculation`,
  `climate_emulator_print_double`), two unbuilt test sources,
  `hamming.jl`, ~40 uncalled functions, 14 declared-but-undefined prototypes,
  and the `#if 0` blocks around them.
- Macros and data structures reviewed (`docs/code-review.md`, section 9).
  Unused names removed from the installed headers: 16 datatype identifiers,
  three tile formats, three sparse modes, `DATATYPE_USE`, `EPS_DP`,
  `THRESHOLD_MEMORY_PER_NODE` (its check could not fire), `str`/`xstr`, 29
  HIP aliases, and the fields `HMB`, `start_time_kernel_sqr_sum_vec` and
  `str_problem` of `hicma_parsec_params_t`. `SYNC_TIME_PRINT` is defined once.
  The local switches of the sources (`CALL_HCORE_DSYRK`, `ACC_DP`,
  `OWN_CONVERTOR`, `USE_OPENMP`, ...) and of the climate emulator are
  settable with `-D`, and each value compiles; two climate debug paths did
  not, fixed. The climate header no longer `#undef`s the genomics switches.
  The 28 unnamed `#if 0`/`#if 1` blocks are unwrapped, deleted (debug
  dumps, code that did not compile, superseded alternatives) or, for the
  genomics SYRK, named `GENOMICS_SYRK_ISYRK`. Matrix compression no longer
  scans every tile for non-zeros to feed a dead `1 || flag` test.
- GEMM reordering (`reorder_gemm.jdf`, `hicma_parsec_reorder_gemm`) is kept for
  future use, still not called. Its launch was an `#if 0` block that did not
  compile; it is now real code on the gathered rank array, so the build
  checks it. Its sort swapped the positions but not the keys it compared, and
  its destructor leaked the work pool; both fixed. `reorder_gemm.jdf` lists
  what wiring it into the Cholesky needs.
- Public headers include the PaRSEC and DPLASMA headers that those projects
  install; DPLASMA's test timing macros are replaced by `hicma_parsec_timing.h`.

### Build, install, tests

- `hicma_parsec_matrix_sum(parsec, A, Ar, decisions, lower)`: the sum of a
  tiled matrix whose tiles are stored in different precisions (FP64/FP32 per
  decision, low-rank U and V with their ranks), identical on every process and
  for every process grid; replaces the single-precision `matrix_dsum` and
  `matrix_ssum`, whose tile task never stored its partial sum. Test
  `testing_matrix_sum` (CTest `matrix_sum`, `matrix_sum_np2`), including
  low-rank tiles in a partial last tile row, where V has more rows than U.
- The compile-time switches of `hicma_parsec_internal.h` can be set with
  `-DNAME=value` in `CMAKE_C_FLAGS` (20 had no `#ifndef`); every non-default
  value was compiled on the CPU and CUDA trees; the dead `CHOLESKY_CPU_ONLY`,
  `PERFORMANCE_BAND_BOUND` and `WRITE_A` are gone; the bound-measurement
  switches `TLR_BOUND_WITHOUT_OFFBAND_*` warn at compile time; `_GNU_SOURCE`
  and `_POSIX_C_SOURCE` are no longer redefined. See `docs/code-review.md`,
  section 8.
- Embedding: `hicma_parsec_init` crashed when given `argc = 0, argv = NULL`
  and when the argument strings were read-only (`--opt=value` was cut in
  place); both work now and `argv` is only read. `libhicma_parsec.so` is
  linked with `-Bsymbolic-functions`: it exports generic names
  (`parse_arguments`, and from STARS-H `diff`, `cube`, ...), and an
  application function of the same name replaced the one the library called
  (`hicma_parsec_init` ran the application's `parse_arguments` and crashed).
  Six file-local JDF helpers with generic names are `static`.
- The BLAS/LAPACKE directories are recorded in the run-time path of the
  built and installed binaries and of executables linking the installed
  package; with an MKL module that sets `LIBRARY_PATH` only, every test of a
  fresh build failed to start ("libmkl_gf_lp64.so.2: cannot open shared
  object file") unless `LD_LIBRARY_PATH` was set.
- Integer options accept hex with a `0x` prefix (`--tensor_gemm 0x10`, as the
  masks are documented); they were rejected as invalid values.
- The CMake test presets run the HiCMA cases only (they ran DPLASMA's and
  PaRSEC's suites as well); the install consumer also builds and runs the
  conversion test.
- `find_package(HICMAParsec)` works: the package config is generated with
  `configure_package_config_file`, finds PaRSEC, DPLASMA, MPI and CUDA/HIP,
  and the library carries its own link interface. `cmake --install` is
  exercised by CI through `tests/install_consumer`. On HIP the config also
  finds `hipblas` and `rocsolver`, which DPLASMA's link interface names but
  DPLASMA's own config does not look for. With ROCm's Flang (Frontier),
  `cmake --install` stopped at PaRSEC's Fortran module directory, which PaRSEC
  installs but does not create when the compiler fails CMake's F90 check; the
  configure step creates it.
- One version: `project(VERSION 0.2.0)`, SOVERSION 0; `VERSION.txt` and the
  version updater script are gone.
- `cmake_minimum_required(3.21)`; configure no longer runs
  `git submodule update --remote --force`.
- One BLAS/LAPACKE stack: `hicma_parsec`, `hcore` and `starsh` link the
  `LAPACKE::LAPACKE` target that DPLASMA finds. The previous build loaded a
  second MKL interface and threading layer (`mkl_intel_lp64` +
  `mkl_gnu_thread` next to DPLASMA's `mkl_gf_lp64` + `mkl_sequential`), and
  `libhcore.so` recorded no dependency on LAPACKE at all, which made every
  driver fail to link with `--as-needed` (Ubuntu). Select the stack with
  `LAPACKE_ROOT` or `BLA_VENDOR`.
- The installed library and drivers carry a runtime search path (`$ORIGIN`,
  the prefix's lib dir, the directories of MPI/GSL/hwloc): with RUNPATH
  semantics a library's dependencies are searched with its own path, and the
  installed `libhicma_parsec.so` could not find `libdplasma.so.2` without
  `LD_LIBRARY_PATH`. A BLAS outside the system directories still needs
  `LD_LIBRARY_PATH` (CMake does not record the MKL directory).
- `HICMA_PARSEC_GENOMICS` option: the genomics drivers are built only when it
  is on (they need a CUDA build: the genomics kernels have no HIP bodies); the
  default build no longer ships binaries that cannot run. The Cholesky CTest
  cases are not registered in such a build, where `testing_potrf` refuses to run.
- Every HiCMA CTest case requires DPLASMA's BLAS thread-safety test as a
  fixture: a BLAS that cannot take concurrent calls now reports "Not Run"
  instead of wrong residuals. The test gets the BLAS run-time path (on an MKL
  module it could not start). The CPU CI job prints the runner's CPU and
  OpenBLAS variant and sets `OPENBLAS_NUM_THREADS=1`; one run had failed
  inside DPLASMA's reference Cholesky on a single runner, while an otherwise
  identical run passed.
- CTest registers the HiCMA tests (labels `hicma`, `mpi`, `gpu`), all with
  `--check 1`; `scripts/test_CI.sh` runs the same cases and no longer depends
  on a private data path; CI runs them and a CUDA compile job. The script and
  `scripts/validation_matrix.sh` launch the two-rank cases with `-n` (mpirun,
  or srun inside a Slurm allocation; skipped without a launcher), skip the
  GPU cases of kinds 1 to 4 on HIP and use `device_hip_memory_use` there.
- `scripts/` reorganized: generic tools at the top level, site recipes under
  `scripts/machines/`, genomics data tools under `scripts/genomics/`,
  byte-identical and obsolete scripts removed; the stale Dockerfiles removed.
- The machine build scripts are rewritten for the current build: one
  configure of the whole tree, `[build-dir] [extra CMake options]`, modules
  loaded in a login shell. They built and installed STARS-H and HCORE
  separately (unused: the build compiles the submodules), passed PaRSEC's GPU
  flags without `HICMA_PARSEC_HAVE_CUDA` (Leonardo without GSL: HiCMA-X built
  without CUDA), ran CMake from the wrong directory (Fugaku), loaded Python 2
  (Shaheen; DPLASMA needs Python 3) or emptied an existing `build/`
  (`build_hicma_with_lapack.sh`). `build_hicma_with_lapack.sh` builds a pinned
  LAPACK (3.12.1) once and passes the search hints STARS-H and HCORE need;
  `build_release.sh` uses `cmake --build`. `scripts/machines/legacy/` (Summit,
  Saturn, ECRC, ExaGeoStat builds) is removed.
- CUDA builds compile for the documented default architectures, 70 80 90 90a
  (80 90 90a with CUDA 13), unless `CMAKE_CUDA_ARCHITECTURES` or `CUDAARCHS`
  is given. The default never applied: `enable_language(CUDA)` cached the
  compiler's default first, so a build with only `-DHICMA_PARSEC_HAVE_CUDA=ON`
  compiled its CUDA code for sm_52 alone, which newer GPUs ran after JIT
  compilation.
- `.gitignore` covers build trees, logs and editor state; LICENSE placeholder
  removed.

### Usability (external-user pass)

- `--nruns n` crashed in TLR runs: run 2 factorized run 1's factor, whose
  ranks outgrow the tile buffers. Where A cannot be regenerated in place,
  one run is done and a warning says why.
- Notes for three silent performance traps: GPUs visible in a GPU build run
  without `--gpus` (7x slower on an A100); kinds 1 and 2 without
  `--band_dense`, where every tile is dense (6.5x slower at N=30720); a
  launcher binding each process to one core, the Open MPI default (10x
  slower with two processes on a 36-core node).
- `--help` and `TESTS.md` gave wrong defaults: `--band_dense` defaults to
  NT (every tile dense), not 1; `--N` and `--NB` default to 2048, not
  "required"; `--HNB` to 300; `--cores` counts physical cores.
- One `R-LO` line per run, without the uninitialized analysis counters of
  dense runs (the drivers zero-initialize the analysis); the version and the
  `git describe` revision are printed (`GITHASH` was never set).
- Internal set-up notes print with `--verbose 1` only; two invalid-option
  exits returned 0.
- `README.md` has a quick start; `scripts/build_release.sh --help` states
  its default build type (RelWithDebInfo).

### Documentation

- `TESTS.md` regenerated from the option table; `BUILD.md`, `README.md` and
  `RELEASE.md` updated; `CHANGELOG.md` added.

### Known limitations (see `docs/release-readiness.md`)

- Cholesky drivers need `N % NB == 0`.
- Kinds 1 to 4 have no HIP bodies. HIP builds are compiled and their CPU cases
  run (Frontier, gfx90a); HIP GPU execution is not validated.
- `--enable_stochastic_rounding` is not implemented for kinds 8 to 10.
- The CPU and GPU runtime-decision GEMMs use different storage policies for
  FP16 updates.
- Mixed-precision runs reach the accuracy of the precisions they use, not an
  FP64-level `--fixedacc`; use `--check_tol`.

## 0.1.0 (2025-11-12)

First public release of HiCMA-X (then named HiCMA-PaRSEC): tile low-rank and
mixed-precision Cholesky factorizations on PaRSEC with CUDA and HIP support,
the climate emulator and the genomics (kernel ridge regression) applications.
