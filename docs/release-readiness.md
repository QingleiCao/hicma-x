# HiCMA-X 0.2.0 release readiness

Branch `qinglei/release-cleanup`, 2026-11-07. Base: `main` at `6e987e3`
(merge of the PaRSEC/DPLASMA update). Review: `docs/code-review.md`; plan and
per-item status: `docs/release-improvement-plan.md`; user-facing notes:
`RELEASE.md`, `CHANGELOG.md`.

Nothing has been tagged, pushed or merged. Those three steps need the
maintainer's authorization (section 7).

## 1. Recommendation

Release as **0.2.0**. The previous tag is `v0.1.0` (2025-11-12). Since then
`main` gained features (runtime precision decision, FP8 GEMMs with per-tile
scaling, the mixed-precision TRMM, the ICLDisco PaRSEC/DPLASMA) and this branch
changes observable behaviour (the correctness check can fail, `--adddiag` is
applied once, `N % NB != 0` is rejected, the genomics drivers are optional, the
library's SONAME changes), which is more than a patch release and, in a 0.x
line, not a major one. `project(VERSION 0.2.0)` in `CMakeLists.txt` is the
single source of the version; the library is `libhicma_parsec.so.0.2.0` with
SONAME `libhicma_parsec.so.0`.

## 2. Commits on the branch

- `35b1007` Check the BLAS for concurrent calls before the HiCMA tests; harden CPU CI
- `2617921` Documentation for the #if block cleanup
- `7d21454` Resolve the 28 unnamed #if 0 / #if 1 blocks
- `f4e8512` code-review.md: number the macro review as the fifth pass
- `00afd86` Documentation for the macro and data-structure review
- `491b44c` Climate emulator: one name per switch, settable with -D
- `a0a8d8e` Make the local compile-time switches settable with -D
- `d28c2e0` Remove unused macros and parameter fields; one SYNC_TIME_PRINT
- `508466d` Release readiness: GEMM reordering kept for future use
- `8d9777d` Keep GEMM reordering for future use
- `a4dacb1` Documentation for the build-script pass
- `f448fd3` Rewrite the build scripts for the single-configure build
- `7841d07` Kind 3: test the sender conversion through a JDF global
- `2de52d2` Apply the documented default CUDA architectures
- `d6bf679` HIP install, genomics gating and test scripts on Frontier
- `093b313` matrix_sum: sum V of a low-rank tile with its own row count
- `0f4e8f5` Look up the trsm/syrk/gemm locals by name in the timing hooks
- `43b52ec` Set-up fixes found on Frontier: HIP handles, GPU count, kind 3 mode 2
- `6e01e6b` Brace the if/else around the tile-to-LAPACK conversions
- `ec39832` Documentation for the third pass: matrix sum, macros, sanitizers
- `e1a381a` Fix the heap corruption of workload-balance strategy 4
- `c55d6a1` Remove GPU_BUFFER_ONCE: keep the GPU workspace allocated once
- `fcf311d` Compile-time switches: settable with -D, dead ones removed
- `68578bc` Free the storage of pooled workspaces when a taskpool is destroyed
- `f6703d3` One matrix sum for tiles stored in different precisions
- `f63e5a3` Release readiness: fresh-clone check and CI of the second pass
- `ffd71fe` Documentation for the second pass: external use, corner cases, guyot
- `5b80e98` Pass half-precision alpha and beta to FP16-compute GEMMs
- `53edbb3` Accept hex integer options; const-correct GPU evaluate hooks
- `6493db1` Fix kind 3 (DENSE_MP_BAND) on GPUs and its sender-conversion mode
- `22ff7b5` Report the forward error of --check; the residual decides
- `88c67e4` Fix corner cases found by the parameter sweep
- `719bd2c` Record the BLAS/LAPACKE directories in the run-time paths
- `5022f9f` Make the library safe to embed: argument vector and symbol binding
- `9ffb772` Merge the tile/LAPACK conversion JDFs into one
- `944ba0a` Release readiness: CI passes on the final commit
- `87f0c23` Give the installed library and drivers a runtime search path
- `1d9b7ad` Release readiness: results after the CI fixes
- `be50c60` State the FP32 residual bound of the dense adaptive cases; print the residual of a failed check
- `f07913f` CI: the CUDA job needs python3, flex and bison
- `d6cb516` Link one BLAS/LAPACKE stack and give hcore and stars-h their dependencies
- `8f7cb5e` Release documentation for 0.2.0
- `77edd5d` Add the validation matrix script
- `eeaa72d` CI, scripts and repository hygiene
- `5e7ba95` Make the installed package usable, unify the version, register the tests
- `1dc97a3` Small fixes: leaks, NULL checks, file readers, genomics helpers
- `bac0083` Remove dead code
- `4be5157` Share the Cholesky wrapper machinery in potrf_common
- `0b5e817` Remove the sink tasks that could outlive their taskpool
- `08bf1b8` Initialize the parameters once and allocate statistics after device discovery
- `94e3633` Refuse unsupported configurations instead of miscomputing
- `449c906` Fix kernel-level bugs found in the review
- `4eca4c3` Apply --adddiag once for the STARS-H kernels that already carry it
- `0c76e56` Make the correctness check binding and add --check_tol

(This documentation commit follows.) Each commit is self-contained in intent;
the trees that were built and tested are the one after the documentation
commit `8f7cb5e` (section 5.3), the one after the CI fixes (section 5.5), the
one after the third pass (`ec39832`; sections 4, 5.1, 5.2, 5.6 and 5.7) and the
final one, which includes the Frontier commits `6e01e6b` to `d6bf679`
(section 5.8: CTest on CPU and CUDA, kind 3 on 2 ranks; the validation matrix
was not rerun after the third pass).
Intermediate commits were not built individually.

## 3. Validation environment

| Item | Value |
|---|---|
| Machine | 1 node, 2 x Intel Xeon Gold 6140 (72 hardware threads), 1 x NVIDIA A100-PCIE-40GB (compute capability 8.0, driver 580.65.06) |
| Compilers / tools | GCC 11.5.0, CMake 3.30.5, Ninja; CUDA 12.6 (nvcc 12.6.77); ROCm 6.4 `hipcc` (compile only, no AMD GPU) |
| Libraries | Open MPI 5.0.5, Intel MKL 2024.2 (BLAS/LAPACKE), GSL present |
| Submodules | dplasma `3553b25a` (ICLDisco master + stage-in fix), parsec `5cd761f6` (4.0.2411-244), hcore `78028a59`, stars-h `5d98f2c2` |
| CPU tree | `-DCMAKE_BUILD_TYPE=Release -DHICMA_PARSEC_HAVE_CUDA=OFF -DDPLASMA_PRECISIONS="s;d" -DPARSEC_DIST_SHORT_LIMIT=1` |
| CUDA tree | same plus `-DHICMA_PARSEC_HAVE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80` |
| HIP tree | same plus `-DHICMA_PARSEC_HAVE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx90a`, compile only |

The baseline for every "before" number is the same configuration built from
`6e987e3` in a separate worktree (CUDA tree) and from the branch point before
any source change (CPU tree), with the same submodule commits.

## 4. Build results

| Configuration | Result |
|---|---|
| CPU | builds; the only compiler diagnostics are PaRSEC's deprecation notes for `parsec_add2arena`/`parsec_del2arena` (pre-existing, in the wrappers and generated JDF code) |
| CUDA (sm_80) | builds, same diagnostics plus the pre-existing nvcc notes in the `.cu` kernels (the 28 incompatible-pointer warnings of the GPU evaluate hooks are gone) |
| HIP (gfx90a) | builds; not executed (no AMD GPU) |
| CUDA + `-DHICMA_PARSEC_GENOMICS=ON` | compiles (`-DHICMA_PARSEC_GENOMICS=ON` with CUDA, same flags as the CUDA tree); the drivers `testing_syrk`, `testing_KRR`, `testing_KRR_file`, `testing_Kmatrix`, `testing_igemm_summa` are built. Not run: they need the genomics data files. |
| Installed package | `cmake --install` of the CPU and CUDA trees, then `tests/install_consumer` (`find_package(HICMAParsec)`, links `HICMAParsec::hicma_parsec` only) configures, builds and runs `--N 2048 --NB 512 --check 1` (residual 2.69e-16) and the conversion test from both prefixes, with `LD_LIBRARY_PATH` unset |

## 5. Test results

All numbers below come from runs executed on 2026-11-07 and 2026-11-08 against
the final tree of the branch, with `LD_LIBRARY_PATH` unset (logs are in the session's scratch directory; the matrix is
reproducible with `scripts/validation_matrix.sh`).

### 5.1 CTest

| Tree | Command | Result |
|---|---|---|
| CPU | `ctest -L hicma -LE gpu --output-on-failure` | 25 / 25 passed (18 single-rank, 7 two-rank) on methane, and 25 / 25 in an AddressSanitizer build |
| CUDA | `ctest -L hicma --output-on-failure` | 39 / 39 passed on methane (one A100) and on guyot (A100-SXM4): the 25 CPU cases plus `gpu_dense_fp64`, `gpu_dense_adaptive`, `gpu_dense_band2`, `gpu_fp8_static`, `gpu_fp8_sp`, `gpu_runtime_decision`, `gpu_runtime_admem`, `gpu_trmm_adaptive`, `gpu_tlr_mp`, `gpu_tlr_dp_band2`, `gpu_band_dp_sp_hp`, `gpu_band_dp_sp_hp_sender`, `gpu_band_sp_hp`, `gpu_band_hp_fp16acc` |

Every registered case runs with `--check 1`; the mixed-precision cases state
their accepted bound with `--check_tol` (1e-6 for the SP band on CPU, 1e-5
with a half-precision band on GPU, 1e-4 with FP16 accumulation, 1e-7 for the
adaptive and runtime-decision kinds). The two-rank GPU cases share the
single A100 and pass `--mca device_cuda_memory_use 30`.

### 5.2 Validation matrix

`scripts/validation_matrix.sh` runs 31 cases on a CPU build and 50 on a GPU
build (the 31 CPU cases against the GPU-enabled library, plus 19 GPU cases).
Every case exited 0 before and after. The residuals are identical to the
baseline except for the virus-mesh cases, whose residual was the doubled
diagonal shift (review finding E1); the GEMM precision counts of every case
are identical. Cases added during the work have no baseline column: kind 2 with a dense band
(which exposed the sink-task crash), and kind 3 on GPU, kind 3 with sender-side
conversion, kind 4 with a half band and kind 3 with FP16 accumulation (which
would have exposed the bugs of section 5.6). The tables are from the final
tree; every earlier run of the release gave the same residuals and GEMM counts
for the cases they share.

CPU tree:

| Case | Ranks | Residual before | Residual after | Factorization time after (s) |
|---|---|---|---|---|
| `tlr_dp_k2_p2` | 1 | 1.157838e-16 | 1.157838e-16 | 0.01968 |
| `tlr_mp_k1_band1_adapt` | 1 | 1.736223e-08 | 1.736223e-08 | 0.05574 |
| `tlr_mp_k1_band1_noadapt` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05596 |
| `tlr_mp_k1_dense_adapt` | 1 | 6.605555e-09 | 6.605555e-09 | 0.03023 |
| `tlr_mp_k1_dense_admem` | 1 | 6.605555e-09 | 6.605555e-09 | 0.02852 |
| `tlr_kind2_band1` | 1 | not in the baseline matrix | 5.014459e-09 | 0.05086 |
| `tlr_kind2_band2` | 1 | not in the baseline matrix | 2.741820e-09 | 0.03936 |
| `tlr_kind2_band2_np2` | 2 | not in the baseline matrix | 2.741820e-09 | 0.08401 |
| `tlr_kind1_band2_np2` | 2 | not in the baseline matrix | 1.740042e-08 | 0.08878 |
| `tlr_dp_k2_p4_autoband` | 1 | 1.328506e-10 | 1.328506e-10 | 0.10783 |
| `dense_k5_fp64_cpu` | 1 | 1.856962e-16 | 1.856962e-16 | 0.06393 |
| `dense_k5_band2_cpu` | 1 | 3.099008e-09 | 3.099008e-09 | 0.06403 |
| `dense_k5_adapt_cpu` | 1 | 4.028860e-10 | 4.028860e-10 | 0.09451 |
| `band_k3_cpu` | 1 | 5.377707e-08 | 5.377707e-08 | 0.11429 |
| `band_k3_sender_cpu` | 1 | not in the baseline matrix | 5.377707e-08 | 0.11135 |
| `band_k4_cpu` | 1 | 1.692654e-07 | 1.692654e-07 | 0.10130 |
| `runtime_k10_cpu` | 1 | 3.754839e-09 | 3.754839e-09 | 0.12459 |
| `runtime_k10_cpu_adapt1` | 1 | 3.776331e-09 | 3.776331e-09 | 0.11826 |
| `sparse_k6_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05310 |
| `sparse_k7_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05225 |
| `sparse_k6_np2` | 2 | 5.014459e-09 | 5.014459e-09 | 0.09400 |
| `sparse_k6_mesh` | 1 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.46801 |
| `sparse_k7_mesh_np2` | 2 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.58500 |
| `trmm_fp64` | 1 | PASSED | PASSED | NA |
| `trmm_adapt` | 1 | PASSED | PASSED | NA |
| `trmm_adapt_sr` | 1 | PASSED | PASSED | NA |
| `tlr_dp_k2_np2` | 2 | 1.157838e-16 | 1.157838e-16 | 0.06400 |
| `dense_k5_fp64_np2` | 2 | 1.856962e-16 | 1.856962e-16 | 0.12500 |
| `dense_k5_band2_np2` | 2 | 3.099008e-09 | 3.099008e-09 | 0.12800 |
| `runtime_k10_np2` | 2 | 3.754839e-09 | 3.754839e-09 | 0.15600 |
| `trmm_adapt_np2` | 2 | PASSED | PASSED | NA |

CUDA tree (same library built with CUDA; `gpu_*` cases use the A100):

| Case | Ranks | Residual before | Residual after | Factorization time after (s) |
|---|---|---|---|---|
| `tlr_dp_k2_p2` | 1 | 1.157838e-16 | 1.157838e-16 | 0.02263 |
| `tlr_mp_k1_band1_adapt` | 1 | 1.736223e-08 | 1.736223e-08 | 0.05222 |
| `tlr_mp_k1_band1_noadapt` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05298 |
| `tlr_mp_k1_dense_adapt` | 1 | 6.605555e-09 | 6.605555e-09 | 0.02823 |
| `tlr_mp_k1_dense_admem` | 1 | 6.605555e-09 | 6.605555e-09 | 0.02801 |
| `tlr_kind2_band1` | 1 | not in the baseline matrix | 5.014459e-09 | 0.05144 |
| `tlr_kind2_band2` | 1 | not in the baseline matrix | 2.741820e-09 | 0.03985 |
| `tlr_kind2_band2_np2` | 2 | not in the baseline matrix | 2.741820e-09 | 0.07000 |
| `tlr_kind1_band2_np2` | 2 | not in the baseline matrix | 1.740042e-08 | 0.08974 |
| `tlr_dp_k2_p4_autoband` | 1 | 1.328506e-10 | 1.328506e-10 | 0.10921 |
| `dense_k5_fp64_cpu` | 1 | 1.856962e-16 | 1.856962e-16 | 0.06234 |
| `dense_k5_band2_cpu` | 1 | 3.099008e-09 | 3.099008e-09 | 0.06970 |
| `dense_k5_adapt_cpu` | 1 | 4.028860e-10 | 4.028860e-10 | 0.09461 |
| `band_k3_cpu` | 1 | 5.377707e-08 | 5.377707e-08 | 0.11190 |
| `band_k3_sender_cpu` | 1 | not in the baseline matrix | 5.377707e-08 | 0.11130 |
| `band_k4_cpu` | 1 | 1.692654e-07 | 1.692654e-07 | 0.10134 |
| `runtime_k10_cpu` | 1 | 3.754839e-09 | 3.754839e-09 | 0.12299 |
| `runtime_k10_cpu_adapt1` | 1 | 3.776331e-09 | 3.776331e-09 | 0.11988 |
| `sparse_k6_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05437 |
| `sparse_k7_cpu` | 1 | 5.014459e-09 | 5.014459e-09 | 0.05349 |
| `sparse_k6_np2` | 2 | 5.014459e-09 | 5.014459e-09 | 0.09500 |
| `sparse_k6_mesh` | 1 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.47127 |
| `sparse_k7_mesh_np2` | 2 | 9.090908e-02 | 2.989478e-06 **(fixed)** | 0.56000 |
| `trmm_fp64` | 1 | PASSED | PASSED | NA |
| `trmm_adapt` | 1 | PASSED | PASSED | NA |
| `trmm_adapt_sr` | 1 | PASSED | PASSED | NA |
| `tlr_dp_k2_np2` | 2 | 1.157838e-16 | 1.157838e-16 | 0.05600 |
| `dense_k5_fp64_np2` | 2 | 1.856962e-16 | 1.856962e-16 | 0.13470 |
| `dense_k5_band2_np2` | 2 | 3.099008e-09 | 3.099008e-09 | 0.11880 |
| `runtime_k10_np2` | 2 | 3.754839e-09 | 3.754839e-09 | 0.20300 |
| `trmm_adapt_np2` | 2 | PASSED | PASSED | NA |
| `gpu_dense_k5_fp64` | 1 | 3.602581e-16 | 3.602581e-16 | 0.13402 |
| `gpu_dense_k5_adapt` | 1 | 2.176706e-09 | 2.176706e-09 | 0.17915 |
| `gpu_dense_k5_band2` | 1 | 2.804213e-09 | 2.804213e-09 | 0.20102 |
| `gpu_dense_k8_adapt` | 1 | 2.176706e-09 | 2.176706e-09 | 0.18067 |
| `gpu_dense_k9_adapt` | 1 | 3.313729e-07 | 3.313729e-07 | 0.15448 |
| `gpu_runtime_k10` | 1 | 1.041065e-08 | 1.041065e-08 | 0.28366 |
| `gpu_runtime_k10_admem` | 1 | 1.041065e-08 | 1.041065e-08 | 0.24876 |
| `gpu_tlr_kind2_band2` | 1 | not in the baseline matrix | 2.741820e-09 | 0.09078 |
| `gpu_tlr_kind1_band2` | 1 | not in the baseline matrix | 1.725825e-08 | 0.09851 |
| `gpu_tlr_mp_k1` | 1 | 5.551907e-09 | 5.551907e-09 | 0.22843 |
| `gpu_dense_k5_dtconv2` | 1 | 2.176706e-09 | 2.176706e-09 | 0.18273 |
| `gpu_trmm_adapt` | 1 | PASSED | PASSED | NA |
| `gpu_band_k3_hp` | 1 | not in the baseline matrix | 8.462557e-07 | 0.24778 |
| `gpu_band_k3_hp_sender` | 1 | not in the baseline matrix | 8.480405e-07 | 0.26568 |
| `gpu_band_k4_hp` | 1 | not in the baseline matrix | 9.075389e-07 | 0.27196 |
| `gpu_band_k3_fp16acc` | 1 | not in the baseline matrix | 2.337890e-05 | 0.28367 |
| `gpu_dense_k5_fp64_np2` | 2 | 3.602581e-16 | 3.602581e-16 | 0.31146 |
| `gpu_runtime_k10_np2` | 2 | 1.041065e-08 | 1.041065e-08 | 0.48400 |
| `gpu_dense_k8_np2` | 2 | 2.176706e-09 | 2.176706e-09 | 0.39605 |

The kind 8 and 9 cases ran on an A100 (compute capability 8.0), where the FP8
GEMMs fall back to FP16/FP32 (`DENSE_FP8= 0` in their GEMM counts): the FP8
code path is compiled and its dispatch exercised, but no FP8 arithmetic ran in
this validation. The H100 results from the development of the runtime decision
are recorded in the pull-request history, not re-run here.

### 5.3 Performance

Factorization time (`hicma_parsec_cholesky`, without generation and check),
three repetitions each, baseline binary from `6e987e3` and the branch binary,
same node, same GPU, run back to back:

| Case (N=32768, NB=1024, one A100) | Before: min / median / max (s) | After: min / median / max (s) | After/before (median) | GEMM precision counts identical |
|---|---|---|---|---|
| kind 5, FP64 (`--kind_of_problem 2 --gpus 1`) | 1.216 / 1.227 / 1.233 | 1.200 / 1.207 / 1.227 | 0.984 | yes |
| kind 5, static mixed precision (`--adaptive_decision 1`) | 1.155 / 1.176 / 1.180 | 1.119 / 1.140 / 1.147 | 0.969 | yes |
| kind 10, runtime decision (`--adaptive_decision_runtime 1`) | 1.180 / 1.181 / 1.186 | 1.181 / 1.186 / 1.197 | 1.005 | yes |

GEMM precision counts of the "after" runs: kind 5 FP64 `DENSE_DP= 4960 DENSE_SP= 0 DENSE_HP= 0 DENSE_FP8= 0 LOW_RANK_DP= 0 LOW_RANK_SP= 0 Number_of_Datatype_Conversions= 0`;
kind 5 adaptive `DENSE_DP= 2156 DENSE_SP= 2067 DENSE_HP= 737 DENSE_FP8= 0 LOW_RANK_DP= 0 LOW_RANK_SP= 0 Number_of_Datatype_Conversions= 6287`; kind 10 `DENSE_DP= 468 DENSE_SP= 2138 DENSE_HP= 2354 DENSE_FP8= 0 LOW_RANK_DP= 0 LOW_RANK_SP= 0 Number_of_Datatype_Conversions= 6766`.

The differences are within run-to-run noise (3 % or less); no change on the
branch touches the GPU kernels' arithmetic or the task graphs' critical path.
The small-case matrix times (tables above) are likewise unchanged within
noise. The large-case timing was taken before the link changes of section
5.5; after them the matrix times of the GPU cases are unchanged within noise
and those of the CPU cases are identical to the earlier runs. No multi-node or
multi-GPU measurement was possible on this machine.

### 5.4 Not validated

- HIP execution (compile only).
- More than one node, more than one GPU per node, GPUs with compute capability
  8.9 or higher (real FP8 arithmetic).
- The climate emulator driver (`testing_climate_emulator`): compiled in every
  configuration, not run (needs the external data set).
- The genomics drivers: see section 4; not run (data files).
- The Fugaku (`ON_FUGAKU`) build.
- The machine build scripts on their machines: only `install_icl.sh` was run
  as written (methane); the others were configured on methane with their
  modules stubbed, and the Fugaku and Shaheen scripts not at all (section 5.8).
- The intermediate commits individually (only the final tree).

### 5.5 First CI run and the follow-up fixes

The first run of the workflow on GitHub (pull request ecrc/hicma-x#13) failed
in both jobs:

- **CPU job**: every test driver failed to link with undefined `LAPACKE_*`
  references from `libhcore.so`. hcore's own CMake records no link
  dependency, and with Ubuntu's `--as-needed` default nothing else on the
  line provided the symbols (locally MKL reached them through DPLASMA). hcore
  and starsh now link `LAPACKE::LAPACKE`, and `hicma_parsec` links the same
  target instead of a separate BLAS discovery plus a hard-coded MKL block,
  which had loaded two MKL interface layers and two threading layers into one
  process.
- **CUDA job**: configure failed because the container has no Python
  (DPLASMA's precision generator); flex and bison were missing as well. They
  are installed now.
- **Found while validating the fix**: with hcore's new dependency the link
  order put `libgslcblas` (GSL's reference CBLAS, linked by STARS-H) ahead of
  MKL, and DPLASMA's `cblas_dgemm` bound to it: the CPU factorizations ran 40x
  slower (dense FP64, N=4096: 0.065 s to 2.64 s), residuals shifted (FP64
  1.9e-16 to 1.4e-15) and the dense adaptive mixed-precision cases crossed
  their 1e-8 bound. `libgslcblas` is removed from the starsh target (any CBLAS
  satisfies GSL) and hcore is linked before starsh; `LD_DEBUG=bindings` then
  shows `cblas_dgemm` bound to `libmkl_gf_lp64`, and every residual and time
  returned to the baseline values.
- `potrf_mp_dense_adaptive`, `potrf_mp_dense_admem` and `gpu_tlr_mp` carry
  `--check_tol 1e-7`: their residual is 6.6e-9 with one BLAS binding and
  1.1e-8 with another, bounded by the FP32 arithmetic of half of their GEMMs
  rather than by the 1e-8 compression accuracy; the banded sibling case
  already used 1e-7. A failed check now prints its residual and bound.

- **Second run, CPU job**: build, CTest and the regression script passed;
  the installed consumer failed at load time, `libdplasma.so.2: cannot open
  shared object file`. The installed `libhicma_parsec.so` had no runtime
  search path, and with RUNPATH semantics (Ubuntu's default) a library's
  dependencies are searched with the library's own path, not the
  executable's (locally the Spack toolchain emits RPATH, which hid this).
  `CMAKE_INSTALL_RPATH` (`$ORIGIN`, the prefix lib dir) and
  `CMAKE_INSTALL_RPATH_USE_LINK_PATH` are set for `hicma_parsec` and the
  drivers; reproduced and verified locally by linking the consumer with
  `--enable-new-dtags` and running it with only the BLAS directory on
  `LD_LIBRARY_PATH` (CPU and CUDA prefixes). CMake does not record MKL's
  directory, so a BLAS outside the system paths still needs `LD_LIBRARY_PATH`.

Re-validation after these commits: CPU, CUDA and HIP trees build; CPU `ctest`
20 / 20; CUDA-tree `ctest` 30 / 30; install consumers from both prefixes
(residual 2.69e-16); the 75-case matrix identical to the baseline except the
intended mesh change (tables in 5.2). Both CI jobs of pull request
ecrc/hicma-x#13 pass on the final commit (CPU: build, CTest, regression
script, install and consumer; CUDA: compile).

### 5.6 Second pass: external user and corner cases (guyot)

Requested after the first review: merge the tile/LAPACK conversion JDFs (item
1 of `TODOs`), look at the project as an external user, and test on guyot (8 x
A100-SXM4-80GB, 256 cores) looking for corner cases.

**Conversion JDFs.** The eight JDFs are one (`lapack_tile_convert.jdf`, 1433
lines became 280) with `hicma_parsec_lapack_tile_convert` as the entry point
and the named functions as wrappers. A new round-trip test
(`testing_lapack_tile`) passed on 1 to 6 processes on methane and 8 on guyot,
and from the installed package.

**External user.** A fresh HTTPS clone with `--recursive` fetched every
pinned submodule commit (all are public), and BUILD.md followed verbatim built
the default configuration. Then:

- every test failed to start: "libmkl_gf_lp64.so.2: cannot open shared object
  file" (the MKL module sets `LIBRARY_PATH` only, whose directories CMake never
  records in a run-time path; the validation scripts had exported
  `LD_LIBRARY_PATH`). Fixed: the BLAS directories are recorded in the built,
  installed and consumer binaries; every run of this section had
  `LD_LIBRARY_PATH` unset;
- an application linked against the installed package crashed in
  `hicma_parsec_init` with `argc = 0, argv = NULL` and with read-only argument
  strings, and an application function named `parse_arguments` replaced the
  library's (it then died on a division by zero). Fixed (argv handling,
  `-Bsymbolic-functions`);
- the CMake test presets ran DPLASMA's and PaRSEC's suites too; BUILD.md
  pointed at nonexistent `HCORE_DIR`/`STARSH_DIR` variables and had no word on
  calling the library from an application. Fixed.

**Corner-case sweeps.** Each case was classified as passed, refused cleanly
(non-zero exit with a message), crashed or hung (timeout):

| Sweep | Machine | Cases | Passed | Refused as expected | Unexpected |
|---|---|---|---|---|---|
| Sizes, options, kinds, grids, 1 to 6 ranks | methane (CPU) | 61 | 44 | 17 | 0 (7 before the fixes) |
| One process, 8 GPUs | guyot | 33 | 28 | 0 | 5: the kind-3 crash, 4 accuracy-envelope cases |
| 8 ranks x 1 GPU | guyot | 20 | 18 | 0 | 2 accuracy-envelope cases |
| 4 and 3 ranks x 2 GPUs | guyot | 6 | 6 | 0 | 0 |
| Round 2 (kinds 3, 4, envelope cases with bounds), 8 GPUs / 8 ranks / 4 x 2 | guyot | 20 | 19 | 0 | 1: FP16 accumulation, fixed |
| Round 3 (FP16 accumulation), 8 GPUs / 8 ranks | guyot | 4 | 1 | 0 | 3: residual 1.6e-4 against a 1e-4 bound, the FP16-accumulation envelope at N=16384 |

The cases covered every kind, fewer tiles than GPUs or ranks (NT = 1, 2, 3
with 8 GPUs or 8 ranks), more GPUs requested than present, 1 and 2 cores with
8 GPUs, recursive kernels (HNB < NB), repeated runs, GPU memory pressure
(`device_cuda_memory_use` 2 to 5 % with evictions), N = 65536 (kind 10 in
0.91 s, kind 5 in 1.07 s on 8 A100s, one run each), and grids 1 x 8, 8 x 1,
2 x 4, 4 x 2. Results are reproducible across layouts: the same case gives the
same residual on 1 process with 8 GPUs and on 8 ranks.

**Bugs found and fixed** (all present before this release cycle; the
evaluate-hook warnings were 14 then and doubled with the wrapper
consolidation):

| Found by | Problem | Effect before |
|---|---|---|
| guyot | kind 3 with a half-precision band on any GPU: `[type = HALF]` input made PaRSEC reshape the FP32 tile into half its size | `MPI_ERR_TRUNCATE` abort |
| guyot | `--tensor_gemm 0x10`: FP16-compute GEMMs given float alpha/beta (kind 3, low-rank GEMMs of kinds 1, 2, 5, 8 to 10) | wrong results (residual 5e+1) |
| methane | kind 3 `--datatype_convert 1`: rewritten to 2 by the send-decision step, mixed tests in the graph | segmentation fault; after the first fix, wrong results on GPU |
| CPU sweep | sparse kinds with the band covering the matrix (e.g. one tile row) | segmentation fault |
| CPU sweep | invalid `--kind_of_cholesky` | error printed, exit 0 |
| CPU sweep | negative `--N`/`--NB` | silently replaced by defaults |
| embedding | `hicma_parsec_init(0, NULL, ...)`, read-only argv, interposed `parse_arguments` | segmentation faults |
| fresh clone | binaries without the BLAS run-time path | tests could not start |
| check review | the forward-error comparison of `--check` was never part of the verdict (its result was overwritten) | now reported as informational, see `TESTS.md` |
| user | hex option values (`--tensor_gemm 0x10`) | rejected |
| build | 28 incompatible-pointer warnings from the consolidated wrappers' evaluate hooks | warnings |

**Accuracy envelope.** Static mixed precision at N = 16384 gives a residual of
2.3e-8 (above `--fixedacc 1e-8`, inside 1e-7), the half band of kinds 3 and 4
3.6e-6 at N = 16384 (8.5e-7 at N = 8192), FP16 accumulation 1.6e-4 at
N = 16384 (2.3e-5 at N = 8192). These grow with N; the registered cases use
bounds measured at their own sizes.

**Fresh clone, final tree.** The walkthrough was repeated on the pushed
branch: HTTPS clone with `--recursive`, `cmake .. -G Ninja` without options,
`ninja`, CPU CTest 23 / 23, `cmake --install`, and the install consumer
(Cholesky driver and conversion test) from the prefix, all with
`LD_LIBRARY_PATH` unset. Both CI jobs of ecrc/hicma-x#13 pass on the commits
of this section.

**Harness note.** Inside a `srun --overlap` step, an Open MPI 5 singleton (a
one-rank test started without `mpirun`, as CTest does) waits forever on the
step's PMIx server; the guyot runs clear the `SLURM_*`/`PMIX_*` variables. Not
a HiCMA-X issue, but worth knowing when running CTest inside an allocation.

### 5.7 Third pass: matrix sum, compile-time switches, sanitizers (methane, guyot)

Requested: merge the matrix sums for multiple precisions (`TODOs` item 3),
check the design of the macros in `hicma_parsec_internal.h` for correctness
and performance without adding CMake options, keep the code changes minimal,
test on guyot.

**Matrix sum.** `hicma_parsec_matrix_sum` (`matrix_sum.jdf`) replaces the
removed FP64-only and FP32-only sums, whose tile task never stored its partial
sum. It reads each tile in its storage precision (decision map, or the
descriptor's type) including low-rank factors, and returns the same value on
every process and grid. `testing_matrix_sum` passes exactly on 1 to 8
processes; on generated kind-1, 2 and 5 matrices the sums are bitwise equal on
1, 2 and 4 processes.

**Compile-time switches** (`code-review.md`, section 8): switches settable with
`-DNAME=value`; every value compiled and run; dead ones removed; two broken
non-default paths found. `GPU_BUFFER_ONCE=0` crashed (it freed the shared GPU
workspace) and was slower: TRMM N=40960 on 8 A100s 11.88 s against 11.31 s
(3 runs each), so the switch was removed. `WORKLOAD_BALANCE=4` corrupted the
heap; fixed.

**Sanitizers.** An AddressSanitizer build (`-DENABLE_SANITIZER=ON`) ran the 25
CPU CTest cases and the 61-case CPU corner sweep with no report. Its leak
checker found that PaRSEC's `parsec_private_memory_fini` frees no pooled
workspace: every factorization lost its pools (167 MB for kind 2 at N=2700, 24
MB for kind 10). HiCMA-X now frees them itself (`hicma_parsec_memory_pool_fini`);
afterwards nothing leaked is allocated by HiCMA-X code (what remains belongs to
MPI and PaRSEC), and repeated factorizations no longer grow:

| Kind 10, N=8192, CPU | 1 run | 4 runs | 8 runs |
|---|---|---|---|
| Peak memory before | 532 MB | 839 MB | 1238 MB |
| Peak memory after | 525 MB | 568 MB | 605 MB |

Remaining one-off leak: the TRMM driver ends with
`hicma_parsec_cleanup_parsec`, which does not free 3.8 KB of statistics
arrays (the Cholesky path does).

**Validation of the final tree.** methane: CTest 25 / 25 (CPU) and 39 / 39
(CUDA); the 31 CPU and 50 GPU-tree matrix cases identical to the previous run
(residuals and GEMM counts; times within noise); install consumers and the
embedding applications pass. guyot: CTest 39 / 39 (twice); `testing_matrix_sum`
and `testing_lapack_tile` on 8 ranks; after the `GPU_BUFFER_ONCE` removal, the
TRMM (N=40960, 8 GPUs; 8 ranks; memory pressure), kind 5 with adaptive
conversion on 8 GPUs and on 8 ranks (also 3 repeated runs), and kind 10 with 3
repeated runs on 8 ranks all pass. HIP compiles.

### 5.8 Fourth pass: build scripts (methane)

Requested: bring the out-of-date install scripts up to date.

**Scripts.** The machine scripts built and installed STARS-H and HCORE before
the main build, which compiles both from the submodules anyway, and had
drifted further (CHANGELOG lists each case). They now configure the whole tree
once with the site's modules and options and take `[build-dir] [extra CMake
options]`. `scripts/machines/legacy/` is removed. `build_hicma_with_lapack.sh`
builds a pinned reference LAPACK and passes the search hints that STARS-H's
own modules need.

| Run on methane | Result |
|---|---|
| `install_icl.sh`, fresh build (ICL modules, MKL, CUDA 12.6) | builds; CTest 39 / 39 on the A100 |
| `build_hicma_with_lapack.sh`, MKL unloaded | LAPACK 3.12.1 built; HiCMA-X links only it; CTest 25 / 25 (CPU) |
| `build_release.sh --test` | CTest 25 / 25 (CPU) |
| Alps, Frontier (ROCm 6.4), Leonardo, Polaris, TACC, Vista: configure with `module` stubbed | all configure; CMake reports no unused variable |
| `install_icl_debug.sh`, `install_leonardo_no_gsl.sh`, `install_shaheen_slurm.sh` | forward their arguments; a later `-D` wins |

**Found while validating.**

- *CUDA architectures.* A build given only `-DHICMA_PARSEC_HAVE_CUDA=ON`
  compiled its CUDA code for sm_52 alone. `enable_language(CUDA)` cached the
  compiler's default before the documented default `70 80 90 90a` could
  apply; CI and the earlier validation always passed the architecture. Fixed:
  a fresh configure gives `70;80;90;90a` (`80;90;90a` with CUDA 13), and a
  user or `CUDAARCHS` value is kept.
- *Kind 3 guards.* The kind-3 fix of the second pass (`6493db1`) used the C
  macro `SENDER_CONVERT` in 17 JDF dependency guards. The JDF compiler printed
  17 "Fatal Error ... unbound variable" lines per build while the generated
  code was right. A hidden JDF global replaces it. The CTest kind-3 cases pass,
  and so do 2 ranks with `--datatype_convert` 0, 1 and 2, with and without
  the GPU.

**GEMM reordering kept (on request).** `reorder_gemm.jdf` and
`hicma_parsec_reorder_gemm`, removed as dead code in `bac0083`, are back and
still not called. Their launch now compiles, the sort compares the keys it
moves, and the work pool is freed. A scratch driver ran it after the
pre-analysis in 19 configurations (1, 2 and 4 ranks; reorder 1, 2, 4; band
1 and 3; CPU and CUDA trees). In each, all 741 order chains were complete
and sorted, with the same counts on every process count. CTest: 25 / 25
(CPU) and 39 / 39 (CUDA). Wiring it into the Cholesky is future work
(`reorder_gemm.jdf` lists the steps).

### 5.9 Fifth pass: macros and data structures (methane)

Requested: check that the macros and data structures are well designed and
necessary; guyot not used. `code-review.md`, section 9, has the findings.

- **Removed as unused.** 16 datatype identifiers, three tile formats, three
  sparse modes, `DATATYPE_USE`, `EPS_DP`, the generic `str`/`xstr`, 29 HIP
  aliases, the JDF `*_SHAPE` macros, and five struct fields (`HMB`,
  `start_time_kernel_sqr_sum_vec`, `str_problem`, and in the climate
  emulator twelve more). A memory check that could not fire is also gone.
- **Fixed.** `SYNC_TIME_PRINT` was defined twice. The local switches of the
  sources and of the climate emulator could not be set with `-D`. The
  climate header `#undef`d three genomics switches, and its switches had two
  names each. Two climate debug paths did not compile.

| Check | Result |
|---|---|
| CPU, CUDA and HIP trees | build, no new warnings |
| CTest | 25 / 25 (CPU), 39 / 39 (CUDA) |
| Validation matrix against the third pass | 31 CPU and 50 GPU-tree cases: identical exit codes, residuals and GEMM counts |
| Every local switch flipped (CPU and CUDA trees) | CTest 25 / 25 and 39 / 39; flipped prints appear |
| Every climate switch value | compiles in both trees (16 sources) |
| Install consumers, `LD_LIBRARY_PATH` unset | build and pass |

Left as they are, with the reasons in section 9 of the review: the
implicit-capture macros `IS_DENSE`/`IS_ALLOCATE_DP`, the generic object-like
names of the installed header, and the 196-field parameter struct.

**Unnamed `#if 0`/`#if 1` blocks (cleaned on request).** All 28 are gone.
Blocks around live code were unwrapped. Debug dumps, code that could not
compile and superseded alternatives were deleted. The genomics SYRK
alternative is now the switch `GENOMICS_SYRK_ISYRK` (review, section 9,
item 8, lists each block). The compression tasks also lost a scan of every
tile that only fed a dead `1 || flag` test. After the cleanup, the CPU, CUDA
and HIP trees build, both values of `GENOMICS_SYRK_ISYRK` compile, CTest
passes (25 / 25 CPU, 39 / 39 CUDA), and the 31 CPU and 50 GPU-tree matrix
cases are again identical to the third pass.

### 5.10 CI hardening

One CI run (commit `f4e8512`, documentation only) failed 21 of 25 CPU cases
inside DPLASMA's reference Cholesky and the LAPACK compressions. A run on the
same code, image and packages 10 seconds earlier passed, so the cause is the
runner, which the log does not name.

- Every HiCMA CTest case now requires DPLASMA's BLAS thread-safety test as a
  fixture. A BLAS that cannot take concurrent calls reports "Not Run"
  instead of wrong residuals.
- The fixture executable gets the BLAS run-time path. On methane's MKL
  module it could not start, which would have skipped every case.
- The CPU job prints the runner's CPU and OpenBLAS variant and sets
  `OPENBLAS_NUM_THREADS=1`.

On methane, CTest passes 26 / 26 (CPU) and 40 / 40 (CUDA), the fixture
included.

## 6. Release checklist

| Item | Status |
|---|---|
| Version set once (`project(VERSION 0.2.0)`, SOVERSION 0) | done |
| Release notes (`RELEASE.md`), changelog (`CHANGELOG.md`) | done |
| Build and install instructions (`BUILD.md`), consumer example (`tests/install_consumer`) | done |
| Test instructions (`TESTS.md`, CTest labels, `scripts/validation_matrix.sh`) | done |
| API / behaviour changes documented (`RELEASE.md` "Upgrading from 0.1.0") | done |
| Known limitations documented (`RELEASE.md`, `TESTS.md`) | done |
| CI covers correctness (`--check 1` in CTest and `scripts/test_CI.sh`), install, CUDA compile | done; the first GitHub run failed on three environment issues, fixed (section 5.5); both jobs pass on the final commit |
| P0 items of the plan | all done and validated |
| P1 items | done except: per-call status checks in the non-FP8 GPU kernels (P1-6, partial), warning-flags option (P1-11) |
| P2 items | done except P2-6, P2-9, P2-12 (deferred) and the remaining parts of P2-5, P2-7, P2-8 |
| Second pass: conversion merge, external-user walkthrough, guyot sweeps (section 5.6) | done; 11 problems fixed |
| Third pass: matrix sum, macro review, sanitizers (section 5.7) | done; pool leak, `GPU_BUFFER_ONCE`, strategy 4 fixed |
| Fourth pass: build scripts (section 5.8) | done; CUDA default architectures and kind-3 guard messages fixed |
| Fifth pass: macros and data structures (section 5.9) | done; unused names removed, switches settable, two climate debug paths fixed |
| Review by the maintainer of the behaviour changes (binding check, `--adddiag`, `N % NB`, genomics option) | **open** |
| Tag `v0.2.0`, GitHub release, merge to `main` | **open** (authorization required) |

## 7. Unresolved blockers and decisions for the maintainer

1. **Behaviour changes to confirm**: `--check 1` now fails runs it used to
   pass; `--adddiag` results change for problem kinds 0, 1, 5, 6, 7; the
   Cholesky drivers reject `N % NB != 0`; the genomics drivers need
   `-DHICMA_PARSEC_GENOMICS=ON`. Each is documented in `RELEASE.md`.
2. **Mixed-precision residual bounds**: the registered CTest cases use
   `--check_tol` values measured on methane and guyot (1e-7 to 1e-4 by
   precision mix). They grow with N (section 5.6), and other BLAS libraries or
   GPUs may need looser bounds; the cases print the residual.
3. **What `--check` decides**: the residual ||LL^T-A||/||A||. The forward error
   against the dense reference factor is printed but not checked (it was never
   effectively checked; binding it would need a second, looser bound).
4. **CI**: passed after the fixes of section 5.5; rerun on the commits of section 5.6.
5. **Hardware not available here**: AMD GPUs, H100 (FP8), multi-node. A run of
   `scripts/validation_matrix.sh <build> <label> gpu` on those systems before
   tagging is recommended.

## 8. Remaining and future work

Deferred from the plan (none blocks the release): P2-6 (zero-initialized check
matrices, `matrix_uncompress` flows), P2-9 (one JDF for the integer
conversions), P2-12 (statistics communicator hygiene), P3-1 to P3-3, the
remaining parts of P1-6, P1-11, P2-5, P2-7, P2-8. The project's own `TODOs`
backlog (API, Python/Julia bindings, merging the potrf JDFs, hcore removal,
tensor_gemm redesign) was out of scope by request; item 1 (merge the
tile/LAPACK conversions) was done on request in the second pass, and item 10
(external users) partly (section 5.6).

Found in the second pass and left for later:

- Parameters can only be set through an argument vector, and invalid
  parameters terminate the process (`TODOs` items 5 and 10).
- The public headers include STARS-H declarations with generic names (`diff`,
  `cube`, `Gaussian`), which an application cannot reuse in files that include
  `hicma_parsec.h` (third-party submodule, unchanged).
- The FP16-compute branch of the two low-rank GPU GEMMs is fixed but no test
  reaches it: the decisions of the test problems never route an update there.
- Kind 3 has no CPU half-precision path; without GPUs the single-precision
  band is widened to the whole matrix.
