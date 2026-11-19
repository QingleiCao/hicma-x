# Mixed-precision iterative refinement

Branch `qinglei/mixed-precision-ir`. This page covers the solver
(`testing_potrf_ir`), the experiment protocol and the results. The research
question is whether an adaptive-precision Cholesky factor (AP, RAP) needs
enough fewer refinement steps than an FP16 factor to recover FP64 accuracy
faster.

## 1. Usage

```
tests/testing_potrf_ir <HiCMA-X options> [--ir 0|1|2] [refinement options]
```

The factorization options are those of `testing_potrf_tlr`. The refinement
supports kinds 5, 9 and 10 in the dense layout, and refuses `--check`,
`--adaptive_memory`, `--auto_band` and `--sparse`. The refinement options
(`--help` lists them):

| Option | Default | Meaning |
|---|---|---|
| `--ir` | 1 | 0: one solve, no refinement; 1: classic IR; 2: GMRES-IR |
| `--ir_maxit` | 30 (classic), 10 (GMRES-IR) | refinement steps |
| `--gmres_tol` | 1e-4 | GMRES stopping test: ‖M⁻¹(r−Ad)‖₂ ≤ tol·‖M⁻¹r‖₂ |
| `--gmres_maxit` | min(N, 500) | GMRES iterations per refinement step (no restart) |
| `--ir_scale` | 1 | factorize F = μ(D⁻¹AD⁻¹ + c·u·I), D = diag(A)^½, μ = 0.1·65504/(1+c·u); 0: F = A + c·u·diag(A) |
| `--ir_shift` | 0 | the c of the shift; u is the unit roundoff of the factor (see below) |
| `--ir_shift_retry` | 0 | 1: on a POTRF breakdown, refactorize with c = 2, 8, 32, … (at most 8 factorizations) |
| `--ir_seed` | 3872 | seed of the right-hand side |
| `--ir_rhs` | 0 | 0: b = A·x_true with random x_true; 1: random b |
| `--ir_trials` | 1 | right-hand sides solved with one factorization (seeds seed, seed+1, …) |
| `--ir_probe` | 0 | power iterations estimating ‖F − L̂L̂ᵀ‖₂/‖F‖₂ |
| `--ir_condest` | 0 | 1: Hager–Higham estimates ‖A‖₁‖M⁻¹‖₁ and ‖H‖₁‖DM⁻¹D‖₁ (H = D⁻¹AD⁻¹): κ₁(A) and κ₁(H) for an FP64 factor |
| `--ir_stop_slow` | 0 | 1: stop once the contraction is ≥ 0.5 for two steps (timing runs) |
| `--ir_xsave FILE`, `--ir_xref FILE` | | write the first solution / read a reference for the forward error of `--ir_rhs 1` |

The unit roundoff u of the shift is chosen per factor:
- ε (`--fixedacc`) for AP and RAP;
- u₁₆ = 4.88e-4 when the static map has FP16 tiles;
- u₃₂ = 5.96e-8 when it has FP32 tiles;
- none for FP64 factors, which are never shifted.

The six strategies of the study:

| Strategy | Options |
|---|---|
| FP16 + IR | `--kind_of_cholesky 10 --band_dense_dp 1 --band_dense_sp 1 --fp8 0` |
| AP + IR | `--kind_of_cholesky 10 --adaptive_decision 1 --fixedacc ε --fp8 0` |
| RAP + IR | `--adaptive_decision_runtime 1 --adaptive_decision 1 --fixedacc ε --fp8 0` |
| FP16-MAGMA + IR | `--kind_of_cholesky 9 --band_dense_sp 1 --fp8 0` |
| FP32 + IR | `--kind_of_cholesky 10 --band_dense_dp 1 --band_dense_sp NT --fp8 0` |
| FP64 direct | `--kind_of_cholesky 5 --band_dense_dp NT --ir_scale 0` (0 steps) |

Precision of each strategy's operations:
- **FP16 + IR, AP + IR, RAP + IR and FP32 + IR** run on kind 10.
  - Diagonal tiles: POTRF and SYRK in FP64.
  - Off-diagonal tiles: TRSM in FP32 and GEMMs in the precision of the map. FP16 GEMMs take FP16 inputs and accumulate in FP32 on tensor cores.
- **FP16-MAGMA** runs on kind 9. Its diagonal POTRF and SYRK are FP32, as in MAGMA's `dshposv` (Abdelfattah, Tomov, Dongarra, ICCS 2020).

Every run prints one line per right-hand side:

```
IR_summary run= trial= N= NB= kind= strategy= solver= eps= scale= c= attempts=
  tiles_dp= tiles_sp= tiles_hp= gemm_dp= gemm_sp= gemm_hp= status= steps= gmres_its=
  minv= av= berr0= berr= ferr= kappaA= kappaH= probe= t_copyA= t_preproc= t_pre=
  t_factor= t_s2d= t_solve0= t_refine= t_total=
```

Field notes:
- **GEMM counts** are -1 unless `--verbose 2` or higher.
- **`minv` and `av`** count the applications of M⁻¹ and of A. They are the cost of the refinement, comparable between classic IR and GMRES-IR.
- **`berr`** is ‖b − Ax‖∞/(‖A‖∞‖x‖∞ + ‖b‖∞), and **`ferr`** is ‖x − x_true‖∞/‖x_true‖∞.

`scripts/ir_sweep.sh <build> <out dir> <cases file> [launcher]` runs a list of cases and writes their summary lines to `ir_results.tsv`.

## 2. Design

**Data.**
- The factor stays in place in `dcAd`.
- `hicma_parsec_ir_init` keeps an FP64 copy of A, `dcAir`, taken with `dplasma_dlacpy` before any preprocessing.
- Later runs and retries restore `dcAd` from that copy, so the matrix is generated once.

**Preprocessing.** A `parsec_apply` pass over the local tiles of `dcAd` does three things:
- scales and shifts the matrix;
- recomputes the tile norms and the global norm, with the convention of the generation (diagonal tiles counted in full), so the AP decisions see the matrix that is factorized;
- prints the quantiles of the decision ratio ‖A_ij‖_F·NT/‖A‖_F. A tile is FP16 below ε/u₁₆ and FP32 below ε/u₃₂.

**Solve.**
- **Conversion.** After the factorization, `hicma_parsec_convert_s2d` converts the factor's FP32 tiles to FP64 in place. `hicma_parsec_change_nb_elts` then resets the tile spans and drops the FP32-sized GPU copies, which DPLASMA's trsm would otherwise overrun.
- **Products.**
  - M⁻¹v is `dplasma_dpotrs` on the GPUs, with μ and D applied on the host.
  - A·v on one process with GPUs is tril(A)v + tril(A)ᵀv − diag(A)∘v: two `dplasma_dtrmm` on the GPU with `dcAir`, which is pinned once and stays resident.
    - The CPU alternative was too slow: on hexane it took 0.14 s at N = 20480 (a `parsec_apply` pass at the bandwidth of one socket), and DPLASMA's `dsymm` took 0.25 s. On the GPU the whole refinement step is about 0.1 s.
    - Otherwise (several processes, or CPUs only), a `parsec_apply` pass multiplies each rank's tiles of `dcAir` with the replicated vector, reading each tile once, and one `MPI_Allreduce` of N doubles adds the parts. No tile moves.
  - With several processes, `dpotrs` moves tiles of the factor to the ranks that own the vector (grid column 0). The driver warns about it, and the refinement time includes the traffic. All the experiments below use one process.
- **Vectors.** They are replicated on every rank. A right-hand side reaches the GPU solve through a host staging descriptor and `dplasma_dlacpy`, so that PaRSEC sees a new version, and results come back the same way.

**Refinement.**
- **Classic IR.** It stops on LAPACK `dsposv`'s test, ‖r‖∞ ≤ √N·u·‖A‖∞‖x‖∞. Each run ends in one of five states:

  | Status | Condition |
  |---|---|
  | converged | the test is met |
  | stagnated | ‖d‖∞/‖x‖∞ ≤ u before the test is met |
  | slow | ‖dᵢ₊₁‖/‖dᵢ‖ ≥ 0.5 twice; counted, and a stop only with `--ir_stop_slow` |
  | diverged | the ratio exceeds 1 twice, or Inf/NaN |
  | maxit | the step limit is reached first |

- **GMRES-IR.** This is Higham & Pranesh (SISC 2021) Alg. 4.1 with u_r = u: every refinement step solves A·d = r with GMRES preconditioned on the left by M, in FP64, with CGS2 orthogonalization.

Deviations from the literature:
- **Solves in FP64.** MAGMA applies its FP32 factor in FP32. The FP64 solve costs twice the bytes per step for every strategy.
- **FP64 diagonal.** The main FP16 baseline keeps the diagonal in FP64; the FP16-MAGMA line has it in FP32.
- **No extra-precise residuals.** For an FP16-level factor, the u_r = u GMRES-IR variant is guaranteed only up to κ of about 10⁶ to 10⁷.

## 3. Changes outside the solver

- **`dcAd` memory hooks** are no-ops instead of NULL once the matrix is pinned. DPLASMA's wrapped collections call them unchecked, so `dplasma_dpotrs` on `dcAd` crashed.
- **POTRF GPU workspace** is sized for `cusolverDnSpotrf` too. On a V100 at NB = 512, every factorization with an FP32 diagonal (kinds 4 and 9) failed with info = 1. This bug is also on the release branch.
- **Kernel parameters.** `--sigma`, `--beta` and `--nu` are kept when given. An explicit `--beta 0.03` used to be replaced by 0.1.
- **Kind 10 static decisions** (`--adaptive_decision_runtime 0`) have GPU tests now. They match kind 5 bit for bit for the FP16, FP32, FP64 and adaptive maps on the H100 (section 4.1).

## 4. Results

**Summary, against the research question.**

- **Accuracy: yes, adaptive precision cuts the refinement work, where the matrix has tile-norm decay** (sections 4.3 and 4.4).
  - **Classic IR range.** FP16 + IR converges up to κ ≈ 6e5. AP at ε = 1e-4, 1e-6 and 1e-8 extends that to κ ≈ 6e6, 6e7 and 7e8; RAP at the same ε to κ ≈ 6e5, 6e7 and 7e8.
  - **GMRES-IR work.** At κ = 7e8 to 7e9, ε = 1e-6 and 1e-8 reduce the number of preconditioner applications 8 to 40 times against FP16; ε = 1e-4 only 1.3 to 2 times.
  - **RAP against AP.** At ε = 1e-6 and 1e-8, RAP needs the same or fewer steps than AP, while running two to three times more of its GEMMs in FP16. At ε = 1e-4 its factor is five times less accurate than AP's (1.5e-6 against 3.1e-7) and it needs more steps.
  - **No norm decay, no gain.** On a matrix with uniform tile norms, AP collapses to a uniform precision and gains nothing.
- **Time to solution: not yet with this implementation** (section 4.5).
  - **Factorization plus refinement:** on the V100, AP 1e-4, RAP 1e-4 and RAP 1e-6 beat FP16 with IR, and RAP with ε = 1e-6 (2 steps) is fastest: 1.68 s against 2.15 s for FP16 + IR (6 steps) and 3.77 s for FP64 direct at N = 40960. On the H100, FP64 direct is faster than every mixed strategy up to N = 40960; AP at all three ε comes within 0.1 s of it.
  - **What decides the totals now:** about 2 s of CPU conversions and re-sending of the factor that every mixed strategy pays, and about 0.15 to 0.2 s of task overhead per refinement step.


### 4.1 Kind 10 static against kind 5 (H100, `--check`, residual ‖LLᵀ−A‖_F/‖A‖_F)

| N, NB | map | kind 5 | kind 10 static | RAP (runtime, `--adaptive_decision` 1 = 0) |
|---|---|---|---|---|
| 8192, 1024 | FP16 off-diagonal | 7.418293e-06 | 7.418293e-06 | |
| 8192, 1024 | FP32 off-diagonal | 5.798050e-08 | 5.798050e-08 | |
| 8192, 1024 | FP64 | 3.403404e-16 | 3.403404e-16 | |
| 8192, 1024 | AP ε=1e-4 (DP/SP/HP 8/15/13) | 2.198278e-07 | 2.198278e-07 | 1.019553e-06 |
| 8192, 1024 | AP ε=1e-6 (8/24/4) | 5.812900e-08 | 5.812900e-08 | 5.949124e-08 |
| 20480, 512 | FP16 off-diagonal | 1.429746e-05 | 1.429746e-05 | |
| 20480, 512 | FP32 off-diagonal | 8.373699e-08 | 8.373699e-08 | |
| 20480, 512 | FP64 | 1.971026e-16 | 1.971026e-16 | |
| 20480, 512 | AP ε=1e-4 (40/255/525) | 5.302061e-07 | 5.302061e-07 | 1.972458e-06 |
| 20480, 512 | AP ε=1e-6 (40/463/317) | 1.056566e-07 | 1.056566e-07 | 1.139832e-07 |

- **GEMM counts per precision** are identical between kind 5 and kind 10 static in every row.
- **RAP:** `--adaptive_decision` 1 and 0 give the same residual and GEMM counts.
- **Kind 9 (FP16-MAGMA)** gives 7.535071e-06 (N = 8192) and 1.428111e-05 (N = 20480).

### 4.2 Condition numbers of the test matrices (A1a)

The table gives κ₁(A) from Hager–Higham with the FP64 factor, measured on the H100 at N = 20480, NB = 512, with σ = 1. For kinds 2 and 4 the diagonal is constant (σ + noise), so κ₁(D⁻¹AD⁻¹) = κ₁(A). The FP64 forward error follows κ₁·2e-16 throughout.

| kind 2 (2-D sqexp), noise | β = 0.03 | β = 0.1 | β = 0.3 |
|---|---|---|---|
| 1e3 | 1.2 | 2.8 | 18 |
| 1e2 | 3.0 | 25 | 230 |
| 1e1 | 22 | 372 | 2.7e3 |
| 1 | 307 | 4.5e3 | 2.9e4 |
| 1e-1 | 3.9e3 | 5.2e4 | 3.1e5 |
| 1e-2 | 4.6e4 | 5.7e5 | 3.4e6 |
| 1e-3 | 5.1e5 | 6.1e6 | 7.3e7 |
| 1e-4 | 5.5e6 | 6.5e7 | 3.6e8 |
| 1e-5 | 5.9e7 | 6.6e8 | 4.4e9 |
| 1e-6 | 6.2e8 | 6.8e9 | 4.0e10 |
| 1e-7 | 6.6e9 | 6.8e10 | 4.0e11 |
| 1e-8 | 6.6e10 | 7.5e11 | 7.9e12 |

| kind 4 (3-D exp), noise | β = 0.01 | 0.03 | 0.1 | 0.3 | 1 |
|---|---|---|---|---|---|
| 1 | 1.3 | 19 | 925 | 1.0e4 | 2.9e4 |
| 1e-1 | 1.5 | 43 | 3.9e3 | 6.3e4 | 2.4e5 |
| 1e-2 | 1.6 | 50 | 5.7e3 | 1.5e5 | 1.0e6 |
| 1e-4 to 1e-8 | 1.6 | 51 | 6.1e3 | 1.7e5 | 1.6e6 |

How κ behaves:
- **Kind 2:** κ₁ is about λ_max/noise, so the noise sets the decade.
- **Kind 4:** κ₁ saturates once the noise falls below the smallest eigenvalue of the kernel, so it cannot exceed about 1.6e6 here.
- **Kind 0 (randtlr):** κ₁ was about 10 times the target κ₂ = 1 + NT/adddiag, from 108 at κ₂ = 10 to 1.8e6 at κ₂ = 1e5. Generating it takes 370 s at this size, against 4 s for kinds 2 and 4, so it is not the uniform-norm control.
- **The control** is kind 2 with β = 10. That correlation length is much longer than the unit square, so all tiles have nearly the same norm.


### 4.3 Refinement work against κ (A1b)

**Setup.**
- Hardware and size: H100, N = 20480, NB = 512 (NT = 40).
- Right-hand sides: three per matrix, b = A·x_true.
- Preprocessing: scaled, with shift-and-retry on breakdown.
- GMRES: τ = 1e-4, at most 100 iterations per step.
- Counts: the cells give M⁻¹ applications (the initial solve included) for runs that met the LAPACK test.
- Failure labels: "no (slow)" means no convergence in 30 steps (10 for GMRES-IR); "div." means diverged; "max" means the step limit was reached.

**Matrices with norm decay: kind 2, β = 0.1, noise from 1e2 to 1e-7.**

Classic IR:

| κ₁ | FP16 | FP16-MAGMA | FP32 | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 |
|---|---|---|---|---|---|---|---|---|---|
| 2.5e1 | 3 | 3 | 2 | 3 | 2 | 2 | 3 | 2 | 2 |
| 3.7e2 | 3 | 3 | 2 | 3 | 2 | 2 | 3 | 2 | 2 |
| 4.5e3 | 4 | 4 | 2 | 3 | 3 | 2 | 4 | 3 | 2 |
| 5.2e4 | 6 | 6 | 3 | 4 | 3 | 2 | 5 | 3 | 2 |
| 5.7e5 | 20 | 19 | 3 | 6 | 4 | 3 | 7 | 3 | 3 |
| 6.1e6 | no (slow) | no (slow) | 4 | 22 | 5 | 3 | no (div.) | 4 | 3 |
| 6.5e7 | no (slow) | no (slow) | 6 | no (slow) | 12 | 5 | no (slow) | 7 | 4 |
| 6.6e8 | no (slow) | no (slow) | no (div.) | no (slow) | no (slow) | 15 | no (div.) | no (div.) | 6 |
| 6.8e9 | no (slow) | no (slow) | no (slow) | no (slow) | no (slow) | no (slow) | no (slow) | no (slow) | no (div.) |
| 6.8e10 | no (slow) | no (slow) | no (slow) | no (slow) | no (slow) | no (slow) | no (slow) | no (slow) | no (div.) |

GMRES-IR:

| κ₁ | FP16 | FP16-MAGMA | FP32 | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 |
|---|---|---|---|---|---|---|---|---|---|
| 2.5e1 | 5 | 5 | 3 | 5 | 3 | 3 | 5 | 3 | 3 |
| 3.7e2 | 4 | 4 | 3 | 5 | 3 | 3 | 5 | 3 | 3 |
| 4.5e3 | 7 | 7 | 3 | 4 | 5 | 3 | 7 | 5 | 3 |
| 5.2e4 | 9 | 9 | 5 | 7 | 5 | 3 | 7 | 5 | 3 |
| 5.7e5 | 19 | 13 | 4 | 9 | 7 | 6 | 9 | 4 | 5 |
| 6.1e6 | 40 | 40 | 7 | 15 | 7 | 4 | 34 | 7 | 4 |
| 6.5e7 | 113 | 112 | 9 | 50 | 13 | 5 | 59 | 10 | 7 |
| 6.6e8 | 287 | 287 | 22 | 144 | 37 | 15 | 138 | 23 | 10 |
| 6.8e9 | 809 | 809 | 59 | 398 | 67 | 28 | 607 | 62 | 20 |
| 6.8e10 | no (max) | no (max) | 165 | 1011 (1/3) | 279 | 81 | no (max) | 174 | 62 |

**Kind 4 (3-D exp, β = 1 and 0.1).**

Classic IR:

| κ₁ | FP16 | FP16-MAGMA | FP32 | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 |
|---|---|---|---|---|---|---|---|---|---|
| 9.3e2 | 4 | 4 | 2 | 3 | 2 | 2 | 3 | 3 | 2 |
| 2.9e4 | 7 | 7 | 3 | 3 | 3 | 1 | 4 | 3 | 2 |
| 2.4e5 | no (slow) | no (slow) | 3 | 3 | 3 | 1 | 5 | 3 | 2 |
| 1.0e6 | no (div.) | no (div.) | 3 | 3 | 3 | 1 | 6 | 3 | 2 |

GMRES-IR:

| κ₁ | FP16 | FP16-MAGMA | FP32 | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 |
|---|---|---|---|---|---|---|---|---|---|
| 9.3e2 | 7 | 7 | 3 | 5 | 3 | 3 | 6 | 5 | 3 |
| 2.9e4 | 9 | 9 | 5 | 5 | 5 | 1 | 7 | 5 | 3 |
| 2.4e5 | 13 | 13 | 4 | 4 | 4 | 1 | 7 | 4 | 3 |
| 1.0e6 | 24 | 24 | 4 | 4 | 4 | 1 | 9 | 4 | 3 |

**Control with uniform tile norms: kind 2, β = 10.** All tile-norm ratios lie between 1.32 and 1.42, so AP and RAP choose one precision for every off-diagonal tile:
- FP32 at ε = 1e-4 and 1e-6;
- FP64 at ε = 1e-8, where they need 0 steps.

Classic IR:

| κ₁ | FP16 | FP16-MAGMA | FP32 | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 |
|---|---|---|---|---|---|---|---|---|---|
| 4.2e1 | 3 | 3 | 2 | 3 | 2 | 2 | 3 | 2 | 2 |
| 4.1e3 | 5 | 5 | 3 | 3 | 3 | 1 | 3 | 3 | 2 |
| 4.9e5 | no (div.) | no (div.) | 3 | 3 | 3 | 1 | 3 | 3 | 2 |
| 8.4e7 | no (slow) | no (slow) | 9 | 9 | 9 | 1 | 9 | 9 | 2 |
| 8.3e9 | no (slow) | no (slow) | no (slow) | no (slow) | no (slow) | 1 | no (slow) | no (slow) | 3 |

GMRES-IR:

| κ₁ | FP16 | FP16-MAGMA | FP32 | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 |
|---|---|---|---|---|---|---|---|---|---|
| 4.2e1 | 5 | 5 | 3 | 5 | 3 | 3 | 5 | 3 | 3 |
| 4.1e3 | 7 | 7 | 5 | 5 | 5 | 1 | 5 | 5 | 3 |
| 4.9e5 | 29 | 29 | 4 | 4 | 4 | 1 | 4 | 4 | 3 |
| 8.4e7 | 34 | 34 | 12 | 12 | 12 | 1 | 12 | 12 | 4 |
| 8.3e9 | 35 | 35 | 44 | 35 | 42 | 1 | 35 | 42 | 5 |

**Factor error and precision map** for two matrices of κ₁ ≈ 5e5. The error is ‖F−L̂L̂ᵀ‖₂/‖F‖₂ by power iteration; c is the shift that was needed.

| strategy | kind 2 error | kind 2 tiles | control error | control tiles |
|---|---|---|---|---|
| FP16 | 8.7e-6 | 40/0/780 | 1.1e-4 (c=512) | 40/0/780 |
| FP16-MAGMA | 8.8e-6 | 0/40/780 | 1.1e-4 (c=512) | 0/40/780 |
| FP32 | 2.9e-8 | 40/780/0 | 1.2e-8 | 40/780/0 |
| AP 1e-4 | 3.1e-7 | 40/255/525 | 1.2e-8 | 40/780/0 |
| AP 1e-6 | 3.1e-8 | 40/463/317 | 1.2e-8 | 40/780/0 |
| AP 1e-8 | 2.5e-9 | 301/319/200 | 7.9e-17 | 820/0/0 |
| RAP 1e-4 | 1.5e-6 | 40/746/34 | 2.5e-8 | 40/780/0 |
| RAP 1e-6 | 2.8e-8 | 40/751/29 | 1.7e-8 | 40/780/0 |
| RAP 1e-8 | 5.4e-9 | 164/633/23 | 1.3e-10 | 79/741/0 |

**Findings.**
- **FP16 + IR fails beyond κ ≈ 6e5.** Classic IR with an FP16 factor needs 20 M⁻¹ applications at κ₁ = 5.7e5 and stops converging beyond that. This matches the FP16-TC error bound (Blanchard et al.): the factor error is about 1e-5 relative here, so κ·err is above 1 near κ = 1e5 to 1e6.
- **FP16-MAGMA behaves like FP16**, within a step. In this setup, the FP32 diagonal of MAGMA's design does not matter.
- **AP and RAP extend the range of classic IR**, about one decade of κ for every two decades of ε:

  | Factor | Classic IR still converges at κ₁ | M⁻¹ applications there |
  |---|---|---|
  | FP16 | 5.7e5 | 20 |
  | AP ε = 1e-4 | 6.1e6 | 22 |
  | AP ε = 1e-6 | 6.5e7 | 12 |
  | AP ε = 1e-8 | 6.6e8 | 15 |
  | RAP ε = 1e-4 | 5.7e5 | 7 |
  | RAP ε = 1e-6 | 6.5e7 | 7 |
  | RAP ε = 1e-8 | 6.6e8 | 6 |

  RAP at ε = 1e-4 does not extend the range: its factor error is 1.5e-6 against AP's 3.1e-7, and classic IR diverges at κ₁ = 6.1e6. It still needs only 7 M⁻¹ applications at 5.7e5, where FP16 needs 20.
- **With GMRES-IR, ε = 1e-6 and 1e-8 cut the work by large factors; ε = 1e-4 by little.** M⁻¹ applications, with the reduction against FP16 in parentheses:

  | κ₁ | FP16 | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 |
  |---|---|---|---|---|---|---|---|
  | 6.6e8 | 287 | 144 (2.0) | 37 (7.8) | 15 (19) | 138 (2.1) | 23 (12) | 10 (29) |
  | 6.8e9 | 809 | 398 (2.0) | 67 (12) | 28 (29) | 607 (1.3) | 62 (13) | 20 (40) |

  FP32 (all off-diagonal tiles FP32) sits between ε = 1e-6 and 1e-8. AP and RAP reach that accuracy with 30 to 60% of the GEMMs in FP16.
- **The control shows where the gain comes from.** On the uniform-norm matrices, AP has no advantage over a uniform precision. It picks FP32 (or FP64) everywhere and matches FP32's steps, and the gain disappears.
- **Scaling to the FP16 range makes no difference here.** Unscaled FP16 and AP 1e-6 need the same steps as scaled ones. These kernel matrices have a constant diagonal and entries of order one, so μ only shifts the FP16 range.
- **Breakdown.** FP16 factors broke down from κ ≈ 6e6 (kind 2) and κ ≈ 5e5 (control). They were rescued by shifts of c = 8 up to 2048, at a factor error of up to 1.1e-4. The shift is counted: every failed attempt is in the time.
- **Forward error.** The LAPACK test stops on the backward error (√N·u), so converged runs end with forward errors up to about 100 times FP64's. The error is κ·berr in both cases, with a larger berr allowed.

### 4.4 ε sweep (A2) and GMRES tolerance (A3)

**Setup.** Kind 2, β = 0.1, N = 20480, NB = 512, three right-hand sides. The runs used four A100s on guyot; the nine cases lost when that allocation ended were rerun on the hexane V100.

**Columns.**
- AP and RAP give M⁻¹ applications.
- The FP16 GEMMs column is the share of off-diagonal GEMMs run in FP16.
- The error column is the factor error.

At κ₁ = 6.1e6, classic IR:

| ε | AP | AP FP16 GEMMs | AP error | RAP | RAP FP16 GEMMs | RAP error |
|---|---|---|---|---|---|---|
| 1e-2 | no (slow) | 100% | 9.1e-6 | no (slow) | 100% | 9.1e-6 |
| 3e-3 | no (slow) | 100% | 9.3e-6 | no (slow) | 100% | 8.6e-6 |
| 1e-3 | no (slow) | 87% | 4.9e-6 | no (div.) | 100% | 9.2e-6 |
| 3e-4 | no (div.) | 69% | 1.1e-6 | no (div.) | 96% | 4.4e-6 |
| 1e-4 | 22 | 57% | 3.4e-7 | no (div.) | 91% | 2.1e-6 |
| 3e-5 | 9 | 52% | 2.0e-7 | 15 | 85% | 5.3e-7 |
| 1e-5 | 7 | 45% | 1.2e-7 | 7 | 80% | 1.8e-7 |
| 3e-6 | 5 | 35% | 4.9e-8 | 5 | 73% | 5.9e-8 |
| 1e-6 | 5 | 29% | 2.6e-8 | 4 | 67% | 2.8e-8 |
| 1e-7 | 4 | 22% | 1.0e-8 | 4 | 57% | 1.7e-8 |
| 1e-8 | 3 | 16% | 3.1e-9 | 3 | 47% | 3.5e-9 |
| 1e-9 | 3 | 13% | 2.5e-9 | 2 | 37% | 5.7e-10 |
| 1e-10 | 3 | 8% | 1.9e-9 | 2 | 30% | 2.8e-10 |

At κ₁ = 6.1e6, GMRES-IR:

| ε | AP | AP FP16 GEMMs | AP error | RAP | RAP FP16 GEMMs | RAP error |
|---|---|---|---|---|---|---|
| 1e-2 | 69 | 100% | 9.1e-6 | 69 | 100% | 9.1e-6 |
| 3e-3 | 46 | 100% | 9.3e-6 | 45 | 100% | 8.6e-6 |
| 1e-3 | 32 | 87% | 4.9e-6 | 49 | 100% | 9.2e-6 |
| 3e-4 | 22 | 69% | 1.1e-6 | 48 | 96% | 4.4e-6 |
| 1e-4 | 15 | 57% | 3.4e-7 | 34 | 91% | 2.1e-6 |
| 3e-5 | 11 | 52% | 2.0e-7 | 13 | 85% | 5.3e-7 |
| 1e-5 | 10 | 45% | 1.2e-7 | 11 | 80% | 1.8e-7 |
| 3e-6 | 9 | 35% | 4.9e-8 | 9 | 73% | 5.9e-8 |
| 1e-6 | 8 | 29% | 2.6e-8 | 7 | 67% | 2.8e-8 |
| 1e-7 | 7 | 22% | 1.0e-8 | 7 | 57% | 1.7e-8 |
| 1e-8 | 4 | 16% | 3.1e-9 | 4 | 47% | 3.5e-9 |
| 1e-9 | 4 | 13% | 2.5e-9 | 3 | 37% | 5.7e-10 |
| 1e-10 | 4 | 8% | 1.9e-9 | 3 | 30% | 2.8e-10 |

At κ₁ = 6.8e9, GMRES-IR:

| ε | AP | AP FP16 GEMMs | AP error | RAP | RAP FP16 GEMMs | RAP error |
|---|---|---|---|---|---|---|
| 1e-2 | no (max) | 100% | 9.1e-6 | no (max) | 100% | 9.1e-6 |
| 3e-3 | 1011 | 100% | 9.1e-6 | 1011 (2/3) | 100% | 9.1e-6 |
| 1e-3 | no (max) | 87% | 4.4e-6 | 1011 (1/3) | 100% | 8.1e-6 |
| 3e-4 | 607 | 69% | 1.0e-6 | 708 | 96% | 3.7e-6 |
| 1e-4 | 299 | 57% | 3.9e-7 | 708 | 92% | 1.7e-6 |
| 3e-5 | 256 | 52% | 2.4e-7 | 404 | 85% | 4.8e-7 |
| 1e-5 | 259 | 45% | 1.5e-7 | 263 | 79% | 1.8e-7 |
| 3e-6 | 154 | 35% | 5.8e-8 | 153 | 72% | 5.9e-8 |
| 1e-6 | 66 | 29% | 3.9e-8 | 62 | 65% | 3.1e-8 |
| 1e-7 | 44 | 22% | 1.5e-8 | 40 | 53% | 1.8e-8 |
| 1e-8 | 28 | 16% | 7.1e-9 | 22 | 39% | 3.1e-9 |
| 1e-9 | 32 | 13% | 5.2e-9 | 5 | 31% | 4.5e-10 |
| 1e-10 | 23 | 8% | 3.8e-9 | 5 | 25% | 2.1e-10 |

**GMRES tolerance τ:**

| matrix | FP16 τ=1e-2 | 1e-4 | 1e-6 | AP 1e-6 τ=1e-2 | 1e-4 | 1e-6 |
|---|---|---|---|---|---|---|
| κ₁ 6.6e8 | 272 | 286 | 304 | 28 | 37 | 35 |
| κ₁ 6.8e9 | 756 | 809 | 809 | 72 | 66 | 99 |

**Findings.**
- **ε controls the factor error almost linearly**, from 9e-6 (all FP16, ε ≥ 3e-3) down to 2e-9 (ε = 1e-10). The refinement work follows the factor error, as theory predicts: classic IR converges once κ·error is below about 1.
- **RAP gets the same factor error with two to three times more FP16 GEMMs than AP**, because it decides each GEMM from the computed L. Examples:
  - ε = 1e-6: 67% FP16 for RAP against 29% for AP, both at 3e-8.
  - ε = 1e-8: 47% against 16%, at 3e-9.

  That is why its factorization is faster at the same accuracy (section 4.5).
- **At the largest ε, RAP fails more often than AP.** It diverges with classic IR at ε = 1e-3 to 1e-4, where AP converges slowly. In that range a few large tiles of L in FP32 storage dominate the error.
- **τ hardly matters.** 1e-2, 1e-4 and 1e-6 change the work by at most 50%, both for FP16 and for AP. The factor's quality decides it.

### 4.5 Cost per phase (B0)

**Setup.** N = 30720 and 40960, NB = 1024, kind 2 β = 0.1 noise 1e-1 (κ₁ about 1e5). Times are the medians of runs 2 and 3, for the first right-hand side. The ε = 1e-8 rows come from a later allocation. Three cases repeated there (FP64, AP 1e-6 and RAP 1e-6 at N = 40960) matched the first session within 2% in every phase, except RAP's factorization on the H100: it varies from 0.74 to 1.08 s between runs in both sessions.

**Column definitions.**
- **decide+convert+stage** is the pre-analysis: precision decisions, FP64→FP32 conversion of the low-precision tiles (CPU), and the warm-up transfer of A to the GPU.
- **factor to FP64** is convert_s2d (CPU).
- **first solve** includes re-sending the FP64 factor.
- **sum** is everything from the pre-analysis to the end of the refinement. FP64 direct is counted with no refinement.
- **copy A + scale** is the FP64 copy kept for the residuals, plus the preprocessing. It is reported apart.

H100:

| N | strategy | decide+convert+stage | factorize | factor to FP64 | first solve | refine | steps | sum | copy A + scale |
|---|---|---|---|---|---|---|---|---|---|
| 30720 | fp64 | 0.21 | 0.47 | 0.00 | 0.03 | - | 0 | 0.71 | 0.73 |
| 30720 | fp16 | 0.56 | 0.26 | 0.58 | 0.23 | 0.56 | 5 | 2.19 | 1.35 |
| 30720 | fp16_gm | 0.56 | 0.25 | 0.59 | 0.23 | 0.95 | 2 | 2.58 | 1.33 |
| 30720 | magma | 0.58 | 0.26 | 0.61 | 0.23 | 0.59 | 5 | 2.27 | 1.35 |
| 30720 | fp32 | 0.56 | 0.46 | 0.58 | 0.23 | 0.25 | 2 | 2.08 | 1.35 |
| 30720 | ap1e-4 | 0.56 | 0.33 | 0.58 | 0.23 | 0.37 | 3 | 2.07 | 1.33 |
| 30720 | rap1e-4 | 0.56 | 0.37 | 0.57 | 0.23 | 0.36 | 3 | 2.09 | 1.32 |
| 30720 | ap1e-6 | 0.56 | 0.39 | 0.58 | 0.23 | 0.24 | 2 | 2.00 | 1.34 |
| 30720 | rap1e-6 | 0.56 | 0.38 | 0.58 | 0.23 | 0.25 | 2 | 2.00 | 1.33 |
| 30720 | ap1e-8 | 0.44 | 0.44 | 0.37 | 0.24 | 0.16 | 1 | 1.64 | 1.34 |
| 30720 | rap1e-8 | 0.44 | 0.50 | 0.45 | 0.23 | 0.16 | 1 | 1.77 | 1.34 |
| 40960 | fp64 | 0.37 | 1.04 | 0.00 | 0.05 | - | 0 | 1.46 | 1.35 |
| 40960 | fp16 | 0.99 | 0.52 | 1.04 | 0.40 | 0.90 | 5 | 3.86 | 2.39 |
| 40960 | fp16_gm | 1.00 | 0.50 | 1.04 | 0.40 | 1.38 | 2 | 4.33 | 2.35 |
| 40960 | magma | 1.03 | 0.50 | 1.09 | 0.41 | 0.89 | 5 | 3.91 | 2.35 |
| 40960 | fp32 | 1.00 | 1.02 | 1.04 | 0.40 | 0.36 | 2 | 3.82 | 2.36 |
| 40960 | ap1e-4 | 1.00 | 0.66 | 1.04 | 0.40 | 0.50 | 3 | 3.61 | 2.38 |
| 40960 | rap1e-4 | 1.00 | 0.73 | 1.06 | 0.41 | 0.69 | 4 | 3.89 | 2.37 |
| 40960 | ap1e-6 | 1.01 | 0.84 | 1.05 | 0.40 | 0.36 | 2 | 3.66 | 2.38 |
| 40960 | rap1e-6 | 1.02 | 0.91 | 1.06 | 0.41 | 0.37 | 2 | 3.77 | 2.39 |
| 40960 | ap1e-8 | 0.80 | 0.93 | 0.71 | 0.41 | 0.22 | 1 | 3.08 | 2.41 |
| 40960 | rap1e-8 | 0.80 | 0.97 | 0.85 | 0.40 | 0.23 | 1 | 3.25 | 2.39 |

V100:

| N | strategy | decide+convert+stage | factorize | factor to FP64 | first solve | refine | steps | sum | copy A + scale |
|---|---|---|---|---|---|---|---|---|---|
| 30720 | fp64 | 0.31 | 1.62 | 0.00 | 0.03 | - | 0 | 1.95 | 0.76 |
| 30720 | fp16 | 0.61 | 0.44 | 0.59 | 0.33 | 0.68 | 5 | 2.66 | 1.33 |
| 30720 | fp16_gm | 0.62 | 0.44 | 0.61 | 0.32 | 1.02 | 2 | 3.01 | 1.33 |
| 30720 | magma | 0.63 | 0.39 | 0.63 | 0.33 | 0.69 | 5 | 2.67 | 1.34 |
| 30720 | fp32 | 0.61 | 0.94 | 0.59 | 0.33 | 0.33 | 2 | 2.79 | 1.33 |
| 30720 | ap1e-4 | 0.61 | 0.67 | 0.59 | 0.37 | 0.40 | 3 | 2.63 | 1.34 |
| 30720 | rap1e-4 | 0.61 | 0.54 | 0.59 | 0.33 | 0.40 | 3 | 2.48 | 1.34 |
| 30720 | ap1e-6 | 0.61 | 0.81 | 0.59 | 0.33 | 0.28 | 2 | 2.62 | 1.33 |
| 30720 | rap1e-6 | 0.61 | 0.63 | 0.59 | 0.33 | 0.27 | 2 | 2.43 | 1.34 |
| 30720 | ap1e-8 | 0.50 | 1.22 | 0.37 | 0.33 | 0.19 | 1 | 2.62 | 1.36 |
| 30720 | rap1e-8 | 0.50 | 0.95 | 0.45 | 0.33 | 0.17 | 1 | 2.39 | 1.35 |
| 40960 | fp64 | 0.54 | 3.71 | 0.00 | 0.06 | - | 0 | 4.31 | 1.33 |
| 40960 | fp16 | 1.10 | 0.90 | 1.07 | 0.57 | 1.25 | 6 | 4.89 | 2.37 |
| 40960 | fp16_gm | 1.08 | 0.89 | 1.05 | 0.57 | 1.62 | 2 | 5.22 | 2.34 |
| 40960 | magma | 1.11 | 0.82 | 1.12 | 0.56 | 1.29 | 6 | 4.90 | 2.38 |
| 40960 | fp32 | 1.09 | 2.15 | 1.06 | 0.57 | 0.47 | 2 | 5.33 | 2.37 |
| 40960 | ap1e-4 | 1.09 | 1.41 | 1.06 | 0.58 | 0.62 | 3 | 4.76 | 2.37 |
| 40960 | rap1e-4 | 1.10 | 1.06 | 1.07 | 0.57 | 0.90 | 4 | 4.69 | 2.38 |
| 40960 | ap1e-6 | 1.08 | 1.78 | 1.07 | 0.57 | 0.42 | 2 | 4.92 | 2.35 |
| 40960 | rap1e-6 | 1.09 | 1.27 | 1.07 | 0.57 | 0.41 | 2 | 4.41 | 2.35 |
| 40960 | ap1e-8 | 0.90 | 2.66 | 0.70 | 0.57 | 0.26 | 1 | 5.10 | 2.36 |
| 40960 | rap1e-8 | 0.90 | 1.90 | 0.84 | 0.57 | 0.40 | 2 | 4.61 | 2.37 |

**Findings.**
- **Without overheads, adaptive precision would win on the V100 but not on the H100.**

  Seconds at N = 40960, with the refinement steps in parentheses (FP64 direct: factorize and one solve):

  | GPU | FP64 direct | FP16+IR | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 |
  |---|---|---|---|---|---|---|---|---|
  | V100, FP64 at 7.8 Tflop/s | 3.77 | 2.15 (6) | 2.04 (3) | 2.20 (2) | 2.92 (1) | 1.95 (4) | **1.68** (2) | 2.30 (2) |
  | H100, FP64 tensor cores | **1.09** | 1.42 (5) | 1.17 (3) | 1.19 (2) | 1.16 (1) | 1.42 (4) | 1.28 (2) | 1.20 (1) |

  - **V100:** AP 1e-4, RAP 1e-4 and RAP 1e-6 beat FP16+IR; AP 1e-6 and both ε = 1e-8 factors do not. RAP 1e-6 is fastest: its two steps beat FP16's six, and the gap to FP64 is 2.2 times. At this κ (about 1e5) ε = 1e-8 saves at most one step against 1e-6, but its larger FP64 and FP32 share (AP keeps about 300 of the 820 tiles in FP64) costs more on the V100 than the step saves.
  - **H100:** at these sizes the FP64 factorization is only twice as slow as the FP16 one. With no conversions, each refinement step costs about as much as 20% of the FP64 factorization. AP at all three ε comes within 0.1 s of FP64 direct, but no mixed factor beats it up to N = 40960.
- **The overheads currently decide the totals.** Every mixed strategy pays them at N = 40960 on top of the table above; ε = 1e-8 pays less because it converts fewer tiles:

  | Overhead at N = 40960 | FP16, FP32, ε = 1e-4 and 1e-6 | ε = 1e-8 |
  |---|---|---|
  | Pre-analysis conversion | 0.6 s | 0.4 s |
  | Converting the factor back to FP64 (convert_s2d) | 1.05 s | 0.7 to 0.85 s |
  | Re-sending the FP64 factor | 0.35 to 0.5 s | 0.35 to 0.5 s |
  | Total | about 2.1 s | about 1.6 s |

  As a result, no mixed strategy beats FP64 direct on either GPU at these sizes. The fastest mixed one is RAP 1e-6 on the V100 (4.41 s against 4.31 s) and AP 1e-8 on the H100 (3.08 s against 1.46 s).
- **Each refinement step costs 0.15 to 0.2 s at N = 40960** (one A·v, one M⁻¹). With one right-hand side, the time goes into task and launch overhead of the tiled kernels, not into bandwidth.
- **What would close the gap** (plan stage C):
  - a triangular solve that reads each tile in its storage precision on the GPU, which removes the conversion back and the re-send;
  - fewer, larger tasks per refinement step.
