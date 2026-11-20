# Mixed-precision iterative refinement

Branch `qinglei/mixed-precision-ir`. This page covers the solver
(`testing_potrf_ir`, sections 1 to 3) and the results. The research question
is whether an adaptive-precision Cholesky factor (AP, RAP) needs enough fewer
refinement iterations than an FP16 factor to recover FP64 accuracy faster.
Section 4 compares the accuracy of the factorizations: how many iterations
each needs, matrix by matrix, and where it stops converging. Section 5 has
the other measurements, including time to solution.

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
| `--ir_maxit` | 30 (classic), 10 (GMRES-IR) | refinement iterations |
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
| `--ir_stop_slow` | 0 | 1: stop once the contraction is ≥ 0.5 for two iterations (timing runs) |
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
| FP64 direct | `--kind_of_cholesky 5 --band_dense_dp NT --ir_scale 0` (0 iterations) |

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
- **Kind 10 static decisions** (`--adaptive_decision_runtime 0`) have GPU tests now. They match kind 5 bit for bit for the FP16, FP32, FP64 and adaptive maps on the H100 (section 5.1).

## 4. Accuracy comparison

**Question.** How many refinement iterations does each factorization need to reach FP64 accuracy, and from which condition number does it stop converging?

Nine factorizations are compared on 58 matrices, each with classic IR and with GMRES-IR. The matrices come from five kernels in ten families, with κ₁(A) from 25 to 7e10:
- Gaussian-process covariances from spatial statistics: squared exponential (2-D and 3-D), exponential (3-D) and Matérn (2-D);
- a Gaussian radial-basis-function matrix for mesh deformation.

FP64 direct solves every matrix with 0 iterations; it is the accuracy reference.

**Main findings.**

1. **The factor's accuracy decides classic IR.** 521 classic-IR runs converged or failed on all three right-hand sides. Sorting them by κ₁(A) × factor error:
   - below 3.7, all 327 converged;
   - above 11.1, all 166 failed;
   - between 3.7 and 11.1, 19 converged and 9 failed, depending on the matrix.

   Each factorization's reach follows from its factor error (section 4.8).
2. **On every kernel whose tile norms decay, AP extends classic IR beyond FP16.**
   - FP16 converges up to κ₁ between 3.9e3 (S) and 5.7e5 (D).
   - Each step from FP16 to AP 1e-4, 1e-6 and 1e-8 extends the range, typically by one to two decades, until it covers every matrix of the family:

     | Factorization | Converges up to κ₁ |
     |---|---|
     | AP 1e-4 | 2.4e5 to 1.6e7 |
     | AP 1e-6 | 5.5e6 to 3.6e8 |
     | AP 1e-8 | 5.9e7 (S), 3.3e8 (T) and 6.6e8 (D); every matrix of L, Mb, Mc and R (up to κ₁ = 1.4e10) |

   - Family Ma never exceeds κ₁ = 1.1e5, and every factorization converges on it.
3. **Only ε = 1e-8 goes beyond FP32.**
   - With classic IR, FP32 converges as far as AP 1e-6 on five of the seven decaying families, and one decade further on S and T.
   - AP 1e-8 goes one decade or more beyond FP32 on D, L, Mc and R, and matches it on S and T.
   - On D, L, Mb, Mc and R, AP 1e-6 converges as far as FP32 while running up to 29% of its GEMMs in FP16 (section 4.8).
4. **With GMRES-IR almost every factorization converges, and the difference moves into the GMRES iterations.**
   - FP16 and FP16-MAGMA fail on 5 of the 58 matrices (D10, S6, Mc6, R5 and R6).
   - AP 1e-4 fails on R6, and on two of the three right-hand sides of D10. RAP 1e-4 fails on D10.
   - Every other factorization converges on all 58.
   - At the hardest matrix of each decaying family where all factorizations converge, FP16 needs 108 to 800 GMRES iterations. AP 1e-6 needs 8 to 23 times fewer, and AP 1e-8 30 to more than 100 times fewer (section 4.5).
5. **RAP against AP.**
   - **At ε = 1e-4**, RAP's factor error is about 2e-6 on every family except U. That is up to 1,200 times AP's error (E3). With classic IR, RAP 1e-4 never converges further than AP 1e-4, and it stops earlier on D, Mb and Mc.
   - **At ε = 1e-6**, the two errors are within about a factor of two on eight of the ten families.
   - **At ε = 1e-8**, the result depends on the family:
     - On S and T, RAP is more accurate (six times on S) and converges a decade further.
     - On Mc and R, AP converges further.
   - At every ε, RAP runs at least as large a share of its GEMMs in FP16 as AP, usually a much larger one.
6. **Without tile-norm decay there is nothing to adapt.** On families E (except E1) and U, AP puts every off-diagonal tile in one precision and needs the same iterations as that uniform precision. The only differences are on U5 with GMRES-IR, where the breakdown shifts differ.
7. **FP16-MAGMA** (an FP32 instead of an FP64 diagonal) has the same outcome as FP16 on all 58 matrices.
   - Classic IR: within one iteration of FP16.
   - GMRES-IR: the range of GMRES iterations over the three right-hand sides is within 10% of FP16's, except on T4 (77–80 against 96).
8. **No run failed outright, but low-precision Cholesky breaks down at large κ.**
   - FP16 first broke down between κ₁ = 2.4e5 (R2) and 1.6e7 (Mb4), depending on the family.
   - AP 1e-8 broke down only on D9–D10, S5–S6 and R6.
   - A diagonal shift rescued every breakdown. The shifted cases are marked †.

### 4.1 Setup

- **Hardware and size.** N = 20480, NB = 512: 40 × 40 tiles, so 820 tiles in the lower triangle, 40 of them diagonal.
  - Families D, E, U, S, L, Ma and Mb, and matrix R6, ran on the hexane H100. Families T and Mc, and R1–R5, ran on the hexane V100.
  - The GPU does not change the counts. Twenty D cases were repeated on the V100. Every run that converged needed the same number of iterations on both GPUs. The only differences were in how three failing runs failed: diverged instead of slow, or at a different iteration.
- **Right-hand sides.** Three per matrix, b = A·x_true with a random x_true, all solved with one factorization.
- **Preprocessing.** Every factorization except FP64 direct uses the scaling of Higham and Pranesh, F = μ·D⁻¹AD⁻¹ (section 1). A diagonal shift is added only if Cholesky breaks down (section 4.8).
- **Stopping test.** LAPACK `dsposv`'s ‖b − Ax‖∞ ≤ √N·u·‖A‖∞‖x‖∞, with u = 1.1e-16. A run that meets it has an FP64-level backward error: between 6e-18 and 2e-14 here.
- **Limits.**
  - Classic IR: 30 iterations.
  - GMRES-IR: 10 refinement iterations, each solving the correction equation with at most 100 GMRES iterations to a relative tolerance τ = 1e-4.
- **Tables.** Generated from the `ir_results.tsv` files that `scripts/ir_sweep.sh` writes.

### 4.2 Factorizations compared

All run on kind 10, except FP16-MAGMA (kind 9) and FP64 (kind 5). On kind 10, the diagonal tiles (POTRF, SYRK) are always FP64, and TRSM is FP32 on every off-diagonal tile; only the GEMMs change precision.

| Factorization | Off-diagonal GEMMs | How the precision is chosen |
|---|---|---|
| FP16 | all FP16 (tensor cores, FP32 accumulation) | fixed |
| FP16-MAGMA | all FP16, with an FP32 diagonal, as MAGMA's `dshposv` | fixed |
| AP ε (ε = 1e-4, 1e-6, 1e-8) | FP16, FP32 or FP64 per tile | before the factorization, from the tile norms of A: a tile is FP16 if ‖A_ij‖·NT/‖A‖ < ε/u₁₆, FP32 if below ε/u₃₂, FP64 otherwise. Every GEMM that updates a tile runs in that tile's precision |
| RAP ε | FP16, FP32 or FP64 per GEMM | during the factorization: the same test applied to each GEMM's ‖L_ik‖·‖L_jk‖ |
| FP32 | all FP32 | fixed |
| FP64 direct | all FP64 | fixed; one solve, no refinement |

The precision AP and RAP choose depends on the matrix; section 4.8 gives the FP16 share for each family.

### 4.3 Test matrices

**How the matrices are built.**
- **Points.** The covariance kernels are evaluated on N points in the unit square (2-D) or cube (3-D). The RBF matrix uses N points on the surface of the unit cube, in Morton order, with the basis exp(−(r/0.05)²). All are generated by STARS-H.
- **Condition number.** Within a family, only the term added to the diagonal changes: the noise (nugget) for the covariance kernels, a regularization for the RBF matrix.
- **Scaling.** Every diagonal entry of a matrix is the same, so the scaling multiplies the matrix by a scalar, and κ of the scaled matrix equals κ₁(A).
- **κ₁(A)** is the Hager–Higham estimate from the FP64 factor (section 5.2).

**Decay.** The correlation length β sets how fast the tile norms decay away from the diagonal: the smaller β, the faster. The last column of the table measures the decay as the number of off-diagonal tiles AP 1e-4 stores in FP16.

| Family | Kernel | Parameters | Matrices | Noise (R: regularization) | κ₁(A) | Off-diagonal tiles in FP16, AP 1e-4 (of 780) |
|---|---|---|---|---|---|---|
| D | 2-D squared exponential | β = 0.1 | D1–D10 | 1e2 to 1e-7 | 2.5e1 to 6.8e10 | 525–653 |
| E | 3-D exponential | β = 0.1 (E1), 1 (E2–E4) | E1–E4 | 1 to 1e-2 | 9.3e2 to 1.0e6 | 0–390 |
| U | 2-D squared exponential | β = 10 | U1–U5 | 1e3 to 1e-5 | 4.2e1 to 8.3e9 | 0–780 |
| S | 2-D squared exponential | β = 0.03 | S1–S6 | 1e-1 to 1e-7 | 3.9e3 to 6.6e9 | 652 |
| L | 2-D squared exponential | β = 0.3 | L1–L6 | 1 to 1e-5 | 2.9e4 to 4.4e9 | 138 |
| T | 3-D squared exponential | β = 0.1 | T1–T6 | 1e-1 to 1e-6 | 2.0e4 to 2.5e9 | 472 |
| Ma | 2-D Matérn | ν = 0.5, β = 0.1 | Ma1–Ma4 | 1 to 1e-4 | 2.3e3 to 1.1e5 | 447 |
| Mb | 2-D Matérn | ν = 1.5, β = 0.1 | Mb1–Mb5 | 1e-1 to 1e-6 | 2.9e4 to 9.7e7 | 498 |
| Mc | 2-D Matérn | ν = 2.5, β = 0.1 | Mc1–Mc6 | 1e-2 to 1e-7 | 3.4e5 to 1.4e10 | 508 |
| R | Gaussian RBF on a cube surface | radius 0.05 | R1–R6 | 1 to 1e-6 | 1.3e4 to 3.6e9 | 198 |

- **S, D, L and U** use the same kernel with β increasing, so the decay weakens from S to U. At ε = 1e-4, AP stores these numbers of off-diagonal tiles in FP16:
  - S: 652;
  - D: 525–653, depending on the noise;
  - L: 138;
  - U: none, except on U1.

  The tile count saturates near 650; the decay on S is still much stronger than on D.
- **U is the control.** Its correlation length is much longer than the unit square, so all tiles have nearly the same norm.
- **Some families saturate.** κ cannot exceed the kernel's own conditioning, so it stops growing once the noise falls below the kernel's smallest eigenvalue. Ma stops at 1.1e5, E at 1.6e6 and Mb at 1.1e8 (section 5.2).

### 4.4 How to read the tables

| Cell | Meaning |
|---|---|
| a number, e.g. `4` | converged after that many refinement iterations; `0` means the first solve already met the test |
| a range, e.g. `17–19` | the three right-hand sides needed different numbers of iterations |
| `2 / 25` (GMRES-IR) | 2 refinement iterations with 25 GMRES iterations in total. Each GMRES iteration costs one solve with the factor and one product with A |
| `(1 of 3)` | only that many of the three right-hand sides converged |
| `NC` | not converged: the iteration limit was reached (in a few cases one right-hand side diverged instead) |
| `DIV` | diverged on all three right-hand sides: the correction grew in two consecutive iterations, and the run stopped |
| `†` | Cholesky broke down on the unshifted matrix; the cell is for a factor of the shifted matrix (section 4.8) |

### 4.5 Summary over all families

**Classic IR: the largest κ₁ up to which a factorization converges on every matrix of the family.** `all` means it converged on every matrix of the family. No factorization converged on a harder matrix after failing on an easier one.

| Family | κ₁ range | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|
| D | 2.5e1–6.8e10 | 5.7e5 | 5.7e5 | 6.1e6 | 6.5e7 | 6.6e8 | 5.7e5 | 6.5e7 | 6.6e8 | 6.5e7 |
| E | 9.3e2–1.0e6 | 2.9e4 | 2.9e4 | all | all | all | all | all | all | all |
| U | 4.2e1–8.3e9 | 4.1e3 | 4.1e3 | 8.4e7 | 8.4e7 | all | 8.4e7 | 8.4e7 | all | 8.4e7 |
| S | 3.9e3–6.6e9 | 3.9e3 | 3.9e3 | 5.1e5 | 5.5e6 | 5.9e7 | 5.1e5 | 5.5e6 | 6.2e8 | 5.9e7 |
| L | 2.9e4–4.4e9 | 2.9e4 | 2.9e4 | 3.4e6 | 3.6e8 | all | 3.4e6 | 3.6e8 | all | 3.6e8 |
| T | 2.0e4–2.5e9 | 2.1e5 | 2.1e5 | 2.5e6 | 3.1e7 | 3.3e8 | 2.5e6 | 3.1e7 | all | 3.3e8 |
| Ma | 2.3e3–1.1e5 | all | all | all | all | all | all | all | all | all |
| Mb | 2.9e4–9.7e7 | 2.7e5 | 2.7e5 | 1.6e7 | all | all | 2.2e6 | all | all | all |
| Mc | 3.4e5–1.4e10 | 3.4e5 | 3.4e5 | 3.3e6 | 2.5e8 | all | 3.4e5 | 2.9e7 | 2.0e9 | 2.5e8 |
| R | 1.3e4–3.6e9 | 1.3e4 | 1.3e4 | 2.4e5 | 2.2e7 | all | 2.4e5 | 2.2e7 | 3.6e8 | 2.2e7 |

- **FP16 and FP16-MAGMA stop first on every family**, between κ₁ = 3.9e3 (S) and 5.7e5 (D), except Ma, where every factorization converges.
- **On the seven families with decay (D, S, L, T, Mb, Mc, R), each ε step extends the range.**
  - FP16 to AP 1e-4: about one decade on D, T, Mc and R; about two on S, L and Mb.
  - AP 1e-4 to AP 1e-6: one to two decades; on Mb, AP 1e-6 already converges on every matrix.
  - AP 1e-6 to AP 1e-8: one decade or more on D, S, L, T, Mc and R. On L, Mc and R, AP 1e-8 converges on every matrix (on R that is at least 2.2 decades further).
- **FP32 converges as far as AP 1e-6, or a decade further (S, T).** Only ε = 1e-8 goes beyond FP32.
- **On E2–E4 and U2–U5, AP 1e-4 and 1e-6 equal FP32**, because there they put every off-diagonal tile in FP32 (section 4.6). AP 1e-8 is an FP64 factorization on E2–E4 and U2–U5.

**GMRES-IR.** Who fails:
- FP16 and FP16-MAGMA fail on 5 of the 58 matrices: D10, S6, Mc6, R5 and R6.
- AP 1e-4 fails on R6, and converges for only one of the three right-hand sides of D10.
- RAP 1e-4 fails on D10.
- AP and RAP at ε = 1e-6 and 1e-8, and FP32, converge on all 58.

The table gives the GMRES iterations at the hardest matrix of each family where every factorization converges:

| Family | Matrix | κ₁ | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D | D9 | 6.8e9 | 800 | 800 | 393–394 | 63–97 | 25–26 | 600–700 | 59–60 | 17–18 | 55–56 |
| E | E4 | 1.0e6 | 19–21 | 19–21 | 2 | 2 | 0 | 6 | 2 | 1 | 2 |
| U | U5 | 8.3e9 | 30–32 | 30–32 | 31 | 37–38 | 0 | 31 | 37–38 | 3 | 40–41 |
| S | S5 | 6.2e8 | 500 | 400–500 | 219–224 | 52 | 15 | 218–221 | 19 | 8 | 32–34 |
| L | L6 | 4.4e9 | 232–235 | 233–235 | 168–170 | 23 | 4 | 171–176 | 23 | 10 | 23 |
| T | T6 | 2.5e9 | 800 | 800–900 | 392 | 34–35 | 18–20 | 392 | 49 | 10 | 21 |
| Ma | Ma4 | 1.1e5 | 5 | 5 | 4 | 2 | 1 | 4 | 2 | 1 | 2 |
| Mb | Mb5 | 9.7e7 | 108 | 108 | 15 | 6 | 2 | 33–52 | 6 | 2 | 6 |
| Mc | Mc5 | 2.0e9 | 600–700 | 600 | 183–184 | 36 | 4–7 | 288–289 | 28 | 8 | 38–39 |
| R | R4 | 2.2e7 | 209–210 | 209–210 | 58–60 | 10 | 2 | 58 | 8 | 2 | 6 |

At that matrix, on the decaying families, the GMRES iterations compare with FP16's as follows:

| Factorization | GMRES iterations, relative to FP16 |
|---|---|
| AP 1e-4 | 1.4 to 7 times fewer |
| AP 1e-6 | 8 to 23 times fewer |
| AP 1e-8 | 30 to more than 100 times fewer |

On E, Ma and U every factorization needs few GMRES iterations, so these matrices do not separate the methods.

### 4.6 Classic IR

**Family D (2-D squared exponential, β = 0.1)**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 1e2 | 2.5e1 | 2 | 2 | 2 | 1 | 1 | 2 | 1 | 1 | 1 |
| D2 | 1e1 | 3.7e2 | 2 | 2 | 2 | 1 | 1 | 2 | 1 | 1 | 1 |
| D3 | 1 | 4.5e3 | 3 | 3 | 2 | 2 | 1 | 3 | 2 | 1 | 1 |
| D4 | 1e-1 | 5.2e4 | 5 | 5 | 3 | 2 | 1 | 3–4 | 2 | 1 | 2 |
| D5 | 1e-2 | 5.7e5 | 17–19 | 16–18 | 5 | 2–3 | 2 | 6–7 | 2 | 2 | 2 |
| D6 | 1e-3 | 6.1e6 | NC† | NC† | 21–23 | 4 | 2 | DIV | 3 | 2 | 3 |
| D7 | 1e-4 | 6.5e7 | NC† | NC† | NC† | 11 | 3–4 | NC† | 6 | 3 | 5 |
| D8 | 1e-5 | 6.6e8 | NC† | NC† | NC† | NC† | 13–15 | DIV† | DIV† | 4–5 | DIV |
| D9 | 1e-6 | 6.8e9 | NC† | NC† | NC† | NC† | NC† | NC† | NC† | DIV† | NC† |
| D10 | 1e-7 | 6.8e10 | NC† | NC† | NC† | NC† | NC† | NC† | NC† | DIV† | NC† |

- **FP16 and FP16-MAGMA** need 2–5 iterations up to κ₁ = 5.2e4 and 16–19 at 5.7e5, and do not converge beyond that.
- **AP** converges one decade further for each two decades of ε: to 6.1e6 with ε = 1e-4 (21–23 iterations), to 6.5e7 with 1e-6 (11), to 6.6e8 with 1e-8 (13–15). Where FP16 needs 17–19 iterations (D5), AP needs 5, 2–3 and 2.
- **RAP** at ε = 1e-6 and 1e-8 converges as far as AP, with fewer iterations near the limit: 6 against 11 at D7 and 4–5 against 13–15 at D8. RAP 1e-4 stops at D5, like FP16, though with 6–7 iterations against FP16's 17–19.
- **FP32** converges up to 6.5e7 (D7), as far as AP 1e-6, with 5 iterations there against AP 1e-6's 11.
- **No factorization** converges with classic IR on D9 or D10 (κ₁ ≥ 6.8e9).

**Family E (3-D exponential)**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 1 | 9.3e2 | 3 | 3 | 2 | 1 | 1 | 2 | 2 | 1 | 1 |
| E2 | 1 | 2.9e4 | 6 | 6 | 2 | 2 | 0 | 3 | 2 | 1 | 2 |
| E3 | 1e-1 | 2.4e5 | NC | NC | 2 | 2 | 0 | 4 | 2 | 1 | 2 |
| E4 | 1e-2 | 1.0e6 | DIV† | DIV† | 2 | 2 | 0 | 5 | 2 | 1 | 2 |

- **FP16 and FP16-MAGMA** do not converge from E3 (κ₁ = 2.4e5).
- **Every other factorization** converges on all four matrices in at most 5 iterations. On E2–E4, AP 1e-4 and 1e-6 put every off-diagonal tile in FP32 and match FP32; AP 1e-8 puts every tile in FP64 and needs 0 iterations.
- **RAP 1e-4** needs 3–5 iterations where AP 1e-4 needs 2. Its runtime test still runs most GEMMs in FP16 (section 4.8).

**Family U (2-D squared exponential, β = 10: the control, uniform tile norms)**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U1 | 1e3 | 4.2e1 | 2 | 2 | 2 | 1 | 1 | 2 | 1 | 1 | 1 |
| U2 | 1e1 | 4.1e3 | 4 | 4 | 2 | 2 | 0 | 2 | 2 | 1 | 2 |
| U3 | 1e-1 | 4.9e5 | DIV† | DIV† | 2 | 2 | 0 | 2 | 2 | 1 | 2 |
| U4 | 1e-3 | 8.4e7 | NC† | NC† | 8 | 8 | 0 | 8 | 8 | 1 | 8 |
| U5 | 1e-5 | 8.3e9 | NC† | NC† | NC† | NC† | 0 | NC† | NC† | 2 | NC† |

- **FP16 and FP16-MAGMA** fail from U3 (κ₁ = 4.9e5).
- **AP at ε = 1e-4 and 1e-6** puts every off-diagonal tile in FP32 on U2–U5 and gives exactly FP32's counts. RAP stores the same FP32 tiles but runs 71–92% of its GEMMs in FP16 there, and also gives FP32's counts. On U1 at ε = 1e-4 both choose FP16 and need FP16's 2 iterations. Adapting gains nothing here.
- **AP 1e-8** is an FP64 factorization on U2–U5 (0 iterations). RAP 1e-8 keeps most tiles in FP32, runs up to 91% of its GEMMs in FP16 (U5), and still converges everywhere, in at most 2 iterations.

**Family S (2-D squared exponential, β = 0.03: the strongest decay)**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 1e-1 | 3.9e3 | 4 | 4 | 3 | 2 | 2 | 3 | 2 | 1 | 2 |
| S2 | 1e-3 | 5.1e5 | DIV | DIV | 9 | 3–4 | 2 | 9 | 2–3 | 2 | 2 |
| S3 | 1e-4 | 5.5e6 | NC† | NC† | NC† | 6–7 | 3 | NC† | 4–5 | 2 | 3–4 |
| S4 | 1e-5 | 5.9e7 | NC† | NC† | NC† | DIV | 5 | NC† | NC | 3 | 7 |
| S5 | 1e-6 | 6.2e8 | NC† | NC† | NC† | NC† | 25 (1 of 3)† | NC† | NC† | 6 | DIV† |
| S6 | 1e-7 | 6.6e9 | NC† | NC† | NC† | NC† | NC† | NC† | NC† | NC† | NC† |

- **FP16** converges only on S1 (κ₁ = 3.9e3): it diverges on S2 and does not converge beyond.
- **AP** reaches S2 with ε = 1e-4 (9 iterations), S3 with 1e-6 (6–7) and S4 with 1e-8 (5). FP32 also reaches S4 (7).
- **RAP 1e-8** reaches S5 (6 iterations), a decade further than AP 1e-8. On this family AP's factor is six times less accurate than RAP's at the same ε (section 4.8).
- **S5 is the one partial result:** AP 1e-8 converged for one right-hand side, after 25 iterations.

**Family L (2-D squared exponential, β = 0.3: weak decay)**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 1 | 2.9e4 | 5 | 5 | 3 | 2 | 1 | 3 | 2 | 1 | 2 |
| L2 | 1e-1 | 3.1e5 | NC | NC | 4 | 2 | 1 | 4–5 | 2 | 1 | 2 |
| L3 | 1e-2 | 3.4e6 | NC† | NC† | 12–13 | 2 | 1 | 13–16 | 2 | 2 | 2 |
| L4 | 1e-3 | 7.3e7 | NC† | NC† | NC† | 4 | 2 | NC† | 4 | 2 | 4 |
| L5 | 1e-4 | 3.6e8 | NC† | NC† | NC† | 11 | 2 | NC† | 11–12 | 3 | 11 |
| L6 | 1e-5 | 4.4e9 | NC† | NC† | NC† | NC† | 4 | NC† | NC† | 10–11 | NC† |

- **FP16** converges only on L1 (κ₁ = 2.9e4).
- **AP** reaches L3 with ε = 1e-4 (12–13 iterations) and L5 with 1e-6 (11), as FP32 does. AP 1e-8 converges on all six matrices in at most 4 iterations; it runs 92% of its GEMMs in FP64 on this family.
- **RAP 1e-8** also converges on all six, with up to 10–11 iterations at L6.

**Family T (3-D squared exponential, β = 0.1)**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 1e-1 | 2.0e4 | 4–5 | 4–5 | 3 | 2 | 1 | 4 | 2 | 1 | 2 |
| T2 | 1e-2 | 2.1e5 | 9 | 9 | 5 | 2 | 2 | 5 | 2 | 1–2 | 2 |
| T3 | 1e-3 | 2.5e6 | NC† | NC† | 14 | 3 | 2 | 14–16 | 3 | 2 | 2–3 |
| T4 | 1e-4 | 3.1e7 | DIV† | NC† | NC† | 6 | 3 | NC† | 5 | 2 | 4 |
| T5 | 1e-5 | 3.3e8 | NC† | NC† | NC† | DIV | 5 | NC† | NC | 3 | 9 |
| T6 | 1e-6 | 2.5e9 | NC† | NC† | NC† | NC† | DIV | NC† | NC† | 9 | NC† |

- **FP16** converges up to T2 (κ₁ = 2.1e5, 9 iterations).
- **AP** reaches T3 with ε = 1e-4 (14 iterations), T4 with 1e-6 (6) and T5 with 1e-8 (5). FP32 also reaches T5 (9).
- **RAP 1e-8** converges on all six, with 9 iterations at T6 (κ₁ = 2.5e9).

**Family Ma (2-D Matérn, ν = 0.5)**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ma1 | 1 | 2.3e3 | 3 | 3 | 2 | 1 | 1 | 2 | 2 | 1 | 1 |
| Ma2 | 1e-1 | 1.6e4 | 4 | 4 | 2 | 2 | 1 | 3 | 2 | 1 | 2 |
| Ma3 | 1e-2 | 7.1e4 | 4–5 | 4 | 3 | 2 | 1 | 4 | 2 | 1 | 2 |
| Ma4 | 1e-4 | 1.1e5 | 5 | 4–5 | 3 | 2 | 1 | 4 | 2 | 1 | 2 |

κ₁ stays at or below 1.1e5. Every factorization converges, FP16 in 3–5 iterations.

**Family Mb (2-D Matérn, ν = 1.5)**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mb1 | 1e-1 | 2.9e4 | 4–5 | 4–5 | 3 | 2 | 1 | 3 | 2 | 1 | 2 |
| Mb2 | 1e-2 | 2.7e5 | 9 | 8–9 | 4 | 2 | 1 | 5 | 2 | 1–2 | 2 |
| Mb3 | 1e-3 | 2.2e6 | DIV | DIV | 7–8 | 3 | 2 | 13–15 | 3 | 2 | 3 |
| Mb4 | 1e-4 | 1.6e7 | NC† | NC† | 22–29 | 3 | 2 | DIV | 4 | 2 | 3 |
| Mb5 | 1e-6 | 9.7e7 | NC† | NC† | DIV | 4 | 2 | NC† | 4 | 2 | 4 |

- **FP16** converges up to Mb2 (κ₁ = 2.7e5, 9 iterations).
- **AP 1e-4** reaches Mb4 (1.6e7) with 22–29 iterations. RAP 1e-4 reaches only Mb3, with 13–15.
- **ε = 1e-6, ε = 1e-8 and FP32** converge on all five, in at most 4 iterations.

**Family Mc (2-D Matérn, ν = 2.5)**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mc1 | 1e-2 | 3.4e5 | 11 | 11 | 4 | 2 | 1 | 5–6 | 2 | 1 | 2 |
| Mc2 | 1e-3 | 3.3e6 | DIV† | DIV† | 9–10 | 3 | 2 | NC | 3 | 2 | 3 |
| Mc3 | 1e-4 | 2.9e7 | NC† | NC† | NC† | 4 | 2 | NC† | 5–6 | 2 | 4 |
| Mc4 | 1e-5 | 2.5e8 | NC† | NC† | NC† | 14–15 | 3 | NC† | DIV | 3 | 14–15 |
| Mc5 | 1e-6 | 2.0e9 | NC† | NC† | NC† | DIV† | 4 | NC† | NC† | 5–6 | DIV† |
| Mc6 | 1e-7 | 1.4e10 | NC† | NC† | NC† | NC† | 9–11 | NC† | NC† | NC | NC† |

- **FP16** converges only on Mc1 (κ₁ = 3.4e5, 11 iterations).
- **AP** reaches Mc2 with ε = 1e-4 (9–10 iterations) and Mc4 with 1e-6 (14–15), as FP32 does.
- **AP 1e-8** converges on all six. At Mc6 (κ₁ = 1.4e10) it needs 9–11 iterations; this is the largest κ at which any factorization converged with classic IR, apart from FP64.
- **RAP** reaches Mc5 with ε = 1e-8. With ε = 1e-6 it stops one matrix before AP (Mc3 against Mc4).

**Family R (Gaussian RBF, mesh deformation)**

| Matrix | regularization | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 1 | 1.3e4 | 6 | 6 | 3 | 2 | 1 | 3 | 2 | 1 | 2 |
| R2 | 1e-1 | 2.4e5 | DIV† | DIV† | 6 | 3 | 1 | 6 | 2–3 | 1 | 2 |
| R3 | 1e-2 | 2.2e6 | NC† | NC† | DIV | 5 | 2 | NC | 3 | 1–2 | 2–3 |
| R4 | 1e-3 | 2.2e7 | NC† | NC† | NC† | 15–16 | 2 | NC† | 8–10 | 2 | 5–6 |
| R5 | 1e-5 | 3.6e8 | NC† | NC† | NC† | NC† | 4–5 | NC† | NC† | 5 | NC† |
| R6 | 1e-6 | 3.6e9 | NC† | NC† | NC† | NC† | 21–23† | NC† | NC† | DIV† | NC† |

- **FP16** converges only on R1 (κ₁ = 1.3e4, 6 iterations).
- **AP** reaches R2 with ε = 1e-4 (6 iterations) and R4 with 1e-6 (15–16), as FP32 does (5–6).
- **AP 1e-8** converges on all six, with 21–23 iterations at R6 after a shift. RAP 1e-8 stops at R5.

### 4.7 GMRES-IR

**Family D**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 1e2 | 2.5e1 | 2 / 2 | 2 / 2 | 2 / 2 | 1 / 1 | 1 / 1 | 2 / 2 | 1 / 1 | 1 / 1 | 1 / 1 |
| D2 | 1e1 | 3.7e2 | 1–2 / 2–3 | 1 / 2 | 2 / 2 | 1 / 1 | 1 / 1 | 2 / 2 | 1 / 1 | 1 / 1 | 1 / 1 |
| D3 | 1 | 4.5e3 | 2 / 4 | 2 / 4 | 1 / 2 | 2 / 2 | 1 / 1 | 2 / 4 | 2 / 2 | 1 / 1 | 1 / 1 |
| D4 | 1e-1 | 5.2e4 | 2 / 6 | 2 / 6 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 2 |
| D5 | 1e-2 | 5.7e5 | 2–3 / 10–15 | 2–3 / 10–15 | 2 / 6 | 1–2 / 2–4 | 1–2 / 2–3 | 2 / 6 | 1 / 2 | 2 / 2 | 1 / 2 |
| D6 | 1e-3 | 6.1e6 | 3 / 36† | 3 / 36† | 2 / 11–12 | 2 / 4–5 | 1 / 2 | 2–3 / 20–30 | 2 / 4 | 1 / 2 | 2 / 4 |
| D7 | 1e-4 | 6.5e7 | 3 / 108–109† | 3 / 107–109† | 3 / 45–46† | 2 / 10 | 1 / 3 | 3 / 54–56† | 2 / 7–8 | 2 / 4 | 2 / 6 |
| D8 | 1e-5 | 6.6e8 | 3 / 283–286† | 3 / 282–284† | 3 / 140† | 2–3 / 22–33† | 2 / 11–12 | 3 / 128–159† | 2 / 20† | 1–2 / 4–7 | 2 / 19 |
| D9 | 1e-6 | 6.8e9 | 8 / 800† | 8 / 800† | 4 / 393–394† | 2–3 / 63–97† | 2 / 25–26† | 6–7 / 600–700† | 2 / 59–60† | 2 / 17–18† | 2 / 55–56† |
| D10 | 1e-7 | 6.8e10 | NC† | NC† | 10 / 1000 (1 of 3)† | 2–3 / 178–277† | 2 / 77–79† | NC† | 2–3 / 171–264† | 3 / 53–59† | 2 / 161–163† |

- **Every factorization converges up to D9 (κ₁ = 6.8e9).** At D10, FP16, FP16-MAGMA and RAP 1e-4 do not converge, and AP 1e-4 converges for one right-hand side of three.
- **The cost moves into the GMRES iterations**, which grow with κ × factor error. At D9:
  - ε = 1e-4: 1.1 to 2 times fewer GMRES iterations than FP16;
  - ε = 1e-6: 8 to 14 times fewer;
  - ε = 1e-8: 31 to 47 times fewer.
- **FP16's 8 / 800 at D9** means every GMRES solve stopped at its 100-iteration limit, and the refinement still converged after 8 of them.

**Family E**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 1 | 9.3e2 | 2 / 4 | 2 / 4 | 2 / 2 | 1 / 1 | 1 / 1 | 2 / 3 | 2 / 2 | 1 / 1 | 1 / 1 |
| E2 | 1 | 2.9e4 | 2 / 5–6 | 2 / 5–6 | 2 / 2 | 2 / 2 | 0 / 0 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 2 |
| E3 | 1e-1 | 2.4e5 | 2–3 / 10–14 | 2–3 / 10–14 | 1 / 2 | 1 / 2 | 0 / 0 | 2 / 4 | 1 / 2 | 1 / 1 | 1 / 2 |
| E4 | 1e-2 | 1.0e6 | 3 / 19–21† | 3 / 19–21† | 1 / 2 | 1 / 2 | 0 / 0 | 2 / 6 | 1 / 2 | 1 / 1 | 1 / 2 |

All factorizations converge. FP16 and FP16-MAGMA need the most GMRES iterations (19–21 at E4); the others need at most 6.

**Family U**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U1 | 1e3 | 4.2e1 | 2 / 2 | 2 / 2 | 2 / 2 | 1 / 1 | 1 / 1 | 2 / 2 | 1 / 1 | 1 / 1 | 1 / 1 |
| U2 | 1e1 | 4.1e3 | 2 / 4 | 2 / 4 | 2 / 2 | 2 / 2 | 0 / 0 | 2 / 2 | 2 / 2 | 1 / 1 | 2 / 2 |
| U3 | 1e-1 | 4.9e5 | 3–4 / 24–33† | 3–4 / 24–33† | 1 / 2 | 1 / 2 | 0 / 0 | 1 / 2 | 1 / 2 | 1 / 1 | 1 / 2 |
| U4 | 1e-3 | 8.4e7 | 2–3 / 19–30† | 2–3 / 19–30† | 2 / 9 | 2 / 9 | 0 / 0 | 2 / 9 | 2 / 9 | 1 / 2 | 2 / 9 |
| U5 | 1e-5 | 8.3e9 | 3 / 30–32† | 3 / 30–32† | 3 / 31† | 3 / 37–38† | 0 / 0 | 3 / 31† | 3 / 37–38† | 1 / 3 | 3 / 40–41† |

All factorizations converge with at most 41 GMRES iterations, even FP16 with a shift as large as the diagonal (U4, U5).
- The U matrices are a numerically low-rank kernel plus noise·I: apart from a few large eigenvalues, every eigenvalue equals the noise.
- GMRES therefore converges in a few dozen iterations with almost any preconditioner, and these runs do not separate the factorizations.

**Family S**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 1e-1 | 3.9e3 | 2 / 4 | 2 / 4 | 2 / 4 | 2 / 2 | 2 / 2 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 2 |
| S2 | 1e-3 | 5.1e5 | 3 / 29–30 | 3 / 27 | 2 / 8 | 2 / 4 | 1 / 2 | 2 / 8 | 1–2 / 2–4 | 2 / 2 | 1 / 2 |
| S3 | 1e-4 | 5.5e6 | 3 / 56–57† | 3 / 56† | 3 / 26† | 2 / 6–7 | 2 / 4 | 3 / 26–27† | 2 / 5–6 | 1 / 2 | 2 / 4 |
| S4 | 1e-5 | 5.9e7 | 3 / 162–163† | 3 / 160† | 3 / 75–76† | 2 / 16–17 | 2 / 6 | 3 / 76† | 2 / 10–11 | 2 / 4 | 2 / 8 |
| S5 | 1e-6 | 6.2e8 | 5 / 500† | 4–5 / 400–500† | 3 / 219–224† | 3 / 52† | 2 / 15† | 3 / 218–221† | 2 / 19† | 2 / 8 | 2 / 32–34† |
| S6 | 1e-7 | 6.6e9 | NC† | NC† | 6 / 600† | 3 / 150–152† | 3 / 57–58† | 6 / 600† | 2–3 / 54–81† | 2 / 21–22† | 2–3 / 53–79† |

- **FP16** converges up to S5 with 500 GMRES iterations, and not at S6.
- **At S5**, AP 1e-8 needs 15 GMRES iterations and RAP 1e-8 needs 8.
- **At S6 (κ₁ = 6.6e9)**, AP 1e-8 needs 57–58 and RAP 1e-8 needs 21–22.

**Family L**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 1 | 2.9e4 | 2 / 6 | 2 / 6 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 2 |
| L2 | 1e-1 | 3.1e5 | 2 / 12–13 | 2 / 12–13 | 2 / 6 | 1 / 2 | 1 / 1 | 2 / 6 | 1 / 2 | 1 / 1 | 1 / 2 |
| L3 | 1e-2 | 3.4e6 | 3 / 41–43† | 3 / 42–43† | 2 / 10 | 1 / 2 | 1 / 1 | 2 / 10–11 | 1 / 2 | 1 / 2 | 1 / 2 |
| L4 | 1e-3 | 7.3e7 | 3 / 111–113† | 3 / 110–113† | 3 / 31–33† | 1 / 3 | 1 / 2 | 3 / 33–35† | 1 / 3 | 1 / 2 | 1 / 3 |
| L5 | 1e-4 | 3.6e8 | 3 / 176–180† | 3 / 177–178† | 3 / 90–91† | 2 / 10 | 1 / 2 | 3 / 97–98† | 2 / 10–11 | 1 / 3 | 2 / 10 |
| L6 | 1e-5 | 4.4e9 | 3 / 232–235† | 3 / 233–235† | 3 / 168–170† | 2 / 23† | 1 / 4 | 3 / 171–176† | 2 / 23† | 2 / 10 | 2 / 23† |

All factorizations converge. At L6 FP16 needs 232–235 GMRES iterations, AP 1e-6 23 and AP 1e-8 4.

**Family T**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T1 | 1e-1 | 2.0e4 | 2 / 4 | 2 / 4 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 2 |
| T2 | 1e-2 | 2.1e5 | 2 / 8 | 2 / 8 | 2 / 6 | 1 / 2 | 2 / 2 | 2 / 6 | 1 / 2 | 1–2 / 1–2 | 1 / 2 |
| T3 | 1e-3 | 2.5e6 | 3 / 24–26† | 3 / 26† | 2 / 10 | 2 / 4 | 1 / 2 | 2 / 10 | 2 / 4 | 1 / 2 | 1 / 2 |
| T4 | 1e-4 | 3.1e7 | 3 / 77–80† | 3 / 96† | 3 / 36–37† | 2 / 6 | 1–2 / 2–4 | 3 / 34† | 2 / 6 | 1 / 2 | 2 / 5 |
| T5 | 1e-5 | 3.3e8 | 3 / 272† | 3 / 272† | 3 / 133–136† | 2 / 22–23 | 2 / 6 | 3 / 131–133† | 2 / 13–14 | 1 / 3 | 2 / 9 |
| T6 | 1e-6 | 2.5e9 | 8 / 800† | 8–9 / 800–900† | 4 / 392† | 2 / 34–35† | 2 / 18–20 | 4 / 392† | 3 / 49† | 2 / 10 | 2 / 21† |

All factorizations converge. At T6 FP16 needs 800 GMRES iterations, AP 1e-6 34–35, AP 1e-8 18–20 and RAP 1e-8 10.

**Family Ma**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Ma1 | 1 | 2.3e3 | 2 / 4 | 2 / 4 | 2 / 2 | 1 / 1 | 1 / 1 | 1 / 2 | 2 / 2 | 1 / 1 | 1 / 1 |
| Ma2 | 1e-1 | 1.6e4 | 2 / 4 | 2 / 4 | 1 / 2 | 2 / 2 | 1 / 1 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 2 |
| Ma3 | 1e-2 | 7.1e4 | 2 / 4 | 2 / 4 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 2 |
| Ma4 | 1e-4 | 1.1e5 | 2 / 5 | 2 / 5 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 2 |

All factorizations converge with at most 5 GMRES iterations.

**Family Mb**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mb1 | 1e-1 | 2.9e4 | 2 / 4 | 2 / 4 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 2 |
| Mb2 | 1e-2 | 2.7e5 | 2 / 8 | 2 / 8 | 2 / 4 | 1 / 2 | 1 / 1 | 2 / 6 | 1 / 2 | 1–2 / 1–2 | 1 / 2 |
| Mb3 | 1e-3 | 2.2e6 | 3 / 35 | 3 / 35 | 2 / 7–8 | 1–2 / 2–4 | 2 / 3 | 2 / 10 | 2 / 4 | 1 / 2 | 1 / 2 |
| Mb4 | 1e-4 | 1.6e7 | 3 / 48–50† | 3 / 51† | 2 / 11–12 | 2 / 4 | 1 / 2 | 3 / 34–35 | 2 / 5 | 1 / 2 | 2 / 4 |
| Mb5 | 1e-6 | 9.7e7 | 3 / 108† | 3 / 108† | 2 / 15 | 2 / 6 | 1 / 2 | 2–3 / 33–52† | 2 / 6 | 1 / 2 | 2 / 6 |

All factorizations converge. At Mb5 FP16 needs 108 GMRES iterations, AP 1e-4 15, AP 1e-6 6 and AP 1e-8 2.

**Family Mc**

| Matrix | noise | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Mc1 | 1e-2 | 3.4e5 | 2–3 / 8–12 | 2–3 / 8–12 | 2 / 4–6 | 1 / 2 | 1 / 1 | 2 / 6 | 1 / 2 | 1 / 1 | 1 / 2 |
| Mc2 | 1e-3 | 3.3e6 | 3 / 38–39† | 3 / 37† | 2 / 8 | 2 / 4 | 1 / 2 | 2 / 12–13 | 2 / 4 | 1 / 2 | 2 / 4 |
| Mc3 | 1e-4 | 2.9e7 | 3 / 94–95† | 3 / 94–96† | 3 / 26–27† | 2 / 6 | 1 / 2 | 3 / 46–47† | 2 / 6 | 1 / 2 | 2 / 6 |
| Mc4 | 1e-5 | 2.5e8 | 3 / 264–265† | 3 / 264–265† | 3 / 72–73† | 2 / 11 | 2 / 5 | 3 / 128–130† | 2 / 14–15 | 1 / 3 | 2 / 11–12 |
| Mc5 | 1e-6 | 2.0e9 | 6–7 / 600–700† | 6 / 600† | 3 / 183–184† | 2 / 36† | 1–2 / 4–7 | 3 / 288–289† | 2 / 28† | 2 / 8 | 2 / 38–39† |
| Mc6 | 1e-7 | 1.4e10 | NC† | NC† | 3 / 288–290† | 2 / 59–60† | 2 / 11 | 7 / 700† | 2 / 61† | 1–2 / 7–14 | 2 / 58–59† |

- **FP16** converges up to Mc5 (600–700 GMRES iterations) and not at Mc6.
- **At Mc6 (κ₁ = 1.4e10)**, AP 1e-4 needs 288–290 GMRES iterations, AP 1e-6 59–60 and AP 1e-8 11.

**Family R**

| Matrix | regularization | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R1 | 1 | 1.3e4 | 2 / 6 | 2 / 6 | 2 / 4 | 2 / 2–3 | 1 / 1 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 2 |
| R2 | 1e-1 | 2.4e5 | 3 / 21† | 3 / 21† | 2 / 6 | 2 / 4 | 1 / 1 | 2 / 6 | 1 / 2 | 1 / 1 | 1 / 2 |
| R3 | 1e-2 | 2.2e6 | 3 / 58–60† | 3 / 58–60† | 2 / 15 | 2 / 6 | 1 / 2 | 2 / 12 | 2 / 4 | 1–2 / 1–2 | 1–2 / 2–4 |
| R4 | 1e-3 | 2.2e7 | 3 / 209–210† | 3 / 209–210† | 3 / 58–60† | 2 / 10 | 1 / 2 | 3 / 58† | 2 / 8 | 1 / 2 | 2 / 6 |
| R5 | 1e-5 | 3.6e8 | NC† | NC† | 4–5 / 380–479† | 3 / 91–94† | 1 / 4 | 4 / 380–385† | 3 / 91–93† | 1 / 4 | 3 / 76–79† |
| R6 | 1e-6 | 3.6e9 | NC† | NC† | NC† | 3 / 138–142† | 2 / 13–14† | 6–9 / 600–900† | 3 / 113–118† | 2 / 24–26† | 3 / 105–106† |

- **FP16** converges up to R4 (209–210 GMRES iterations), and not at R5 or R6.
- **At R6**:
  - AP 1e-4 does not converge;
  - AP 1e-6 needs 138–142 GMRES iterations;
  - AP 1e-8 needs 13–14;
  - RAP 1e-8 needs 24–26.

### 4.8 Why: factor error, precision and breakdowns

**Factor error.** The error is ‖F − L̂L̂ᵀ‖₂/‖F‖₂, estimated by power iteration. One matrix per family, the one closest to κ₁ = 5e5:

| Family | Matrix | κ₁ | FP16 | AP 1e-4 | RAP 1e-4 | AP 1e-6 | RAP 1e-6 | AP 1e-8 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|
| D | D5 | 5.7e5 | 8.7e-6 | 3.1e-7 | 1.5e-6 | 3.1e-8 | 2.8e-8 | 2.5e-9 | 5.4e-9 | 2.9e-8 |
| E | E3 | 2.4e5 | 6.2e-5 | 2.3e-9 | 2.7e-6 | 2.3e-9 | 1.1e-8 | 9.8e-17 | 2.9e-10 | 2.3e-9 |
| U | U3 | 4.9e5 | 1.1e-4 | 1.2e-8 | 2.5e-8 | 1.2e-8 | 1.7e-8 | 7.9e-17 | 1.3e-10 | 1.2e-8 |
| S | S2 | 5.1e5 | 1.6e-5 | 2.2e-6 | 2.0e-6 | 1.4e-7 | 2.2e-8 | 1.4e-8 | 2.2e-9 | 2.2e-8 |
| L | L2 | 3.1e5 | 2.4e-5 | 4.6e-7 | 2.4e-6 | 2.6e-9 | 5.5e-9 | 8.8e-11 | 3.3e-10 | 2.6e-9 |
| T | T2 | 2.1e5 | 1.5e-5 | 1.9e-6 | 2.2e-6 | 3.2e-8 | 4.3e-8 | 3.7e-9 | 2.3e-9 | 3.0e-8 |
| Ma | Ma4 | 1.1e5 | 7.6e-6 | 4.8e-7 | 2.2e-6 | 2.4e-8 | 4.7e-8 | 1.3e-9 | 7.5e-9 | 2.4e-8 |
| Mb | Mb2 | 2.7e5 | 8.1e-6 | 3.8e-7 | 1.9e-6 | 3.1e-8 | 4.8e-8 | 9.0e-10 | 5.2e-9 | 3.0e-8 |
| Mc | Mc1 | 3.4e5 | 9.4e-6 | 3.4e-7 | 1.9e-6 | 2.8e-8 | 4.3e-8 | 5.9e-10 | 3.4e-9 | 2.8e-8 |
| R | R2 | 2.4e5 | 1.4e-4 | 2.8e-6 | 2.6e-6 | 1.0e-7 | 1.0e-7 | 1.9e-9 | 1.6e-9 | 1.0e-7 |

**Share of the off-diagonal GEMMs that run in FP16 (AP / RAP)**, same matrices:

| Family | ε = 1e-4 | ε = 1e-6 | ε = 1e-8 |
|---|---|---|---|
| D | 57% / 92% | 29% / 69% | 16% / 49% |
| E | 0% / 89% | 0% / 10% | 0% / 0% |
| U | 0% / 92% | 0% / 89% | 0% / 18% |
| S | 77% / 98% | 69% / 96% | 61% / 92% |
| L | 11% / 84% | 0% / 11% | 0% / 0% |
| T | 51% / 88% | 19% / 57% | 7% / 30% |
| Ma | 47% / 93% | 1% / 52% | 0% / 14% |
| Mb | 53% / 93% | 12% / 64% | 0% / 32% |
| Mc | 55% / 93% | 16% / 67% | 2% / 38% |
| R | 30% / 89% | 11% / 42% | 8% / 14% |

**AP.**
- AP's error falls as ε decreases, except on E3 and U3, where AP 1e-4 and 1e-6 are the same FP32 factorization.
- **On the decaying families other than S, it stays below ε:**
  - 35 to 330 times below at ε = 1e-4;
  - 10 to 36 times below at 1e-6 (L: 380 times);
  - 3 to 17 times below at 1e-8 (L, where AP 1e-8 runs 92% of its GEMMs in FP64: 110 times).
- **On S it comes closest:** 45, 7 and 0.7 times below. At ε = 1e-8 its error (1.4e-8) exceeds ε. There AP still keeps 61% of the GEMMs in FP16 at ε = 1e-8, more than on any other family.
- **On E3 and U3** it runs no GEMM in FP16.

**RAP.** RAP tests each GEMM's ‖L_ik‖·‖L_jk‖ instead of the tile's norm. One update is much smaller than the whole tile, so many more GEMMs pass the FP16 test.
- **At ε = 1e-4**, 84% to 98% of the GEMMs run in FP16 on the families shown. Each of these GEMMs adds its own FP16 error to the tile, and the factor error is about 2e-6 on every family except U.
- **At ε = 1e-6** only the small updates pass the test, and RAP's error is within about a factor of two of AP's on eight families.
- **At ε = 1e-8** RAP's error is 2 to 6 times AP's on D, L, Ma, Mb and Mc, and smaller than AP's on S, T and R. On E and U, where AP 1e-8 is an FP64 factorization, it is far above AP's.
- **On S, RAP is more accurate than AP at ε = 1e-6 and 1e-8.** It is six times more accurate there, with more FP16 GEMMs (92% against 61% at 1e-8). AP sends a whole tile to FP16 when the tile's norm is small, and on S this holds for most tiles. RAP sends only the individual small updates to FP16, and runs the large updates of the same tile in FP32 or FP64.

**κ × factor error decides classic IR.** Each classic IR iteration multiplies the error by roughly κ × ‖E‖/‖A‖. The table gives, per family, the largest κ₁(A) × factor error among factorizations that converged and the smallest among those that did not (classic IR, all three right-hand sides):

| Family | Converged: largest κ₁ × error | Not converged: smallest κ₁ × error |
|---|---|---|
| D | 5 | 11 |
| E | 2.8 | 15 |
| U | 1 | 54 |
| S | 1.4 | 3.7 |
| L | 7.6 | 7.6 |
| T | 11 | 14 |
| Ma | 0.87 | - |
| Mb | 7 | 19 |
| Mc | 9.7 | 7.1 |
| R | 5.5 | 4.5 |

- **Over all families:**
  - every run below 3.7 converged (327 runs);
  - every run above 11.1 failed (166 runs);
  - between 3.7 and 11.1, 19 converged and 9 failed.
- **The threshold is lowest on S**, where RAP 1e-6 failed at S4 with 3.7. It is highest on T, where FP32 converged at T5 with 11.1.
- **It lies above 1 because κ₁ overestimates the 2-norm condition number** that the theory uses. The overestimate differs from kernel to kernel.
- **One run converged only partly.** At S5, AP 1e-8 (product 37) converged for one right-hand side of three.

**Breakdowns and shifts.** At large κ the low-precision factor of a positive definite matrix can meet a non-positive pivot. The solver then refactorizes F + s·I with s = c·u, where u is the unit roundoff of the factor (ε for AP and RAP). It tries c = 2, 8, 32, …, 8192: up to seven shifts, eight attempts in all. The table gives the first matrix of each family at which each factorization broke down (blank: never):

| Family | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|
| D | D6 (6.1e6) | D6 (6.1e6) | D7 (6.5e7) | D8 (6.6e8) | D9 (6.8e9) | D7 (6.5e7) | D8 (6.6e8) | D9 (6.8e9) | D9 (6.8e9) |
| E | E4 (1.0e6) | E4 (1.0e6) |  |  |  |  |  |  |  |
| U | U3 (4.9e5) | U3 (4.9e5) | U5 (8.3e9) | U5 (8.3e9) |  | U5 (8.3e9) | U5 (8.3e9) |  | U5 (8.3e9) |
| S | S3 (5.5e6) | S3 (5.5e6) | S3 (5.5e6) | S5 (6.2e8) | S5 (6.2e8) | S3 (5.5e6) | S5 (6.2e8) | S6 (6.6e9) | S5 (6.2e8) |
| L | L3 (3.4e6) | L3 (3.4e6) | L4 (7.3e7) | L6 (4.4e9) |  | L4 (7.3e7) | L6 (4.4e9) |  | L6 (4.4e9) |
| T | T3 (2.5e6) | T3 (2.5e6) | T4 (3.1e7) | T6 (2.5e9) |  | T4 (3.1e7) | T6 (2.5e9) |  | T6 (2.5e9) |
| Ma |  |  |  |  |  |  |  |  |  |
| Mb | Mb4 (1.6e7) | Mb4 (1.6e7) |  |  |  | Mb5 (9.7e7) |  |  |  |
| Mc | Mc2 (3.3e6) | Mc2 (3.3e6) | Mc3 (2.9e7) | Mc5 (2.0e9) |  | Mc3 (2.9e7) | Mc5 (2.0e9) |  | Mc5 (2.0e9) |
| R | R2 (2.4e5) | R2 (2.4e5) | R4 (2.2e7) | R5 (3.6e8) | R6 (3.6e9) | R4 (2.2e7) | R5 (3.6e8) | R6 (3.6e9) | R5 (3.6e8) |

- **No breakdown went unrescued.** The eighth and last attempt (c = 8192) was needed three times, all for FP32: on U5, R5 and R6.
- **The largest shifts**, relative to the unit diagonal of F:

  | Factorization | Largest shift s | Where |
  |---|---|---|
  | FP16, FP16-MAGMA | 1.0, the diagonal itself | U4–U5 |
  | AP 1e-4, RAP 1e-4 | 1.3e-2 | R4–R6 |
  | AP 1e-6, RAP 1e-6 | 5.1e-4 | U5, R5–R6 |
  | AP 1e-8 | 5.1e-6 | D9–D10 |
  | RAP 1e-8 | 1.3e-6 | D10 |
  | FP32 | 4.9e-4 | U5, R5–R6 |

- **Shifts limit refinement.** A shifted factor approximates F + s·I, so the refinement also has to remove s. The factor errors above are measured against the shifted F and do not include s.
- **GMRES-IR tolerates large shifts on U.** On U4 and U5 the FP16 shift equals the diagonal, and GMRES-IR still converges, for the spectral reason given in section 4.7. Classic IR does not.

### 4.9 Effect of ε

AP and RAP over ε from 1e-2 to 1e-10 on two matrices of family D: D6 (κ₁ = 6.1e6) and D9 (κ₁ = 6.8e9). These runs used four A100s (guyot), and nine cases were rerun on the hexane V100. Where they overlap with sections 4.6 and 4.7, the counts are close to the H100 ones.

| ε | AP D6 classic | RAP D6 classic | AP D6 GMRES-IR | RAP D6 GMRES-IR | AP D9 GMRES-IR | RAP D9 GMRES-IR |
|---|---|---|---|---|---|---|
| 1e-2 | NC† | NC† | 3 / 64–66† | 3 / 64–66† | NC† | NC† |
| 3e-3 | NC† | NC† | 3 / 41–42† | 3 / 41–42† | 10 / 1000† | 10 / 1000 (2 of 3)† |
| 1e-3 | NC† | DIV† | 3 / 28† | 3 / 45† | NC† | 10 / 1000 (1 of 3)† |
| 3e-4 | DIV | DIV† | 2 / 19 | 2–3 / 32–48† | 6 / 600† | 6–7 / 600–700† |
| 1e-4 | 20–23 | DIV | 2 / 11–12 | 2–3 / 22–31 | 3–4 / 295–393† | 7 / 700† |
| 3e-5 | 8 | 14–15 | 2 / 8 | 2 / 10 | 3 / 251–253† | 3–4 / 298–400† |
| 1e-5 | 6–7 | 6–7 | 2 / 7 | 2 / 7–8 | 3 / 254–256† | 3 / 258–261† |
| 3e-6 | 4 | 4 | 2 / 6 | 2 / 6 | 2–3 / 97–152† | 3 / 149† |
| 1e-6 | 4 | 3 | 2 / 4–5 | 2 / 4 | 2 / 63† | 2 / 59–60† |
| 1e-7 | 3 | 3 | 2 / 4 | 2 / 4 | 2 / 40–42† | 2 / 37† |
| 1e-8 | 2 | 2 | 1 / 2 | 1 / 2 | 2 / 25† | 2 / 18–20† |
| 1e-9 | 2 | 1 | 1 / 2 | 1 / 1 | 2 / 29† | 1 / 3 |
| 1e-10 | 2 | 1 | 1 / 2 | 1 / 1 | 2 / 19–20† | 1 / 2–3 |

| ε | AP FP16 GEMMs | AP factor error | RAP FP16 GEMMs | RAP factor error |
|---|---|---|---|---|
| 1e-2 | 100% | 9.1e-6 | 100% | 9.1e-6 |
| 3e-3 | 100% | 9.3e-6 | 100% | 8.6e-6 |
| 1e-3 | 87% | 4.9e-6 | 100% | 9.2e-6 |
| 3e-4 | 69% | 1.1e-6 | 96% | 4.4e-6 |
| 1e-4 | 57% | 3.4e-7 | 91% | 2.1e-6 |
| 3e-5 | 52% | 2.0e-7 | 85% | 5.3e-7 |
| 1e-5 | 45% | 1.2e-7 | 80% | 1.8e-7 |
| 3e-6 | 35% | 4.9e-8 | 73% | 5.9e-8 |
| 1e-6 | 29% | 2.6e-8 | 67% | 2.8e-8 |
| 1e-7 | 22% | 1.0e-8 | 57% | 1.7e-8 |
| 1e-8 | 16% | 3.1e-9 | 47% | 3.5e-9 |
| 1e-9 | 13% | 2.5e-9 | 37% | 5.7e-10 |
| 1e-10 | 8% | 1.9e-9 | 30% | 2.8e-10 |

- **ε ≥ 3e-3 is FP16.** Every GEMM runs in FP16, and AP and RAP behave like FP16.
- **The factor error falls steadily as ε decreases**, from 9e-6 (all FP16) to 2e-9 for AP and 3e-10 for RAP at ε = 1e-10, and the iterations follow it. Classic IR on D6 converges from ε = 1e-4 for AP and from 3e-5 for RAP.
- **For ε ≤ 1e-5, RAP reaches AP's error, or a smaller one, with two to four times more FP16 GEMMs.** For ε ≥ 3e-5 it is less accurate than AP at the same ε.

### 4.10 Scaling and GMRES tolerance

**Scaling.** Classic IR on family D, with and without the Higham–Pranesh scaling:

| matrix | κ₁(A) | FP16 scaled | FP16 unscaled | AP 1e-6 scaled | AP 1e-6 unscaled |
|---|---|---|---|---|---|
| D1 | 2.5e1 | 2 | 2 | 1 | 1 |
| D2 | 3.7e2 | 2 | 2 | 1 | 1 |
| D3 | 4.5e3 | 3 | 3 | 2 | 2 |
| D4 | 5.2e4 | 5 | 5 | 2 | 2 |
| D5 | 5.7e5 | 17–19 | 17–18 | 2–3 | 2–3 |
| D6 | 6.1e6 | NC† | NC† | 4 | 4 |
| D7 | 6.5e7 | NC† | NC† | 11 | 9–10 |
| D8 | 6.6e8 | NC† | NC† | NC† | NC† |
| D9 | 6.8e9 | NC† | NC† | NC† | NC† |
| D10 | 6.8e10 | NC† | NC† | NC† | NC† |

Scaling changes the counts by at most two iterations on these matrices. Their diagonal is constant and their entries are of order one, so the scaling only multiplies the matrix by a scalar, and the precision maps stay the same.

**GMRES tolerance τ.** GMRES-IR with τ = 1e-2, 1e-4 and 1e-6 (A100 runs):

| matrix | κ₁(A) | FP16 τ = 1e-2 | FP16 τ = 1e-4 | FP16 τ = 1e-6 | AP 1e-6 τ = 1e-2 | AP 1e-6 τ = 1e-4 | AP 1e-6 τ = 1e-6 |
|---|---|---|---|---|---|---|---|
| D8 | 6.6e8 | 5 / 264–271† | 3 / 282–284† | 3 / 300† | 4 / 22–23† | 3 / 33† | 2 / 31–32† |
| D9 | 6.8e9 | 8 / 746–752† | 7–8 / 700–800† | 7–8 / 700–800† | 4 / 66–68† | 2 / 63† | 2 / 95–96† |

τ changes the GMRES iterations by at most 50%. The factor's accuracy decides the cost, not τ.

## 5. Other measurements

### 5.1 Kind 10 static against kind 5 (H100, `--check`, residual ‖LLᵀ−A‖_F/‖A‖_F)

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

### 5.2 Condition numbers of the generated matrices (A1a, A1c)

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

**The A1c kernels** (FP64 runs, same size; kind 3, kind 8 with β = 0.1, kind 7 with the Gaussian basis and Morton order). For kind 7 the first column is the diagonal regularization (`--adddiag`):

| noise or regularization | 3-D sq. exp., β = 0.1 | Matérn, ν = 0.5 | Matérn, ν = 1.5 | Matérn, ν = 2.5 | RBF, radius 0.05 | RBF, radius 0.1 |
|---|---|---|---|---|---|---|
| 1e1 | 1.0e2 | 2.6e2 | 2.9e2 | 3.1e2 | 1.1e3 | 1.2e3 |
| 1 | 1.5e3 | 2.3e3 | 3.0e3 | 3.2e3 | 1.3e4 | 2.5e4 |
| 1e-1 | 2.0e4 | 1.6e4 | 2.9e4 | 3.4e4 | 2.4e5 | 3.1e5 |
| 1e-2 | 2.1e5 | 7.1e4 | 2.7e5 | 3.4e5 | 2.2e6 | 3.8e6 |
| 1e-3 | 2.5e6 | 1.1e5 | 2.2e6 | 3.3e6 | 2.2e7 | 3.4e7 |
| 1e-4 | 3.1e7 | 1.1e5 | 1.6e7 | 2.9e7 | 7.9e7 | 4.6e8 |
| 1e-5 | 3.3e8 | 1.1e5 | 6.7e7 | 2.5e8 | 3.6e8 | 4.5e9 |
| 1e-6 | 2.5e9 | 1.1e5 | 9.7e7 | 2.0e9 | 3.6e9 | 3.5e10 |
| 1e-7 | 3.7e10 | 1.1e5 | 1.1e8 | 1.4e10 | 3.6e10 | 3.1e11 |
| 1e-8 | 4.0e11 | 1.1e5 | 1.1e8 | 2.8e10 | 3.6e11 | 3.0e12 |

- **Matérn:** κ₁ saturates at 1.1e5 for ν = 0.5 and 1.1e8 for ν = 1.5. For ν = 2.5 it keeps growing to 2.8e10 at 1e-8.
- **Section 4** uses radius 0.05 for the RBF matrix.


### 5.3 Cost per phase and time to solution (B0)

**Summary.** Time to solution does not improve yet with this implementation.
- **Factorization plus refinement only:** on the V100, AP 1e-4, RAP 1e-4 and RAP 1e-6 beat FP16 with IR, and RAP with ε = 1e-6 (2 iterations) is fastest: 1.68 s against 2.15 s for FP16 + IR (6 iterations) and 3.77 s for FP64 direct at N = 40960. On the H100, FP64 direct is faster than every mixed strategy up to N = 40960; AP at all three ε comes within 0.1 s of it.
- **What decides the totals now:** about 2 s of CPU conversions and re-sending of the factor that every mixed strategy pays, and about 0.15 to 0.2 s of task overhead per refinement iteration.

**Setup.** N = 30720 and 40960, NB = 1024, kind 2 β = 0.1 noise 1e-1 (κ₁ about 1e5). Times are the medians of runs 2 and 3, for the first right-hand side. The ε = 1e-8 rows come from a later allocation. Three cases repeated there (FP64, AP 1e-6 and RAP 1e-6 at N = 40960) matched the first session within 2% in every phase, except RAP's factorization on the H100: it varies from 0.74 to 1.08 s between runs in both sessions.

**Column definitions.**
- **decide+convert+stage** is the pre-analysis: precision decisions, FP64→FP32 conversion of the low-precision tiles (CPU), and the warm-up transfer of A to the GPU.
- **factor to FP64** is convert_s2d (CPU).
- **first solve** includes re-sending the FP64 factor.
- **sum** is everything from the pre-analysis to the end of the refinement. FP64 direct is counted with no refinement.
- **copy A + scale** is the FP64 copy kept for the residuals, plus the preprocessing. It is reported apart.

H100:

| N | strategy | decide+convert+stage | factorize | factor to FP64 | first solve | refine | iterations | sum | copy A + scale |
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

| N | strategy | decide+convert+stage | factorize | factor to FP64 | first solve | refine | iterations | sum | copy A + scale |
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

  Seconds at N = 40960, with the refinement iterations in parentheses (FP64 direct: factorize and one solve):

  | GPU | FP64 direct | FP16+IR | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 |
  |---|---|---|---|---|---|---|---|---|
  | V100, FP64 at 7.8 Tflop/s | 3.77 | 2.15 (6) | 2.04 (3) | 2.20 (2) | 2.92 (1) | 1.95 (4) | **1.68** (2) | 2.30 (2) |
  | H100, FP64 tensor cores | **1.09** | 1.42 (5) | 1.17 (3) | 1.19 (2) | 1.16 (1) | 1.42 (4) | 1.28 (2) | 1.20 (1) |

  - **V100:** AP 1e-4, RAP 1e-4 and RAP 1e-6 beat FP16+IR; AP 1e-6 and both ε = 1e-8 factors do not. RAP 1e-6 is fastest: its two iterations beat FP16's six, and the gap to FP64 is 2.2 times. At this κ (about 1e5) ε = 1e-8 saves at most one iteration against 1e-6, but its larger FP64 and FP32 share (AP keeps about 300 of the 820 tiles in FP64) costs more on the V100 than the iteration saves.
  - **H100:** at these sizes the FP64 factorization is only twice as slow as the FP16 one. With no conversions, each refinement iteration costs about as much as 20% of the FP64 factorization. AP at all three ε comes within 0.1 s of FP64 direct, but no mixed factor beats it up to N = 40960.
- **The overheads currently decide the totals.** Every mixed strategy pays them at N = 40960 on top of the table above; ε = 1e-8 pays less because it converts fewer tiles:

  | Overhead at N = 40960 | FP16, FP32, ε = 1e-4 and 1e-6 | ε = 1e-8 |
  |---|---|---|
  | Pre-analysis conversion | 0.6 s | 0.4 s |
  | Converting the factor back to FP64 (convert_s2d) | 1.05 s | 0.7 to 0.85 s |
  | Re-sending the FP64 factor | 0.35 to 0.5 s | 0.35 to 0.5 s |
  | Total | about 2.1 s | about 1.6 s |

  As a result, no mixed strategy beats FP64 direct on either GPU at these sizes. The fastest mixed one is RAP 1e-6 on the V100 (4.41 s against 4.31 s) and AP 1e-8 on the H100 (3.08 s against 1.46 s).
- **Each refinement iteration costs 0.15 to 0.2 s at N = 40960** (one A·v, one M⁻¹). With one right-hand side, the time goes into task and launch overhead of the tiled kernels, not into bandwidth.
- **What would close the gap** (plan stage C):
  - a triangular solve that reads each tile in its storage precision on the GPU, which removes the conversion back and the re-send;
  - fewer, larger tasks per refinement iteration.
