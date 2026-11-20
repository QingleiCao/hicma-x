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

Nine factorizations are compared on 19 matrices with κ₁(A) from 25 to 7e10, each with classic IR and with GMRES-IR. FP64 direct solves every matrix with 0 iterations; it is the accuracy reference.

**Main findings.**

1. **The factor's accuracy decides the outcome.** Over all 19 matrices, classic IR converged whenever κ₁(A) × factor error was at most 5, and failed whenever it was 11 or more (section 4.8). Everything below follows from how each factorization sets that error.
2. **On matrices whose tile norms decay (family D), AP and RAP at ε = 1e-6 and 1e-8 converge where FP16 cannot.**
   - Classic IR with FP16 converges up to κ₁ = 5.7e5, with 17–19 iterations there.
   - With ε = 1e-6 it converges up to 6.5e7 (AP 11 iterations, RAP 6).
   - With ε = 1e-8 it converges up to 6.6e8 (AP 13–15 iterations, RAP 4–5).
3. **With GMRES-IR almost every factorization converges, and the difference moves into the GMRES iterations.** At κ₁ = 6.8e9, FP16 needs 800 GMRES iterations. ε = 1e-6 needs 59–97 and ε = 1e-8 needs 17–26: 8 to 47 times fewer.
4. **ε = 1e-4 gains little.** AP 1e-4 extends classic IR by one decade, to κ₁ = 6.1e6 with 21–23 iterations, and halves FP16's GMRES iterations. RAP 1e-4 extends classic IR not at all.
5. **RAP against AP.** On family D, at ε = 1e-6 and 1e-8, RAP converges as far as AP with the same or fewer iterations. At ε = 1e-4 its factor is about five times less accurate than AP's, and it does worse. On E and U, AP 1e-8 is an FP64 factorization and needs 0 iterations where RAP 1e-8 needs 1–2.
6. **Without tile-norm decay there is nothing to adapt.** On families E (except E1) and U, AP puts every off-diagonal tile in one precision and needs the same iterations as that uniform precision. The only differences are on U5 with GMRES-IR, where the breakdown shifts differ.
7. **FP16-MAGMA** (an FP32 instead of an FP64 diagonal) is within one iteration of FP16 everywhere.
8. **No run failed outright, but low-precision Cholesky breaks down at large κ.**
   - FP16 broke down from κ₁ ≈ 5e5 to 6e6, depending on the family.
   - AP, RAP and FP32 broke down from 6.5e7 to 6.8e9 on family D, and at 8.3e9 on family U.
   - A diagonal shift rescued every breakdown. The shifted cases are marked †.

### 4.1 Setup

- **Hardware and size.** One H100 (hexane), N = 20480, NB = 512: 40 × 40 tiles, so 820 tiles in the lower triangle, 40 of them diagonal.
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

The precision maps AP and RAP choose depend on the matrix; section 4.8 lists them.

### 4.3 Test matrices

Three families of kernel matrices, with κ swept by the noise added to the diagonal. Every diagonal entry is the same (1 + noise), so the scaling does not change κ. κ₁(A) is the Hager–Higham estimate from the FP64 factor.

| Family | Kernel | What it tests |
|---|---|---|
| D | kind 2 (2-D squared exponential), β = 0.1 | Tile norms decay away from the diagonal, so AP mixes precisions (at ε = 1e-4, 525 to 653 of the 780 off-diagonal tiles are FP16). The main test. |
| E | kind 4 (3-D exponential) | E1 (β = 0.1) has decay. E2–E4 (β = 1) have nearly uniform tile norms. κ of kind 4 cannot exceed about 1.6e6 (section 5.2). |
| U | kind 2, β = 10 | The control: all tiles have almost the same norm, so AP chooses one precision for every off-diagonal tile. |

| Matrix | Kernel | β | noise | κ₁(A) |
|---|---|---|---|---|
| D1 | kind 2 | 0.1 | 1e2 | 2.5e1 |
| D2 | kind 2 | 0.1 | 1e1 | 3.7e2 |
| D3 | kind 2 | 0.1 | 1 | 4.5e3 |
| D4 | kind 2 | 0.1 | 1e-1 | 5.2e4 |
| D5 | kind 2 | 0.1 | 1e-2 | 5.7e5 |
| D6 | kind 2 | 0.1 | 1e-3 | 6.1e6 |
| D7 | kind 2 | 0.1 | 1e-4 | 6.5e7 |
| D8 | kind 2 | 0.1 | 1e-5 | 6.6e8 |
| D9 | kind 2 | 0.1 | 1e-6 | 6.8e9 |
| D10 | kind 2 | 0.1 | 1e-7 | 6.8e10 |
| E1 | kind 4 | 0.1 | 1 | 9.3e2 |
| E2 | kind 4 | 1 | 1 | 2.9e4 |
| E3 | kind 4 | 1 | 1e-1 | 2.4e5 |
| E4 | kind 4 | 1 | 1e-2 | 1.0e6 |
| U1 | kind 2 | 10 | 1e3 | 4.2e1 |
| U2 | kind 2 | 10 | 1e1 | 4.1e3 |
| U3 | kind 2 | 10 | 1e-1 | 4.9e5 |
| U4 | kind 2 | 10 | 1e-3 | 8.4e7 |
| U5 | kind 2 | 10 | 1e-5 | 8.3e9 |

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

### 4.5 Classic IR

**Family D (tile-norm decay)**

| matrix | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 2.5e1 | 2 | 2 | 2 | 1 | 1 | 2 | 1 | 1 | 1 |
| D2 | 3.7e2 | 2 | 2 | 2 | 1 | 1 | 2 | 1 | 1 | 1 |
| D3 | 4.5e3 | 3 | 3 | 2 | 2 | 1 | 3 | 2 | 1 | 1 |
| D4 | 5.2e4 | 5 | 5 | 3 | 2 | 1 | 3–4 | 2 | 1 | 2 |
| D5 | 5.7e5 | 17–19 | 16–18 | 5 | 2–3 | 2 | 6–7 | 2 | 2 | 2 |
| D6 | 6.1e6 | NC† | NC† | 21–23 | 4 | 2 | DIV | 3 | 2 | 3 |
| D7 | 6.5e7 | NC† | NC† | NC† | 11 | 3–4 | NC† | 6 | 3 | 5 |
| D8 | 6.6e8 | NC† | NC† | NC† | NC† | 13–15 | DIV† | DIV† | 4–5 | DIV |
| D9 | 6.8e9 | NC† | NC† | NC† | NC† | NC† | NC† | NC† | DIV† | NC† |
| D10 | 6.8e10 | NC† | NC† | NC† | NC† | NC† | NC† | NC† | DIV† | NC† |

- **FP16 and FP16-MAGMA** need 2–5 iterations up to κ₁ = 5.2e4 and 16–19 at 5.7e5, and do not converge beyond that.
- **AP** converges one decade further for each two decades of ε: to 6.1e6 with ε = 1e-4 (21–23 iterations), to 6.5e7 with 1e-6 (11), to 6.6e8 with 1e-8 (13–15). Where FP16 needs 17–19 iterations (D5), AP needs 5, 2–3 and 2.
- **RAP** at ε = 1e-6 and 1e-8 converges as far as AP, with fewer iterations near the limit: 6 against 11 at D7 and 4–5 against 13–15 at D8. RAP 1e-4 stops at D5, like FP16, though with 6–7 iterations against FP16's 17–19.
- **FP32** converges up to 6.5e7 (5 iterations), between AP 1e-6 and AP 1e-8.
- **No factorization** converges with classic IR at κ₁ ≥ 6.8e9.

**Family E (kind 4)**

| matrix | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 9.3e2 | 3 | 3 | 2 | 1 | 1 | 2 | 2 | 1 | 1 |
| E2 | 2.9e4 | 6 | 6 | 2 | 2 | 0 | 3 | 2 | 1 | 2 |
| E3 | 2.4e5 | NC | NC | 2 | 2 | 0 | 4 | 2 | 1 | 2 |
| E4 | 1.0e6 | DIV† | DIV† | 2 | 2 | 0 | 5 | 2 | 1 | 2 |

- **FP16 and FP16-MAGMA** do not converge from E3 (κ₁ = 2.4e5).
- **Every other factorization** converges on all four matrices in at most 5 iterations. On E2–E4, AP 1e-4 and 1e-6 put every off-diagonal tile in FP32 and match FP32; AP 1e-8 puts every tile in FP64 and needs 0 iterations.
- **RAP 1e-4** needs 3–5 iterations where AP 1e-4 needs 2. Its runtime test still runs most GEMMs in FP16 (section 4.8).

**Family U (control, uniform tile norms)**

| matrix | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|
| U1 | 4.2e1 | 2 | 2 | 2 | 1 | 1 | 2 | 1 | 1 | 1 |
| U2 | 4.1e3 | 4 | 4 | 2 | 2 | 0 | 2 | 2 | 1 | 2 |
| U3 | 4.9e5 | DIV† | DIV† | 2 | 2 | 0 | 2 | 2 | 1 | 2 |
| U4 | 8.4e7 | NC† | NC† | 8 | 8 | 0 | 8 | 8 | 1 | 8 |
| U5 | 8.3e9 | NC† | NC† | NC† | NC† | 0 | NC† | NC† | 2 | NC† |

- **FP16 and FP16-MAGMA** fail from U3 (κ₁ = 4.9e5).
- **AP and RAP at ε = 1e-4 and 1e-6** choose FP32 for every off-diagonal tile and give exactly FP32's counts. The exception is U1 at ε = 1e-4, where they choose FP16 and need FP16's 2 iterations. Adapting gains nothing here.
- **AP 1e-8** is an FP64 factorization on U2–U5 (0 iterations). RAP 1e-8 keeps most tiles in FP32 and still converges everywhere, in at most 2 iterations.

### 4.6 GMRES-IR

**Family D**

| matrix | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | 2.5e1 | 2 / 2 | 2 / 2 | 2 / 2 | 1 / 1 | 1 / 1 | 2 / 2 | 1 / 1 | 1 / 1 | 1 / 1 |
| D2 | 3.7e2 | 1–2 / 2–3 | 1 / 2 | 2 / 2 | 1 / 1 | 1 / 1 | 2 / 2 | 1 / 1 | 1 / 1 | 1 / 1 |
| D3 | 4.5e3 | 2 / 4 | 2 / 4 | 1 / 2 | 2 / 2 | 1 / 1 | 2 / 4 | 2 / 2 | 1 / 1 | 1 / 1 |
| D4 | 5.2e4 | 2 / 6 | 2 / 6 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 2 |
| D5 | 5.7e5 | 2–3 / 10–15 | 2–3 / 10–15 | 2 / 6 | 1–2 / 2–4 | 1–2 / 2–3 | 2 / 6 | 1 / 2 | 2 / 2 | 1 / 2 |
| D6 | 6.1e6 | 3 / 36† | 3 / 36† | 2 / 11–12 | 2 / 4–5 | 1 / 2 | 2–3 / 20–30 | 2 / 4 | 1 / 2 | 2 / 4 |
| D7 | 6.5e7 | 3 / 108–109† | 3 / 107–109† | 3 / 45–46† | 2 / 10 | 1 / 3 | 3 / 54–56† | 2 / 7–8 | 2 / 4 | 2 / 6 |
| D8 | 6.6e8 | 3 / 283–286† | 3 / 282–284† | 3 / 140† | 2–3 / 22–33† | 2 / 11–12 | 3 / 128–159† | 2 / 20† | 1–2 / 4–7 | 2 / 19 |
| D9 | 6.8e9 | 8 / 800† | 8 / 800† | 4 / 393–394† | 2–3 / 63–97† | 2 / 25–26† | 6–7 / 600–700† | 2 / 59–60† | 2 / 17–18† | 2 / 55–56† |
| D10 | 6.8e10 | NC† | NC† | 10 / 1000 (1 of 3)† | 2–3 / 178–277† | 2 / 77–79† | NC† | 2–3 / 171–264† | 3 / 53–59† | 2 / 161–163† |

- **Every factorization converges up to D9 (κ₁ = 6.8e9).** At D10, FP16, FP16-MAGMA and RAP 1e-4 do not converge, and AP 1e-4 converges for one right-hand side of three.
- **The cost moves into the GMRES iterations**, which grow with κ × factor error:

  | | FP16 | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
  |---|---|---|---|---|---|---|---|---|
  | D8 (κ₁ 6.6e8) | 283–286 | 140 | 22–33 | 11–12 | 128–159 | 20 | 4–7 | 19 |
  | D9 (κ₁ 6.8e9) | 800 | 393–394 | 63–97 | 25–26 | 600–700 | 59–60 | 17–18 | 55–56 |

  At D9 that is 1.1 to 2 times fewer GMRES iterations than FP16 for ε = 1e-4, 8 to 14 times fewer for ε = 1e-6, and 31 to 47 times fewer for ε = 1e-8.
- **FP16's 8 / 800 at D9** means every GMRES solve stopped at its 100-iteration limit, and the refinement still converged after 8 of them.

**Family E**

| matrix | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | 9.3e2 | 2 / 4 | 2 / 4 | 2 / 2 | 1 / 1 | 1 / 1 | 2 / 3 | 2 / 2 | 1 / 1 | 1 / 1 |
| E2 | 2.9e4 | 2 / 5–6 | 2 / 5–6 | 2 / 2 | 2 / 2 | 0 / 0 | 2 / 4 | 2 / 2 | 1 / 1 | 2 / 2 |
| E3 | 2.4e5 | 2–3 / 10–14 | 2–3 / 10–14 | 1 / 2 | 1 / 2 | 0 / 0 | 2 / 4 | 1 / 2 | 1 / 1 | 1 / 2 |
| E4 | 1.0e6 | 3 / 19–21† | 3 / 19–21† | 1 / 2 | 1 / 2 | 0 / 0 | 2 / 6 | 1 / 2 | 1 / 1 | 1 / 2 |

All factorizations converge. FP16 and FP16-MAGMA need the most GMRES iterations (19–21 at E4); the others need at most 6.

**Family U**

| matrix | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|
| U1 | 4.2e1 | 2 / 2 | 2 / 2 | 2 / 2 | 1 / 1 | 1 / 1 | 2 / 2 | 1 / 1 | 1 / 1 | 1 / 1 |
| U2 | 4.1e3 | 2 / 4 | 2 / 4 | 2 / 2 | 2 / 2 | 0 / 0 | 2 / 2 | 2 / 2 | 1 / 1 | 2 / 2 |
| U3 | 4.9e5 | 3–4 / 24–33† | 3–4 / 24–33† | 1 / 2 | 1 / 2 | 0 / 0 | 1 / 2 | 1 / 2 | 1 / 1 | 1 / 2 |
| U4 | 8.4e7 | 2–3 / 19–30† | 2–3 / 19–30† | 2 / 9 | 2 / 9 | 0 / 0 | 2 / 9 | 2 / 9 | 1 / 2 | 2 / 9 |
| U5 | 8.3e9 | 3 / 30–32† | 3 / 30–32† | 3 / 31† | 3 / 37–38† | 0 / 0 | 3 / 31† | 3 / 37–38† | 1 / 3 | 3 / 40–41† |

All factorizations converge with at most 41 GMRES iterations, even FP16 with a shift as large as the diagonal (U4, U5). The U matrices are a numerically low-rank kernel plus noise·I: apart from a few large eigenvalues, every eigenvalue equals the noise. GMRES therefore converges in a few dozen iterations with almost any preconditioner, and these runs do not separate the factorizations.

### 4.7 Where each factorization stops converging

Largest κ₁(A) at which all three right-hand sides converged, with the iterations there. On families E and U, GMRES-IR converged on every matrix for every factorization. 1.0e6 and 8.3e9 are the largest matrices of E and U, so those entries mean "converged on every matrix".

| strategy | D, classic IR | D, GMRES-IR | E, classic IR | U, classic IR |
|---|---|---|---|---|
| FP16 | 5.7e5 (17–19) | 6.8e9 (8 / 800†) | 2.9e4 (6) | 4.1e3 (4) |
| FP16-MAGMA | 5.7e5 (16–18) | 6.8e9 (8 / 800†) | 2.9e4 (6) | 4.1e3 (4) |
| AP 1e-4 | 6.1e6 (21–23) | 6.8e9 (4 / 393–394†) | 1.0e6 (2) | 8.4e7 (8) |
| AP 1e-6 | 6.5e7 (11) | 6.8e10 (2–3 / 178–277†) | 1.0e6 (2) | 8.4e7 (8) |
| AP 1e-8 | 6.6e8 (13–15) | 6.8e10 (2 / 77–79†) | 1.0e6 (0) | 8.3e9 (0) |
| RAP 1e-4 | 5.7e5 (6–7) | 6.8e9 (6–7 / 600–700†) | 1.0e6 (5) | 8.4e7 (8) |
| RAP 1e-6 | 6.5e7 (6) | 6.8e10 (2–3 / 171–264†) | 1.0e6 (2) | 8.4e7 (8) |
| RAP 1e-8 | 6.6e8 (4–5) | 6.8e10 (3 / 53–59†) | 1.0e6 (1) | 8.3e9 (2) |
| FP32 | 6.5e7 (5) | 6.8e10 (2 / 161–163†) | 1.0e6 (2) | 8.4e7 (8) |

### 4.8 Why: factor error, precision maps and breakdowns

**Factor error and precision map.** The error is ‖F − L̂L̂ᵀ‖₂/‖F‖₂ by power iteration. The map gives the tiles stored in FP64/FP32/FP16; one matrix of each family near κ₁ = 5e5.

| strategy | D (κ₁ 5.7e5) | E (κ₁ 2.4e5) | U (κ₁ 4.9e5) |
|---|---|---|---|
| FP16 | 8.7e-6, 40/0/780 | 6.2e-5, 40/0/780 | 1.1e-4, 40/0/780 |
| FP16-MAGMA | 8.8e-6, 0/40/780 | 6.2e-5, 0/40/780 | 1.1e-4, 0/40/780 |
| AP 1e-4 | 3.1e-7, 40/255/525 | 2.3e-9, 40/780/0 | 1.2e-8, 40/780/0 |
| AP 1e-6 | 3.1e-8, 40/463/317 | 2.3e-9, 40/780/0 | 1.2e-8, 40/780/0 |
| AP 1e-8 | 2.5e-9, 301/319/200 | 9.8e-17, 820/0/0 | 7.9e-17, 820/0/0 |
| RAP 1e-4 | 1.5e-6, 40/746/34 | 2.7e-6, 40/780/0 | 2.5e-8, 40/780/0 |
| RAP 1e-6 | 2.8e-8, 40/751/29 | 1.1e-8, 40/780/0 | 1.7e-8, 40/780/0 |
| RAP 1e-8 | 5.4e-9, 164/633/23 | 2.9e-10, 88/732/0 | 1.3e-10, 79/741/0 |
| FP32 | 2.9e-8, 40/780/0 | 2.3e-9, 40/780/0 | 1.2e-8, 40/780/0 |

- **AP's error follows ε** on D (3.1e-7, 3.1e-8, 2.5e-9). On E3 and U3 it uses no FP16 tiles at all.
- **RAP's error at ε = 1e-4 is five times AP's on D** (1.5e-6 against 3.1e-7), and a thousand times AP's on E3 (2.7e-6 against 2.3e-9), where both store every off-diagonal tile in FP32. RAP tests each GEMM's ‖L_ik‖·‖L_jk‖ instead of the tile's norm. A single update is much smaller than the whole tile, so many more GEMMs pass the FP16 test (91% against AP's 57% on D6, section 4.9). Each passing GEMM contributes its own FP16 error to the tile, so at a loose ε these errors add up. At ε = 1e-6 and 1e-8, only small updates pass, and RAP's error matches AP's.

**κ × factor error decides classic IR.** Each classic IR iteration multiplies the error by roughly κ × ‖E‖/‖A‖. The table gives κ₁(A) × factor error at the last matrix of family D where each factorization converged and at the first where it did not:

| strategy | last converged | first not converged |
|---|---|---|
| FP16 | 4.9 (κ₁ 5.7e5) | 53 (κ₁ 6.1e6) |
| FP16-MAGMA | 5 (κ₁ 5.7e5) | 54 (κ₁ 6.1e6) |
| AP 1e-4 | 2.2 (κ₁ 6.1e6) | 24 (κ₁ 6.5e7) |
| AP 1e-6 | 2.3 (κ₁ 6.5e7) | 24 (κ₁ 6.6e8) |
| AP 1e-8 | 4 (κ₁ 6.6e8) | 48 (κ₁ 6.8e9) |
| RAP 1e-4 | 0.86 (κ₁ 5.7e5) | 11 (κ₁ 6.1e6) |
| RAP 1e-6 | 2.2 (κ₁ 6.5e7) | 24 (κ₁ 6.6e8) |
| RAP 1e-8 | 2.9 (κ₁ 6.6e8) | 27 (κ₁ 6.8e9) |
| FP32 | 2 (κ₁ 6.5e7) | 19 (κ₁ 6.6e8) |

The threshold is between 5 and 11 for every factorization, and the same holds on E (at most 2.8 converged, at least 14.8 not) and U (at most 1.0 converged, at least 54 not). It lies above 1 because κ₁ overestimates the 2-norm condition number of the theory.

**Breakdowns and shifts.** At large κ the low-precision factor of a positive definite matrix can meet a non-positive pivot. The solver then refactorizes F + s·I with s = c·u (u is the unit roundoff of the factor; ε for AP and RAP), trying c = 2, 8, 32, … up to eight attempts. Shift s needed, relative to the unit diagonal of F (blank: no breakdown):

| D matrix | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|
| D6 | 6.1e6 | 3.9e-3 (c=8) | 3.9e-3 (c=8) |  |  |  |  |  |  |  |
| D7 | 6.5e7 | 3.9e-3 (c=8) | 3.9e-3 (c=8) | 8.0e-4 (c=8) |  |  | 8.0e-4 (c=8) |  |  |  |
| D8 | 6.6e8 | 3.9e-3 (c=8) | 3.9e-3 (c=8) | 8.0e-4 (c=8) | 3.2e-5 (c=32) |  | 8.0e-4 (c=8) | 8.0e-6 (c=8) |  |  |
| D9 | 6.8e9 | 3.9e-3 (c=8) | 3.9e-3 (c=8) | 8.0e-4 (c=8) | 3.2e-5 (c=32) | 5.1e-6 (c=512) | 3.2e-3 (c=32) | 3.2e-5 (c=32) | 3.2e-7 (c=32) | 3.1e-5 (c=512) |
| D10 | 6.8e10 | 3.9e-3 (c=8) | 3.9e-3 (c=8) | 8.0e-4 (c=8) | 3.2e-5 (c=32) | 5.1e-6 (c=512) | 3.2e-3 (c=32) | 3.2e-5 (c=32) | 1.3e-6 (c=128) | 3.1e-5 (c=512) |

| E matrix | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|
| E4 | 1.0e6 | 1.6e-2 (c=32) | 1.6e-2 (c=32) |  |  |  |  |  |  |  |

| U matrix | κ₁(A) | FP16 | FP16-MAGMA | AP 1e-4 | AP 1e-6 | AP 1e-8 | RAP 1e-4 | RAP 1e-6 | RAP 1e-8 | FP32 |
|---|---|---|---|---|---|---|---|---|---|---|
| U3 | 4.9e5 | 2.5e-1 (c=512) | 2.5e-1 (c=512) |  |  |  |  |  |  |  |
| U4 | 8.4e7 | 1.0e0 (c=2048) | 1.0e0 (c=2048) |  |  |  |  |  |  |  |
| U5 | 8.3e9 | 1.0e0 (c=2048) | 1.0e0 (c=2048) | 8.0e-4 (c=8) | 5.1e-4 (c=512) |  | 8.0e-4 (c=8) | 5.1e-4 (c=512) |  | 4.9e-4 (c=8192) |

- **No breakdown went unrescued.** The eighth and last attempt was needed once, for FP32 on U5 (c = 8192).
- **Shifts limit refinement.** A shifted factor approximates F + s·I, so the refinement also has to remove s. The factor errors above are measured against the shifted F and do not include s.
- **The FP16 shifts on U are large.** On U4 and U5, s equals the diagonal itself. GMRES-IR still converges there, for the spectral reason given in section 4.6; classic IR does not.

### 4.9 Effect of ε

AP and RAP over ε from 1e-2 to 1e-10 on two matrices of family D: D6 (κ₁ = 6.1e6) and D9 (κ₁ = 6.8e9). These runs used four A100s (guyot), and nine cases were rerun on the hexane V100. Where they overlap with sections 4.5 and 4.6, the counts are close to the H100 ones.

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

### 5.2 Condition numbers of the generated matrices (A1a)

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
