/**
 * @file hicma_parsec_ir.h
 * @brief Mixed-precision iterative refinement (classic IR and GMRES-IR) with
 *        the factor of a HiCMA-X Cholesky factorization (kinds 5, 9 and 10).
 *
 * The factorization runs on a preprocessed copy of A: optionally scaled to
 * F = mu*(D^-1 A D^-1 + c*u_eff*I) with D = diag(a_ii)^(1/2), as in
 * Higham & Pranesh (SISC 2021, Alg. 3.1) and MAGMA's dshposv, or shifted by
 * c*u_eff*diag(a_ii) when not scaled. The refinement keeps an unchanged FP64
 * copy of A for the residuals and applies the factor, converted to FP64, as
 * M^-1 = mu*D^-1 (L L^T)^-1 D^-1. Classic IR follows LAPACK dsposv; GMRES-IR
 * is Higham & Pranesh Alg. 4.1 with u_r = u (left-preconditioned GMRES in FP64).
 *
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#ifndef HICMA_PARSEC_IR_H
#define HICMA_PARSEC_IR_H

#include "hicma_parsec.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Options of the refinement; the driver takes them out of argv (--ir*, --gmres*) */
typedef struct hicma_parsec_ir_opts_s {
    int    solver;       /**< --ir: 0 factorize and solve once, 1 classic IR (default), 2 GMRES-IR */
    int    maxit;        /**< --ir_maxit: refinement steps (default 30 classic, 10 GMRES-IR) */
    double gmres_tol;    /**< --gmres_tol: relative preconditioned residual of GMRES (default 1e-4) */
    int    gmres_maxit;  /**< --gmres_maxit: GMRES iterations per step (default min(N, 500)) */
    int    scale;        /**< --ir_scale: 1 (default) factorize mu*D^-1 A D^-1, 0 factorize A */
    double shift;        /**< --ir_shift: initial c of the diagonal shift c*u_eff (default 0) */
    int    shift_retry;  /**< --ir_shift_retry: on breakdown retry with c = 2, 8, 32, ... (default 0) */
    int    seed;         /**< --ir_seed: seed of the right-hand sides (default 3872) */
    int    rhs;          /**< --ir_rhs: 0 b = A*x_true (default), 1 random b */
    int    trials;       /**< --ir_trials: right-hand sides per factorization (default 1) */
    int    probe;        /**< --ir_probe: power iterations for ||F - LL^T||_2/||F||_2 (default 0, off) */
    int    condest;      /**< --ir_condest: 1 estimates kappa_1(A) and kappa_1(D^-1 A D^-1) */
    int    stop_slow;    /**< --ir_stop_slow: 1 stops when the contraction stays >= 0.5 (timing runs) */
    char  *xsave;        /**< --ir_xsave FILE: rank 0 writes the first solution (binary doubles) */
    char  *xref;         /**< --ir_xref FILE: reference solution for the forward error of --ir_rhs 1 */
} hicma_parsec_ir_opts_t;

/** State of the refinement for one process */
typedef struct hicma_parsec_ir_s {
    hicma_parsec_ir_opts_t opt;
    parsec_context_t *parsec;
    hicma_parsec_data_t *data;
    hicma_parsec_params_t *params;

    int N, NB, NT, rank;
    parsec_matrix_sym_block_cyclic_t dcAir; /**< unchanged FP64 copy of A (lower) */
    parsec_matrix_block_cyclic_t dcR;       /**< right-hand side of the GPU solves */
    parsec_matrix_block_cyclic_t dcR2;      /**< second GPU vector of A*v */
    parsec_matrix_block_cyclic_t dcS;       /**< host staging of dcR, and the random vectors */
    double *avbuf;      /**< per-thread partial products of A*v, nthreads x N */
    double *avtmp;      /**< second half of A*v on the GPU */
    int av_gpu;         /**< 1: A*v with two DPLASMA trmm on the GPU (one process) */
    int dcAir_pinned;   /**< dcAir registered with cudaHostRegister */
    uint16_t *decisions0; /**< precision map before the first factorization */
    double berr_max;    /**< largest final backward error so far (NaN if any) */

    double *d;          /**< sqrt(a_ii) of the original A, replicated */
    double normA_inf;   /**< ||A||_inf = ||A||_1 */
    double normH_one;   /**< ||D^-1 A D^-1||_1 */
    double u_eff;       /**< unit roundoff the shift is scaled with */
    double c;           /**< shift multiplier of the current factor */
    double mu;          /**< scale of the current factor (1 when not scaled) */
    int attempts;       /**< factorizations of the current run */
    int factor_fp64;    /**< 1 once the current factor is FP64 in dcAd */
    int breakdown;      /**< 1 when the current run's factorization failed */
    int nthreads;

    /* replicated vectors */
    double *b, *xtrue, *x, *r, *dv, *w, *tmp, *xref;
    double *V, *Hm, *cs, *sn, *g, *yk;  /**< GMRES workspace */
    int Vcols;

    /* counters and times of the current trial / run */
    long n_minv, n_av;
    double t_copyA, t_preproc, t_pre, t_factor, t_s2d, t_solve0, t_refine;
    int dp_tiles, sp_tiles, hp_tiles;
    long gemms[NB_DECISIONS+1];      /**< GEMMs per precision (--verbose 2 or more; -1 otherwise) */
    double kappaA, kappaH, probe_err;
} hicma_parsec_ir_t;

/**
 * @brief Take the --ir* and --gmres* options out of argv
 * @return 0 on success, -1 on a malformed option
 */
int hicma_parsec_ir_parse_args(int *argc, char **argv, hicma_parsec_ir_opts_t *opt);

/** @brief Print the refinement options (--help) */
void hicma_parsec_ir_print_help(void);

/**
 * @brief Check the options, allocate the vectors and keep an FP64 copy of A
 *
 * Call after hicma_parsec_matrix_generation and before the pre-analysis.
 * @return 0 on success; refuses unsupported setups with exit(1)
 */
int hicma_parsec_ir_init(parsec_context_t *parsec, hicma_parsec_data_t *data,
                         hicma_parsec_params_t *params, hicma_parsec_ir_opts_t *opt,
                         hicma_parsec_ir_t *ir);

/**
 * @brief Factorize: restore A (later runs), preprocess, pre-analysis and
 *        hicma_parsec_potrf, retrying with a larger shift on breakdown
 * @return 0 on success, the POTRF info otherwise
 */
int hicma_parsec_ir_factorize(hicma_parsec_ir_t *ir, starsh_params_t *params_kernel,
                              hicma_parsec_matrix_analysis_t *analysis, int run);

/**
 * @brief Convert the factor to FP64 and solve every trial with refinement
 * @return 0 when every trial met the stopping test, 1 otherwise
 */
int hicma_parsec_ir_solve(hicma_parsec_ir_t *ir, int run);

/** @brief Free the vectors and the copy of A */
void hicma_parsec_ir_fini(hicma_parsec_ir_t *ir);

#ifdef __cplusplus
}
#endif

#endif /* HICMA_PARSEC_IR_H */
