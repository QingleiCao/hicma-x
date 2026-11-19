/**
 * @file hicma_parsec_ir.c
 * @brief Mixed-precision iterative refinement with a HiCMA-X Cholesky factor
 *
 * See hicma_parsec_ir.h. Vectors are replicated on every rank: the O(N^2)
 * products (A*x with dplasma_dsymm on the CPU, M^-1*x with dplasma_dpotrs on
 * the GPUs) run on distributed N x 1 descriptors, the O(N) and O(N*m) work
 * (norms, GMRES orthogonalization) on the host of every rank.
 *
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec_ir.h"
#include <cblas.h>
#include <float.h>

#define IR_UD         1.1102230246251565e-16  /* LAPACK dlamch('E') = 2^-53 */
#define IR_XMAX       65504.0                 /* largest finite FP16 number */
#define IR_THETA      0.1                     /* headroom of the FP16 range (Higham & Pranesh, MAGMA) */
#define IR_MAX_TRIES  8                       /* factorizations per run with --ir_shift_retry */

enum { IR_CONVERGED = 0, IR_STAGNATED, IR_SLOW, IR_DIVERGED, IR_MAXIT, IR_BREAKDOWN };
static const char *ir_status_str[] = { "converged", "stagnated", "slow", "diverged", "maxit", "breakdown" };

static double ir_now(void) { MPI_Barrier(MPI_COMM_WORLD); return MPI_Wtime(); }

/* ------------------------------------------------------------------------ */
/* Options                                                                  */
/* ------------------------------------------------------------------------ */

void hicma_parsec_ir_print_help(void)
{
    printf("Iterative refinement options (testing_potrf_ir):\n"
           "  --ir                   0: one solve, no refinement; 1: classic IR (default); 2: GMRES-IR\n"
           "  --ir_maxit             refinement steps (default 30 classic, 10 GMRES-IR)\n"
           "  --gmres_tol            GMRES relative preconditioned residual (default 1e-4)\n"
           "  --gmres_maxit          GMRES iterations per refinement step (default min(N, 500))\n"
           "  --ir_scale             1: factorize mu*D^-1 A D^-1, D = diag(A)^1/2, mu = 0.1*65504/(1+c*u) (default); 0: A\n"
           "  --ir_shift             c of the diagonal shift c*u (u: unit roundoff of the factor; default 0)\n"
           "  --ir_shift_retry       1: on breakdown refactorize with c = 2, 8, 32, ... (at most %d factorizations)\n"
           "  --ir_seed              seed of the right-hand sides (default 3872)\n"
           "  --ir_rhs               0: b = A*x_true with random x_true (default); 1: random b\n"
           "  --ir_trials            right-hand sides solved with one factorization (default 1)\n"
           "  --ir_probe             power iterations estimating ||F-LL^T||_2/||F||_2 of the factored F (default 0, off)\n"
           "  --ir_condest           1: estimate kappa_1(A) and kappa_1(D^-1 A D^-1)\n"
           "  --ir_stop_slow         1: stop when the contraction stays >= 0.5 for two steps (timing runs)\n"
           "  --ir_xsave FILE        rank 0 writes the first solution (N doubles)\n"
           "  --ir_xref FILE         reference solution for the forward error with --ir_rhs 1\n\n",
           IR_MAX_TRIES);
}

int hicma_parsec_ir_parse_args(int *argc, char **argv, hicma_parsec_ir_opts_t *opt)
{
    memset(opt, 0, sizeof(*opt));
    opt->solver = 1; opt->maxit = -1; opt->gmres_tol = 1e-4; opt->gmres_maxit = -1;
    opt->scale = 1; opt->seed = 3872; opt->trials = 1;

    struct { const char *name; int type; void *target; } tab[] = {   /* type 0 int, 1 double, 2 string */
        {"ir", 0, &opt->solver}, {"ir_maxit", 0, &opt->maxit}, {"gmres_tol", 1, &opt->gmres_tol},
        {"gmres_maxit", 0, &opt->gmres_maxit}, {"ir_scale", 0, &opt->scale}, {"ir_shift", 1, &opt->shift},
        {"ir_shift_retry", 0, &opt->shift_retry}, {"ir_seed", 0, &opt->seed}, {"ir_rhs", 0, &opt->rhs},
        {"ir_trials", 0, &opt->trials}, {"ir_probe", 0, &opt->probe}, {"ir_condest", 0, &opt->condest},
        {"ir_stop_slow", 0, &opt->stop_slow}, {"ir_xsave", 2, &opt->xsave}, {"ir_xref", 2, &opt->xref},
        {NULL, 0, NULL}
    };

    int out = 1;
    for( int i = 1; i < *argc; i++ ) {
        char *arg = argv[i];
        if( 0 == strcmp(arg, "--") ) {             /* the rest belongs to PaRSEC */
            for( ; i < *argc; i++ ) argv[out++] = argv[i];
            break;
        }
        int found = -1;
        const char *value = NULL;
        if( 0 == strncmp(arg, "--", 2) ) {
            const char *name = arg + 2, *eq = strchr(name, '=');
            size_t len = eq ? (size_t)(eq - name) : strlen(name);
            for( int k = 0; NULL != tab[k].name; k++ )
                if( strlen(tab[k].name) == len && 0 == strncmp(name, tab[k].name, len) ) { found = k; break; }
            if( found >= 0 ) {
                if( eq ) value = eq + 1;
                else if( i + 1 < *argc ) value = argv[++i];
            }
        }
        if( found < 0 ) { argv[out++] = arg; continue; }
        if( NULL == value ) {
            fprintf(stderr, "Error: option '--%s' requires a value\n", tab[found].name);
            return -1;
        }
        char *end = NULL;
        if( 0 == tab[found].type )      *(int *)tab[found].target = (int)strtol(value, &end, 10);
        else if( 1 == tab[found].type ) *(double *)tab[found].target = strtod(value, &end);
        else                            { *(char **)tab[found].target = strdup(value); continue; }
        if( end == value || '\0' != *end ) {
            fprintf(stderr, "Error: invalid value '%s' for option '--%s'\n", value, tab[found].name);
            return -1;
        }
    }
    *argc = out;
    argv[out] = NULL;
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Tiles and vectors                                                        */
/* ------------------------------------------------------------------------ */

static double *ir_tile(parsec_tiled_matrix_t *A, int m, int n)
{
    parsec_data_collection_t *dc = &A->super;
    return (double *)parsec_data_copy_get_ptr(parsec_data_get_copy(dc->data_of(dc, m, n), 0));
}

static int ir_mine(parsec_tiled_matrix_t *A, int m, int n)
{
    return A->super.rank_of(&A->super, m, n) == A->super.myrank;
}

/* N x 1 vector with NB x 1 tiles on the grid of A (dpotrs and dtrmm build
 * their datatypes per collection, so B tiles need not be square) */
static void ir_vec_init(hicma_parsec_ir_t *ir, parsec_matrix_block_cyclic_t *v, const char *key)
{
    hicma_parsec_params_t *p = ir->params;
    parsec_matrix_block_cyclic_init(v, PARSEC_MATRIX_DOUBLE, PARSEC_MATRIX_TILE, p->rank,
                                    p->NB, 1, p->N, 1, 0, 0, p->N, 1, p->P, p->nodes / p->P, 1, 1, 0, 0);
    size_t bytes = (size_t)v->super.nb_local_tiles * (size_t)v->super.bsiz * sizeof(double);
    v->mat = parsec_data_allocate(bytes > 0 ? bytes : sizeof(double));
    memset(v->mat, 0, bytes > 0 ? bytes : sizeof(double));
    parsec_data_collection_set_key((parsec_data_collection_t *)v, (char *)key);
}

static void ir_vec_fini(parsec_matrix_block_cyclic_t *v)
{
    parsec_data_free(v->mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)v);
}

/* host array -> local tiles (host copies; only for descriptors no GPU touches) */
static void ir_scatter(hicma_parsec_ir_t *ir, parsec_matrix_block_cyclic_t *v, const double *x)
{
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)v;
    for( int m = 0; m < A->mt; m++ ) {
        if( !ir_mine(A, m, 0) ) continue;
        int rows = (m == A->mt - 1) ? ir->N - m * A->mb : A->mb;
        memcpy(ir_tile(A, m, 0), x + (size_t)m * A->mb, rows * sizeof(double));
    }
}

/* local tiles -> host array on every rank */
static void ir_gather(hicma_parsec_ir_t *ir, parsec_matrix_block_cyclic_t *v, double *x)
{
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)v;
    memset(x, 0, (size_t)ir->N * sizeof(double));
    for( int m = 0; m < A->mt; m++ ) {
        if( !ir_mine(A, m, 0) ) continue;
        int rows = (m == A->mt - 1) ? ir->N - m * A->mb : A->mb;
        memcpy(x + (size_t)m * A->mb, ir_tile(A, m, 0), rows * sizeof(double));
    }
    MPI_Allreduce(MPI_IN_PLACE, x, ir->N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

static double ir_norm_inf(int n, const double *x)
{
    double s = 0.0;
    for( int i = 0; i < n; i++ ) { double a = fabs(x[i]); if( a > s || a != a ) s = a; }
    return s;
}

/* ------------------------------------------------------------------------ */
/* The two operators: A*v (CPU dsymm) and M^-1*v (GPU dpotrs)               */
/* ------------------------------------------------------------------------ */

typedef struct {
    hicma_parsec_ir_t *ir;
    const double *x;
    double *y;              /* nthreads x N partial products */
} ir_av_args_t;

/* y += A(m,n) x_n and, below the diagonal, y_n += A(m,n)^T x_m, reading the
 * tile once; the diagonal tile is symmetric with its lower triangle stored */
static int ir_op_av(parsec_execution_stream_t *es, const parsec_tiled_matrix_t *desc,
                    void *data, int uplo, int m, int n, void *op_args)
{
    (void)uplo;
    ir_av_args_t *a = (ir_av_args_t *)op_args;
    const int N = a->ir->N, ld = desc->mb, NB = desc->mb;
    const int rows = (m == desc->mt - 1) ? N - m * NB : NB;
    const int cols = (n == desc->nt - 1) ? N - n * NB : NB;
    const double *T = (const double *)data;
    const double *xm = a->x + (size_t)m * NB, *xn = a->x + (size_t)n * NB;
    double *y = a->y + (size_t)es->th_id * N;
    double *ym = y + (size_t)m * NB, *yn = y + (size_t)n * NB;
    /* column by column, so the tile is read from memory once and the second
     * use of each column comes from L1; BLAS for the vectorized kernels */
    for( int j = 0; j < cols; j++ ) {
        const double *c = T + (size_t)j * ld;
        const int i0 = (m == n) ? j + 1 : 0;
        if( rows > i0 ) {
            cblas_daxpy(rows - i0, xn[j], c + i0, 1, ym + i0, 1);
            yn[j] += cblas_ddot(rows - i0, c + i0, 1, xm + i0, 1);
        }
        if( m == n ) yn[j] += c[j] * xn[j];
    }
    return 0;
}

static void ir_av_gpu(hicma_parsec_ir_t *ir, const double *in, double *out);

/* A*v with the FP64 copy of A. On one process with GPUs: two triangular
 * products on the GPU (ir_av_gpu). Otherwise on the CPUs: every rank
 * multiplies its own tiles with the replicated v and one MPI_Allreduce of N
 * doubles adds up the partial results, so no tile moves */
static void ir_av(hicma_parsec_ir_t *ir, const double *in, double *out)
{
    const int N = ir->N;
    ir->n_av++;
    if( ir->av_gpu ) { ir_av_gpu(ir, in, out); return; }
    ir_av_args_t args = { .ir = ir, .x = in, .y = ir->avbuf };
    memset(ir->avbuf, 0, (size_t)ir->nthreads * N * sizeof(double));
    parsec_apply(ir->parsec, PARSEC_MATRIX_LOWER, (parsec_tiled_matrix_t *)&ir->dcAir, ir_op_av, &args);
    for( int i = 0; i < N; i++ ) {
        double t = 0.0;
        for( int th = 0; th < ir->nthreads; th++ ) t += ir->avbuf[(size_t)th * N + i];
        out[i] = t;
    }
    MPI_Allreduce(MPI_IN_PLACE, out, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

/* Copy a host vector into dcR through a CPU task (dlacpy from the host-only
 * staging dcS), so that PaRSEC sees a new version and re-sends it to the GPU;
 * plain host stores into dcR would let the GPU reuse its previous copy. */
static void ir_to_gpu_vec(hicma_parsec_ir_t *ir, const double *in)
{
    ir_scatter(ir, &ir->dcS, in);
    dplasma_dlacpy(ir->parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&ir->dcS, (parsec_tiled_matrix_t *)&ir->dcR);
}

static void ir_from_gpu_vec(hicma_parsec_ir_t *ir, double *out)
{
    dplasma_dlacpy(ir->parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&ir->dcR, (parsec_tiled_matrix_t *)&ir->dcS);
    ir_gather(ir, &ir->dcS, out);
}

/* A v = tril(A) v + tril(A)^T v - diag(A) v, with dcAir resident on the GPU */
static void ir_av_gpu(hicma_parsec_ir_t *ir, const double *in, double *out)
{
    const int N = ir->N;
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&ir->dcAir;
    ir_scatter(ir, &ir->dcS, in);
    dplasma_dlacpy(ir->parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&ir->dcS, (parsec_tiled_matrix_t *)&ir->dcR);
    dplasma_dlacpy(ir->parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&ir->dcS, (parsec_tiled_matrix_t *)&ir->dcR2);
    dplasma_dtrmm(ir->parsec, dplasmaLeft, dplasmaLower, dplasmaNoTrans, dplasmaNonUnit, 1.0, A, (parsec_tiled_matrix_t *)&ir->dcR);
    dplasma_dtrmm(ir->parsec, dplasmaLeft, dplasmaLower, dplasmaTrans, dplasmaNonUnit, 1.0, A, (parsec_tiled_matrix_t *)&ir->dcR2);
    ir_from_gpu_vec(ir, out);
    dplasma_dlacpy(ir->parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&ir->dcR2, (parsec_tiled_matrix_t *)&ir->dcS);
    ir_gather(ir, &ir->dcS, ir->avtmp);
    for( int i = 0; i < N; i++ ) out[i] += ir->avtmp[i] - ir->d[i] * ir->d[i] * in[i];
}

/* (L L^T)^-1 v with the FP64 factor in dcAd */
static void ir_llt_solve(hicma_parsec_ir_t *ir, const double *in, double *out)
{
    ir_to_gpu_vec(ir, in);
    dplasma_dpotrs(ir->parsec, dplasmaLower, (parsec_tiled_matrix_t *)&ir->data->dcAd, (parsec_tiled_matrix_t *)&ir->dcR);
    ir_from_gpu_vec(ir, out);
}

/* M^-1 v = mu * D^-1 (L L^T)^-1 D^-1 v when scaled, (L L^T)^-1 v otherwise */
static void ir_minv(hicma_parsec_ir_t *ir, const double *in, double *out)
{
    const int N = ir->N;
    if( ir->opt.scale ) {
        for( int i = 0; i < N; i++ ) ir->tmp[i] = in[i] / ir->d[i];
        ir_llt_solve(ir, ir->tmp, out);
        for( int i = 0; i < N; i++ ) out[i] *= ir->mu / ir->d[i];
    } else {
        ir_llt_solve(ir, in, out);
    }
    ir->n_minv++;
}

/* ------------------------------------------------------------------------ */
/* Per-tile operators (parsec_apply: every local tile, on the PaRSEC cores) */
/* ------------------------------------------------------------------------ */

typedef struct {
    hicma_parsec_ir_t *ir;
    double mu, cu;        /* scale and shift c*u_eff */
    int scale;
    double *colA, *colH;  /* per-thread column sums, nthreads x N */
} ir_op_args_t;

/* F = mu*(D^-1 A D^-1 + c*u*I) when scaled, A + c*u*diag(A) otherwise; then
 * the squared Frobenius norm of the tile, counting a diagonal tile in full
 * from its lower triangle, as the generation does (matrix_compress.jdf) */
static int ir_op_preprocess(parsec_execution_stream_t *es, const parsec_tiled_matrix_t *desc,
                            void *data, int uplo, int m, int n, void *op_args)
{
    (void)es; (void)uplo;
    ir_op_args_t *a = (ir_op_args_t *)op_args;
    hicma_parsec_ir_t *ir = a->ir;
    double *T = (double *)data;
    const int ld = desc->mb, NB = desc->mb;
    const int rows = (m == desc->mt - 1) ? ir->N - m * NB : NB;
    const int cols = (n == desc->nt - 1) ? ir->N - n * NB : NB;
    const double *d = ir->d;
    double sq = 0.0;
    for( int j = 0; j < cols; j++ ) {
        const int gj = n * NB + j;
        for( int i = 0; i < rows; i++ ) {
            const int gi = m * NB + i;
            double v = T[(size_t)j * ld + i];
            if( a->scale ) v = (gi == gj) ? a->mu * (1.0 + a->cu) : a->mu * (v / (d[gi] * d[gj]));
            else if( gi == gj ) v *= 1.0 + a->cu;
            T[(size_t)j * ld + i] = v;
            if( m != n || gi > gj ) sq += (m == n ? 2.0 : 1.0) * v * v;
            else if( gi == gj ) sq += v * v;
        }
    }
    ir->params->norm_tile[n * ir->NT + m] = sq;
    return 0;
}

/* column sums of |A| and |D^-1 A D^-1| from the lower triangle (symmetric) */
static int ir_op_colsum(parsec_execution_stream_t *es, const parsec_tiled_matrix_t *desc,
                        void *data, int uplo, int m, int n, void *op_args)
{
    (void)uplo;
    ir_op_args_t *a = (ir_op_args_t *)op_args;
    hicma_parsec_ir_t *ir = a->ir;
    const double *T = (const double *)data;
    const int ld = desc->mb, NB = desc->mb;
    const int rows = (m == desc->mt - 1) ? ir->N - m * NB : NB;
    const int cols = (n == desc->nt - 1) ? ir->N - n * NB : NB;
    double *cA = a->colA + (size_t)es->th_id * ir->N;
    double *cH = a->colH + (size_t)es->th_id * ir->N;
    for( int j = 0; j < cols; j++ ) {
        const int gj = n * NB + j;
        for( int i = 0; i < rows; i++ ) {
            const int gi = m * NB + i;
            if( gi < gj ) continue;
            double v = fabs(T[(size_t)j * ld + i]), h = v / (ir->d[gi] * ir->d[gj]);
            cA[gj] += v; cH[gj] += h;
            if( gi > gj ) { cA[gi] += v; cH[gi] += h; }
        }
    }
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Setup                                                                    */
/* ------------------------------------------------------------------------ */

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
static int ir_memory_noop(parsec_data_collection_t *dc, parsec_device_module_t *device)
{
    (void)dc; (void)device;
    return PARSEC_SUCCESS;
}
#endif

static const char *ir_strategy(const hicma_parsec_params_t *p, double u_eff)
{
    if( p->adaptive_decision_runtime ) return "RAP";
    if( p->adaptive_decision ) return "AP";
    if( DENSE_MP_GPU_FP8_SP == p->kind_of_cholesky ) return "FP16-MAGMA";
    if( 0.0 == u_eff ) return "FP64";
    return u_eff > 1e-6 ? "FP16" : "FP32";
}

int hicma_parsec_ir_init(parsec_context_t *parsec, hicma_parsec_data_t *data,
                         hicma_parsec_params_t *params, hicma_parsec_ir_opts_t *opt,
                         hicma_parsec_ir_t *ir)
{
    memset(ir, 0, sizeof(*ir));
    ir->opt = *opt;
    ir->parsec = parsec; ir->data = data; ir->params = params;
    ir->N = params->N; ir->NB = params->NB; ir->NT = params->NT; ir->rank = params->rank;
    const int N = ir->N, NT = ir->NT;

    /* What the refinement supports */
    const char *why = NULL;
#if GENOMICS
    why = "a build without GENOMICS (dcAd is FP32 there)";
#else
    if( params->band_size_dense < NT || params->auto_band || params->adaptive_memory || params->sparse )
        why = "the dense layout (--band_dense >= NT, no --auto_band, --adaptive_memory or --sparse)";
    else if( DENSE_MP_GPU != params->kind_of_cholesky && DENSE_MP_GPU_FP8_ADAPTIVE != params->kind_of_cholesky
             && DENSE_MP_GPU_FP8_SP != params->kind_of_cholesky )
        why = "--kind_of_cholesky 5, 9 or 10";
    else if( params->check )
        why = "no --check (the refinement measures the factor itself, see --ir_probe)";
    else if( opt->solver < 0 || opt->solver > 2 || opt->trials < 1 || opt->scale < 0 || opt->scale > 1
             || opt->rhs < 0 || opt->rhs > 1 || opt->gmres_tol <= 0.0 || opt->shift < 0.0 )
        why = "valid --ir, --ir_trials, --ir_scale, --ir_rhs, --gmres_tol and --ir_shift values";
#endif
    if( NULL != why ) {
        if( 0 == params->rank ) fprintf(stderr, "Error: iterative refinement needs %s\n", why);
        exit(1);
    }
    if( ir->opt.maxit < 0 ) ir->opt.maxit = (2 == opt->solver) ? 10 : 30;
    if( ir->opt.gmres_maxit <= 0 ) ir->opt.gmres_maxit = N < 500 ? N : 500;

    /* Unit roundoff the shift is scaled with (Higham & Pranesh: c*u of the
     * factorization): epsilon for the adaptive kinds, else the lowest
     * precision of the static band map; FP64 factors are never shifted */
    if( params->adaptive_decision || params->adaptive_decision_runtime ) {
        ir->u_eff = params->fixedacc;
    } else {
        int fp8 = 0, hp = 0, sp = 0;
        for( int n = 0; n < NT; n++ )
            for( int m = n; m < NT; m++ ) {
                uint16_t t = params->decisions[n * NT + m];
                fp8 |= DENSE_FP8 == t;
                hp |= DENSE_HP == t;
                sp |= DENSE_SP == t;
            }
        ir->u_eff = fp8 ? EPS_FP8 : (hp ? HICMA_EPS_HP(params) : (sp ? EPS_SP : 0.0));
    }
    /* an FP32 diagonal (kind 9) bounds the factor's accuracy whatever epsilon */
    if( (params->adaptive_decision || params->adaptive_decision_runtime) && DENSE_SP == params->decisions[0]
        && ir->u_eff < EPS_SP )
        ir->u_eff = EPS_SP;
    /* the map before any factorization: the run-time kinds change it */
    ir->decisions0 = malloc((size_t)params->MT * NT * sizeof(uint16_t));
    memcpy(ir->decisions0, params->decisions, (size_t)params->MT * NT * sizeof(uint16_t));

    ir_vec_init(ir, &ir->dcR, "ir_R");
    ir_vec_init(ir, &ir->dcR2, "ir_R2");
    ir_vec_init(ir, &ir->dcS, "ir_S");

    ir->nthreads = parsec->virtual_processes[0]->nb_cores;
    size_t vbytes = (size_t)N * sizeof(double);
    ir->d = malloc(vbytes); ir->b = malloc(vbytes); ir->xtrue = malloc(vbytes); ir->x = malloc(vbytes);
    ir->r = malloc(vbytes); ir->dv = malloc(vbytes); ir->w = malloc(vbytes); ir->tmp = malloc(vbytes);
    ir->avbuf = malloc((size_t)ir->nthreads * vbytes);
    ir->avtmp = malloc(vbytes);
    ir->av_gpu = 1 == params->nodes && params->gpus > 0;
    if( params->nodes > 1 && 0 == params->rank )
        fprintf(stderr, "Warning: with several processes every M^-1 v (dplasma_dpotrs) moves tiles of the factor "
                "to the ranks owning the vector (process-grid column 0); the refinement time includes that traffic\n");

    /* The unchanged FP64 copy of A, taken before any preprocessing */
    parsec_matrix_sym_block_cyclic_init(&ir->dcAir, PARSEC_MATRIX_DOUBLE, params->rank, ir->NB, ir->NB,
                                        N, N, 0, 0, N, N, params->P, params->nodes / params->P, params->uplo);
    ir->dcAir.mat = parsec_data_allocate((size_t)ir->dcAir.super.nb_local_tiles * (size_t)ir->dcAir.super.bsiz * sizeof(double));
    parsec_data_collection_set_key((parsec_data_collection_t *)&ir->dcAir, "ir_A");
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    /* A*v runs on the GPU: pin the copy once, as dcAd is, instead of PaRSEC
     * registering and unregistering it around every taskpool */
    if( params->gpus > 0 && cudaSuccess == cudaHostRegister(ir->dcAir.mat, (size_t)ir->dcAir.super.nb_local_tiles
                                                            * (size_t)ir->dcAir.super.bsiz * sizeof(double), cudaHostRegisterDefault) ) {
        ir->dcAir.super.super.register_memory = ir_memory_noop;
        ir->dcAir.super.super.unregister_memory = ir_memory_noop;
        ir->dcAir_pinned = 1;
    }
#endif
    double t = ir_now();
    dplasma_dlacpy(parsec, dplasmaLower, (parsec_tiled_matrix_t *)&data->dcAd, (parsec_tiled_matrix_t *)&ir->dcAir);
    ir->t_copyA = ir_now() - t;

    /* D = diag(A)^1/2 */
    parsec_tiled_matrix_t *Air = (parsec_tiled_matrix_t *)&ir->dcAir;
    memset(ir->d, 0, vbytes);
    for( int k = 0; k < NT; k++ ) {
        if( !ir_mine(Air, k, k) ) continue;
        const double *T = ir_tile(Air, k, k);
        int rows = (k == NT - 1) ? N - k * ir->NB : ir->NB;
        for( int i = 0; i < rows; i++ ) ir->d[k * ir->NB + i] = T[(size_t)i * Air->mb + i];
    }
    MPI_Allreduce(MPI_IN_PLACE, ir->d, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for( int i = 0; i < N; i++ ) {
        if( !(ir->d[i] > 0.0) ) {
            if( 0 == ir->rank ) fprintf(stderr, "Error: a_%d%d = %g is not positive\n", i, i, ir->d[i]);
            exit(1);
        }
        ir->d[i] = sqrt(ir->d[i]);
    }

    /* ||A||_inf = ||A||_1 and ||D^-1 A D^-1||_1 */
    ir_op_args_t args = { .ir = ir };
    args.colA = calloc((size_t)ir->nthreads * 2 * N, sizeof(double));
    args.colH = args.colA + (size_t)ir->nthreads * N;
    parsec_apply(parsec, PARSEC_MATRIX_LOWER, Air, ir_op_colsum, &args);
    for( int th = 1; th < ir->nthreads; th++ )
        for( int i = 0; i < N; i++ ) {
            args.colA[i] += args.colA[(size_t)th * N + i];
            args.colH[i] += args.colH[(size_t)th * N + i];
        }
    MPI_Allreduce(MPI_IN_PLACE, args.colA, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, args.colH, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    ir->normA_inf = ir_norm_inf(N, args.colA);
    ir->normH_one = ir_norm_inf(N, args.colH);
    free(args.colA);

    if( 0 == ir->rank )
        printf("IR setup: strategy= %s solver= %s scale= %d u_eff= %.3e ||A||_inf= %.6e ||H||_1= %.6e "
               "maxit= %d gmres_tol= %.1e gmres_maxit= %d trials= %d t_copyA= %.4f\n",
               ir_strategy(params, ir->u_eff), 0 == opt->solver ? "none" : (1 == opt->solver ? "ir" : "gmres-ir"),
               opt->scale, ir->u_eff, ir->normA_inf, ir->normH_one, ir->opt.maxit, opt->gmres_tol,
               ir->opt.gmres_maxit, opt->trials, ir->t_copyA);
    return 0;
}

/* ------------------------------------------------------------------------ */
/* Factorization                                                            */
/* ------------------------------------------------------------------------ */

/* Back to the unchanged A: FP64 spans and no GPU copies of the old factor */
static void ir_restore(hicma_parsec_ir_t *ir)
{
    parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&ir->data->dcAd;
    hicma_parsec_change_nb_elts(ir->parsec, dplasmaLower, A, sizeof(double));
    dplasma_dlacpy(ir->parsec, dplasmaLower, (parsec_tiled_matrix_t *)&ir->dcAir, A);
}

static int ir_cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Scale and shift dcAd, recompute the tile norms the precision decisions use */
static void ir_preprocess(hicma_parsec_ir_t *ir, int first)
{
    hicma_parsec_params_t *p = ir->params;
    const int NT = ir->NT;
    static int norms_changed = 0;

    ir->mu = ir->opt.scale ? IR_THETA * IR_XMAX / (1.0 + ir->c * ir->u_eff) : 1.0;
    if( !ir->opt.scale && 0.0 == ir->c && !norms_changed ) return;

    ir_op_args_t args = { .ir = ir, .mu = ir->mu, .cu = ir->c * ir->u_eff, .scale = ir->opt.scale };
    memset(p->norm_tile, 0, (size_t)NT * NT * sizeof(double));
    parsec_apply(ir->parsec, PARSEC_MATRIX_LOWER, (parsec_tiled_matrix_t *)&ir->data->dcAd, ir_op_preprocess, &args);
    MPI_Allreduce(MPI_IN_PLACE, p->norm_tile, NT * NT, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double sq = 0.0;
    for( int n = 0; n < NT; n++ )
        for( int m = n; m < NT; m++ ) {
            sq += p->norm_tile[n * NT + m];
            p->norm_tile[n * NT + m] = sqrt(p->norm_tile[n * NT + m]);
        }
    p->norm_global = sqrt(sq);
    norms_changed = ir->opt.scale || ir->c > 0.0;

    /* Decision ratios norm_tile*NT/norm_global of the off-diagonal tiles: a
     * tile is FP16 below eps/u_h and FP32 below eps/u_s (decision_make_comp) */
    if( first && 0 == ir->rank && NT > 1 ) {
        int cnt = NT * (NT - 1) / 2, k = 0;
        double *rr = malloc(cnt * sizeof(double));
        for( int n = 0; n < NT; n++ )
            for( int m = n + 1; m < NT; m++ ) rr[k++] = p->norm_tile[n * NT + m] * NT / p->norm_global;
        qsort(rr, cnt, sizeof(double), ir_cmp_double);
        printf("IR decision ratios: min= %.3e q10= %.3e q25= %.3e q50= %.3e q75= %.3e q90= %.3e max= %.3e "
               "(FP16 below eps/%.2e, FP32 below eps/%.2e)\n",
               rr[0], rr[cnt / 10], rr[cnt / 4], rr[cnt / 2], rr[(3 * cnt) / 4], rr[(9 * cnt) / 10], rr[cnt - 1],
               HICMA_EPS_HP(p), EPS_SP);
        free(rr);
    }
}

int hicma_parsec_ir_factorize(hicma_parsec_ir_t *ir, starsh_params_t *params_kernel,
                              hicma_parsec_matrix_analysis_t *analysis, int run)
{
    hicma_parsec_params_t *p = ir->params;
    const int NT = ir->NT;
    double c = ir->u_eff > 0.0 ? ir->opt.shift : 0.0, t;

    if( run > 0 ) ir->t_copyA = 0.0;
    ir->t_preproc = ir->t_pre = ir->t_factor = 0.0;
    ir->attempts = 0; ir->factor_fp64 = 0; ir->breakdown = 0;

    for( ;; ) {
        if( run > 0 || ir->attempts > 0 ) {
            t = ir_now(); ir_restore(ir); ir->t_copyA += ir_now() - t;
        }
        ir->c = c;
        memcpy(p->decisions, ir->decisions0, (size_t)p->MT * NT * sizeof(uint16_t));
        t = ir_now(); ir_preprocess(ir, 0 == ir->attempts); ir->t_preproc += ir_now() - t;
        t = ir_now(); hicma_parsec_matrix_pre_analysis(ir->parsec, ir->data, p, params_kernel, analysis); ir->t_pre += ir_now() - t;

        if( p->adaptive_decision_runtime ) {
            memset(p->decisions_gemm_first, 0, (size_t)p->MT * p->NT * sizeof(uint16_t));
            memset(p->decisions_gemm_last, 0, (size_t)p->MT * p->NT * sizeof(uint16_t));
        }
        p->info = 0;
        if( p->gpus > 0 && NULL != p->info_gpu ) memset(p->info_gpu, 0, NT * sizeof(int));

        t = ir_now(); hicma_parsec_potrf(ir->parsec, ir->data, p, analysis); ir->t_factor += ir_now() - t;
        ir->attempts++;
        if( 0 == p->info ) break;
        if( !ir->opt.shift_retry || ir->attempts >= IR_MAX_TRIES || 0.0 == ir->u_eff ) {
            ir->breakdown = 1;
            if( 0 == ir->rank )
                fprintf(stderr, "IR: the factorization broke down (info = %d, c = %g) after %d attempt(s)\n",
                        p->info, c, ir->attempts);
            break;
        }
        if( 0 == ir->rank )
            fprintf(stderr, "IR: breakdown (info = %d) with c = %g, refactorizing with c = %g\n",
                    p->info, c, 0.0 == c ? 2.0 : 4.0 * c);
        c = (0.0 == c) ? 2.0 : 4.0 * c;
    }

    /* Precision map (owner ranks hold the run-time changes) and GEMM counts */
    get_decisions(p->decisions, (size_t)p->MT * p->NT, (parsec_tiled_matrix_t *)&ir->data->dcAd);
    ir->dp_tiles = ir->sp_tiles = ir->hp_tiles = 0;
    for( int n = 0; n < NT; n++ )
        for( int m = n; m < NT; m++ ) {
            uint16_t d = p->decisions[n * NT + m];
            ir->dp_tiles += DENSE_DP == d;
            ir->sp_tiles += DENSE_SP == d;
            ir->hp_tiles += DENSE_HP == d || DENSE_FP8 == d;
        }
    /* GEMMs are counted with --verbose 2 or more only; hicma_parsec_potrf has
     * then reduced them into the first slot */
    for( int i = 0; i <= NB_DECISIONS; i++ )
        ir->gemms[i] = p->verbose > 1 ? (long)p->nb_gemms[i * p->counter_stride] : -1;
    return ir->breakdown ? p->info : 0;
}

/* ------------------------------------------------------------------------ */
/* Measurements of the factor                                               */
/* ------------------------------------------------------------------------ */

/* Hager-Higham estimate of ||op||_1 for a symmetric operator */
static double ir_onenorm_est(hicma_parsec_ir_t *ir, void (*op)(hicma_parsec_ir_t *, const double *, double *))
{
    const int N = ir->N;
    double *x = malloc(N * sizeof(double)), *y = malloc(N * sizeof(double));
    double *s = malloc(N * sizeof(double)), *z = malloc(N * sizeof(double)), est = 0.0;
    for( int i = 0; i < N; i++ ) x[i] = 1.0 / N;
    for( int it = 0; it < 5; it++ ) {
        op(ir, x, y);                                   /* y = B x */
        double e = 0.0;
        for( int i = 0; i < N; i++ ) e += fabs(y[i]);
        if( it > 0 && e <= est ) break;
        est = e;
        for( int i = 0; i < N; i++ ) s[i] = y[i] >= 0.0 ? 1.0 : -1.0;
        op(ir, s, z);                                   /* z = B^T sign(y), B symmetric */
        int j = 0;
        double ztx = 0.0;
        for( int i = 0; i < N; i++ ) { ztx += z[i] * x[i]; if( fabs(z[i]) > fabs(z[j]) ) j = i; }
        if( it > 0 && fabs(z[j]) <= ztx ) break;
        memset(x, 0, N * sizeof(double));
        x[j] = 1.0;
    }
    /* Higham's extra vector against cancellation */
    for( int i = 0; i < N; i++ ) x[i] = (i % 2 ? -1.0 : 1.0) * (1.0 + (double)i / (N > 1 ? N - 1 : 1));
    op(ir, x, y);
    double alt = 0.0;
    for( int i = 0; i < N; i++ ) alt += fabs(y[i]);
    alt *= 2.0 / (3.0 * N);
    free(x); free(y); free(s); free(z);
    return est > alt ? est : alt;
}

static void ir_op_ainv(hicma_parsec_ir_t *ir, const double *in, double *out) { ir_minv(ir, in, out); }

static void ir_op_hinv(hicma_parsec_ir_t *ir, const double *in, double *out)   /* H^-1 = D A^-1 D */
{
    double *t = malloc(ir->N * sizeof(double));
    for( int i = 0; i < ir->N; i++ ) t[i] = in[i] * ir->d[i];
    ir_minv(ir, t, out);
    for( int i = 0; i < ir->N; i++ ) out[i] *= ir->d[i];
    free(t);
}

/* F*v for the factored matrix F: mu*(D^-1 A D^-1 + c*u*I) or A + c*u*diag(A) */
static void ir_fv(hicma_parsec_ir_t *ir, const double *in, double *out)
{
    const int N = ir->N;
    const double cu = ir->c * ir->u_eff;
    if( ir->opt.scale ) {
        for( int i = 0; i < N; i++ ) ir->tmp[i] = in[i] / ir->d[i];
        ir_av(ir, ir->tmp, out);
        for( int i = 0; i < N; i++ ) out[i] = ir->mu * (out[i] / ir->d[i] + cu * in[i]);
    } else {
        ir_av(ir, in, out);
        for( int i = 0; i < N; i++ ) out[i] += cu * ir->d[i] * ir->d[i] * in[i];
    }
}

/* ||F - L L^T||_2 / ||F||_2 by power iteration (both symmetric) */
static double ir_probe_factor(hicma_parsec_ir_t *ir, int iters)
{
    const int N = ir->N;
    double *v = malloc(N * sizeof(double)), *fv = malloc(N * sizeof(double)), *lv = malloc(N * sizeof(double));
    double nE = 0.0, nF = 0.0;
    for( int pass = 0; pass < 2; pass++ ) {
        unsigned long s = 12345;
        for( int i = 0; i < N; i++ ) { s = s * 6364136223846793005UL + 1442695040888963407UL; v[i] = (double)(s >> 11) / 9007199254740992.0 - 0.5; }
        double nrm = cblas_dnrm2(N, v, 1), lambda = 0.0;
        cblas_dscal(N, 1.0 / nrm, v, 1);
        for( int it = 0; it < iters; it++ ) {
            ir_fv(ir, v, fv);
            if( 0 == pass ) {                 /* E v = F v - L (L^T v) */
                ir_to_gpu_vec(ir, v);
                dplasma_dtrmm(ir->parsec, dplasmaLeft, dplasmaLower, dplasmaTrans, dplasmaNonUnit, 1.0,
                              (parsec_tiled_matrix_t *)&ir->data->dcAd, (parsec_tiled_matrix_t *)&ir->dcR);
                dplasma_dtrmm(ir->parsec, dplasmaLeft, dplasmaLower, dplasmaNoTrans, dplasmaNonUnit, 1.0,
                              (parsec_tiled_matrix_t *)&ir->data->dcAd, (parsec_tiled_matrix_t *)&ir->dcR);
                ir_from_gpu_vec(ir, lv);
                for( int i = 0; i < N; i++ ) fv[i] -= lv[i];
            }
            lambda = cblas_dnrm2(N, fv, 1);
            if( !(lambda > 0.0) ) break;
            for( int i = 0; i < N; i++ ) v[i] = fv[i] / lambda;
        }
        if( 0 == pass ) nE = lambda; else nF = lambda;
    }
    free(v); free(fv); free(lv);
    return nF > 0.0 ? nE / nF : NAN;
}

/* ------------------------------------------------------------------------ */
/* Refinement                                                               */
/* ------------------------------------------------------------------------ */

typedef struct {
    int status, steps, gmres_its;
    double berr, ferr, berr0;
} ir_result_t;

/* r = b - A x; returns the normwise backward error, prints the step */
static double ir_residual(hicma_parsec_ir_t *ir, int step, int gmres_its, double dx, double ratio,
                          double pbe, double *rnorm, double *xnorm, double *ferr)
{
    const int N = ir->N;
    ir_av(ir, ir->x, ir->w);
    for( int i = 0; i < N; i++ ) ir->r[i] = ir->b[i] - ir->w[i];
    *rnorm = ir_norm_inf(N, ir->r);
    *xnorm = ir_norm_inf(N, ir->x);
    double berr = *rnorm / (ir->normA_inf * *xnorm + ir_norm_inf(N, ir->b));
    *ferr = NAN;
    const double *xs = (0 == ir->opt.rhs) ? ir->xtrue : ir->xref;
    if( NULL != xs ) {
        double e = 0.0;
        for( int i = 0; i < N; i++ ) { double a = fabs(ir->x[i] - xs[i]); if( a > e || a != a ) e = a; }
        *ferr = e / ir_norm_inf(N, xs);
    }
    if( 0 == ir->rank )
        printf("IR step %3d: berr= %.3e ferr= %.3e dx= %.3e ratio= %.3e gmres= %d pbe= %.3e minv= %ld av= %ld\n",
               step, berr, *ferr, dx, ratio, gmres_its, pbe, ir->n_minv, ir->n_av);
    return berr;
}

/* Status after a correction d (norm dn) of x; ratios ||d_i||/||d_i-1|| */
static int ir_check_step(hicma_parsec_ir_t *ir, double dn, double xn, double ratio, int *up, int *slow, int *stag)
{
    if( !isfinite(dn) || !isfinite(xn) ) return IR_DIVERGED;
    if( dn <= IR_UD * xn ) *stag = 1;
    if( isfinite(ratio) ) {
        *up = ratio > 1.0 ? *up + 1 : 0;
        *slow = ratio >= 0.5 ? *slow + 1 : 0;
    }
    if( *up >= 2 ) return IR_DIVERGED;
    if( ir->opt.stop_slow && *slow >= 2 ) return IR_SLOW;
    return -1;
}

/* GMRES on M^-1 A d = M^-1 r, x0 = 0; returns iterations, d in ir->dv */
static int ir_gmres(hicma_parsec_ir_t *ir, const double *r, double *pbe)
{
    const int N = ir->N, mmax = ir->opt.gmres_maxit, ldh = mmax + 1;
    if( NULL == ir->Hm ) {   /* small arrays at full size; the basis V grows on demand */
        ir->Hm = calloc((size_t)ldh * mmax, sizeof(double));
        ir->cs = malloc(mmax * sizeof(double)); ir->sn = malloc(mmax * sizeof(double));
        ir->g = malloc((mmax + 1) * sizeof(double)); ir->yk = malloc((mmax + 1) * sizeof(double));
    }
    if( ir->Vcols < 2 ) {
        ir->Vcols = mmax + 1 < 32 ? mmax + 1 : 32;
        ir->V = malloc((size_t)N * ir->Vcols * sizeof(double));
    }
    double *V = ir->V, *H = ir->Hm, *h2 = ir->yk;
    double *s = ir->dv;
    ir_minv(ir, r, s);
    double beta = cblas_dnrm2(N, s, 1), anorm = 0.0;
    *pbe = NAN;
    if( !isfinite(beta) ) { for( int i = 0; i < N; i++ ) ir->dv[i] = NAN; return 0; }   /* -> diverged */
    if( !(beta > 0.0) ) { memset(ir->dv, 0, N * sizeof(double)); return 0; }
    for( int i = 0; i < N; i++ ) V[i] = s[i] / beta;
    memset(ir->g, 0, (mmax + 1) * sizeof(double));
    ir->g[0] = beta;
    int k, kused = 0;
    double res = beta;
    for( k = 0; k < mmax; k++ ) {
        if( k + 2 > ir->Vcols ) {
            int cols = 2 * ir->Vcols < mmax + 1 ? 2 * ir->Vcols : mmax + 1;
            double *nv = realloc(ir->V, (size_t)N * cols * sizeof(double));
            if( NULL == nv ) { fprintf(stderr, "Error: no memory for %d GMRES vectors of %d\n", cols, N); exit(1); }
            ir->V = V = nv; ir->Vcols = cols;
        }
        double *w = V + (size_t)(k + 1) * N, *hk = H + (size_t)k * ldh;
        ir_av(ir, V + (size_t)k * N, ir->tmp);
        ir_minv(ir, ir->tmp, w);
        /* CGS2 */
        cblas_dgemv(CblasColMajor, CblasTrans, N, k + 1, 1.0, V, N, w, 1, 0.0, hk, 1);
        cblas_dgemv(CblasColMajor, CblasNoTrans, N, k + 1, -1.0, V, N, hk, 1, 1.0, w, 1);
        cblas_dgemv(CblasColMajor, CblasTrans, N, k + 1, 1.0, V, N, w, 1, 0.0, h2, 1);
        cblas_dgemv(CblasColMajor, CblasNoTrans, N, k + 1, -1.0, V, N, h2, 1, 1.0, w, 1);
        for( int j = 0; j <= k; j++ ) hk[j] += h2[j];
        double hn = cblas_dnrm2(N, w, 1);
        hk[k + 1] = hn;
        double cn = cblas_dnrm2(k + 2, hk, 1);
        if( cn > anorm ) anorm = cn;
        /* Givens rotations */
        for( int j = 0; j < k; j++ ) {
            double t = ir->cs[j] * hk[j] + ir->sn[j] * hk[j + 1];
            hk[j + 1] = -ir->sn[j] * hk[j] + ir->cs[j] * hk[j + 1];
            hk[j] = t;
        }
        double den = hypot(hk[k], hk[k + 1]);
        ir->cs[k] = den > 0.0 ? hk[k] / den : 1.0;
        ir->sn[k] = den > 0.0 ? hk[k + 1] / den : 0.0;
        hk[k] = den; hk[k + 1] = 0.0;
        ir->g[k + 1] = -ir->sn[k] * ir->g[k];
        ir->g[k] = ir->cs[k] * ir->g[k];
        kused = k + 1;
        res = fabs(ir->g[k + 1]);
        if( res <= ir->opt.gmres_tol * beta || !(hn > 0.0) || !isfinite(res) ) break;
        cblas_dscal(N, 1.0 / hn, w, 1);
    }
    /* y = R^-1 g, d = V y */
    double *y = ir->yk;
    for( int i = kused - 1; i >= 0; i-- ) {
        double t = ir->g[i];
        for( int j = i + 1; j < kused; j++ ) t -= H[(size_t)j * ldh + i] * y[j];
        y[i] = t / H[(size_t)i * ldh + i];
    }
    cblas_dgemv(CblasColMajor, CblasNoTrans, N, kused, 1.0, V, N, y, 1, 0.0, ir->dv, 1);
    /* preconditioned backward error (Higham & Pranesh): ||s - A~ d||/(||A~|| ||d|| + ||s||) */
    *pbe = res / (anorm * cblas_dnrm2(kused, y, 1) + beta);
    return kused;
}

static ir_result_t ir_refine(hicma_parsec_ir_t *ir)
{
    const int N = ir->N;
    ir_result_t res = { .status = IR_MAXIT };
    double t = ir_now();
    ir_minv(ir, ir->b, ir->x);
    ir->t_solve0 = ir_now() - t;
    t = ir_now();

    double rn, xn, ferr, dn_prev = NAN, dx = NAN, ratio = NAN, pbe = NAN;
    int up = 0, slow = 0, stag = 0, its = 0, pending = -1;
    for( int i = 0; ; i++ ) {
        double berr = ir_residual(ir, i, its, dx, ratio, pbe, &rn, &xn, &ferr);
        if( 0 == i ) res.berr0 = berr;
        res.berr = berr; res.ferr = ferr; res.steps = i;
        if( !isfinite(rn) || !isfinite(xn) ) { res.status = IR_DIVERGED; break; }
        if( rn <= sqrt((double)N) * IR_UD * ir->normA_inf * xn ) { res.status = IR_CONVERGED; break; }
        if( stag ) { res.status = IR_STAGNATED; break; }
        if( pending >= 0 ) { res.status = pending; break; }   /* diverged or slow, and not converged */
        if( 0 == ir->opt.solver || i >= ir->opt.maxit ) { res.status = slow >= 2 ? IR_SLOW : IR_MAXIT; break; }
        if( 2 == ir->opt.solver ) {
            its = ir_gmres(ir, ir->r, &pbe);
            res.gmres_its += its;
        } else {
            ir_minv(ir, ir->r, ir->dv);
        }
        double dn = ir_norm_inf(N, ir->dv);
        for( int k = 0; k < N; k++ ) ir->x[k] += ir->dv[k];
        ratio = (i > 0) ? dn / dn_prev : NAN;
        dx = dn / ir_norm_inf(N, ir->x);
        dn_prev = dn;
        pending = ir_check_step(ir, dn, ir_norm_inf(N, ir->x), ratio, &up, &slow, &stag);
    }
    ir->t_refine = ir_now() - t;
    return res;
}

int hicma_parsec_ir_solve(hicma_parsec_ir_t *ir, int run)
{
    hicma_parsec_params_t *p = ir->params;
    const int N = ir->N, NT = ir->NT;
    int failed = 0;

    ir->t_s2d = 0.0;
    ir->kappaA = ir->kappaH = ir->probe_err = NAN;
    if( !ir->breakdown ) {
        /* The factor to FP64 in place; then FP64 spans and no stale FP32-sized
         * GPU copies, which DPLASMA's trsm would overrun */
        int nondp = 0;
        parsec_tiled_matrix_t *A = (parsec_tiled_matrix_t *)&ir->data->dcAd;
        for( int n = 0; n < NT; n++ )
            for( int m = n; m < NT; m++ )
                if( ir_mine(A, m, n) && DENSE_DP != p->decisions[n * NT + m] ) nondp = 1;
        MPI_Allreduce(MPI_IN_PLACE, &nondp, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        double t = ir_now();
        if( nondp ) {
            hicma_parsec_convert_s2d(ir->parsec, ir->data, p);
            hicma_parsec_change_nb_elts(ir->parsec, dplasmaLower, A, sizeof(double));
        }
        ir->t_s2d = ir_now() - t;
        ir->factor_fp64 = 1;

        if( ir->opt.condest ) {
            ir->kappaA = ir->normA_inf * ir_onenorm_est(ir, ir_op_ainv);
            ir->kappaH = ir->normH_one * ir_onenorm_est(ir, ir_op_hinv);
        }
        if( ir->opt.probe > 0 ) ir->probe_err = ir_probe_factor(ir, ir->opt.probe);
    }

    if( NULL != ir->opt.xref && NULL == ir->xref ) {
        ir->xref = malloc(N * sizeof(double));
        int ok = 1;
        if( 0 == ir->rank ) {
            FILE *f = fopen(ir->opt.xref, "rb");
            ok = NULL != f && (size_t)N == fread(ir->xref, sizeof(double), N, f);
            if( f ) fclose(f);
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if( !ok ) { if( 0 == ir->rank ) fprintf(stderr, "Error: cannot read %d doubles from %s\n", N, ir->opt.xref); exit(1); }
        MPI_Bcast(ir->xref, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    for( int trial = 0; trial < ir->opt.trials; trial++ ) {
        /* Right-hand side: b = A*x_true, or b random */
        dplasma_dplrnt(ir->parsec, 0, (parsec_tiled_matrix_t *)&ir->dcS, (unsigned long long)ir->opt.seed + trial);
        ir_gather(ir, &ir->dcS, 0 == ir->opt.rhs ? ir->xtrue : ir->b);
        if( 0 == ir->opt.rhs ) ir_av(ir, ir->xtrue, ir->b);
        ir->n_minv = ir->n_av = 0;

        ir_result_t res = { .status = IR_BREAKDOWN, .berr = NAN, .ferr = NAN, .berr0 = NAN };
        ir->t_solve0 = ir->t_refine = 0.0;
        if( !ir->breakdown ) res = ir_refine(ir);
        failed |= IR_CONVERGED != res.status;
        if( !(res.berr <= ir->berr_max) ) ir->berr_max = res.berr;   /* NaN stays */

        double total = ir->t_copyA + ir->t_preproc + ir->t_pre + ir->t_factor + ir->t_s2d + ir->t_solve0 + ir->t_refine;
        if( 0 == ir->rank ) {
            printf("IR_summary run= %d trial= %d N= %d NB= %d kind= %d strategy= %s solver= %s eps= %.2e scale= %d "
                   "c= %g attempts= %d tiles_dp= %d tiles_sp= %d tiles_hp= %d gemm_dp= %ld gemm_sp= %ld gemm_hp= %ld "
                   "status= %s steps= %d gmres_its= %d minv= %ld av= %ld berr0= %.3e berr= %.3e ferr= %.3e "
                   "kappaA= %.3e kappaH= %.3e probe= %.3e t_copyA= %.4f t_preproc= %.4f t_pre= %.4f t_factor= %.4f "
                   "t_s2d= %.4f t_solve0= %.4f t_refine= %.4f t_total= %.4f\n",
                   run, trial, N, ir->NB, p->kind_of_cholesky, ir_strategy(p, ir->u_eff),
                   0 == ir->opt.solver ? "none" : (1 == ir->opt.solver ? "ir" : "gmres-ir"),
                   (p->adaptive_decision || p->adaptive_decision_runtime) ? p->fixedacc : 0.0,
                   ir->opt.scale, ir->c, ir->attempts, ir->dp_tiles, ir->sp_tiles, ir->hp_tiles,
                   ir->gemms[DENSE_DP], ir->gemms[DENSE_SP], ir->gemms[DENSE_HP],
                   ir_status_str[res.status], res.steps, res.gmres_its, ir->n_minv, ir->n_av, res.berr0, res.berr, res.ferr,
                   ir->kappaA, ir->kappaH, ir->probe_err, ir->t_copyA, ir->t_preproc, ir->t_pre, ir->t_factor,
                   ir->t_s2d, ir->t_solve0, ir->t_refine, total);
            printf("result_accuracy= %.6e\n", res.berr);
            fflush(stdout);
        }
        if( 0 == trial && 0 == run && NULL != ir->opt.xsave && 0 == ir->rank && !ir->breakdown ) {
            FILE *f = fopen(ir->opt.xsave, "wb");
            if( NULL == f || (size_t)N != fwrite(ir->x, sizeof(double), N, f) )
                fprintf(stderr, "Error: cannot write %s\n", ir->opt.xsave);
            if( f ) fclose(f);
        }
    }
    return failed;
}

void hicma_parsec_ir_fini(hicma_parsec_ir_t *ir)
{
    ir_vec_fini(&ir->dcR); ir_vec_fini(&ir->dcR2); ir_vec_fini(&ir->dcS);
    free(ir->avbuf); free(ir->avtmp); free(ir->decisions0);
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
    if( ir->dcAir_pinned ) cudaHostUnregister(ir->dcAir.mat);
#endif
    parsec_data_free(ir->dcAir.mat);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&ir->dcAir);
    free(ir->d); free(ir->b); free(ir->xtrue); free(ir->x); free(ir->r); free(ir->dv); free(ir->w); free(ir->tmp);
    free(ir->xref); free(ir->V); free(ir->Hm); free(ir->cs); free(ir->sn); free(ir->g); free(ir->yk);
    free(ir->opt.xsave); free(ir->opt.xref);
}
