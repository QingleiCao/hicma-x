#include "hicma_parsec.h"
#include "dplasma/tests/common.h"

#include <errno.h>
#include <sys/time.h>

static int parse_percentage(const char *text, double *value)
{
    char *end = NULL;
    double parsed;

    errno = 0;
    parsed = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' ||
        !isfinite(parsed) || parsed < 0.0 || parsed > 100.0) {
        return -1;
    }
    *value = parsed;
    return 0;
}

static int parse_portion_arguments(int *argc, char ***argv,
                                   double *portion_dp,
                                   double *portion_sp,
                                   double *portion_hp)
{
    char **args = *argv;
    int write_index = 1;

    *portion_dp = 20.0;
    *portion_sp = 30.0;
    *portion_hp = 50.0;

    for (int read_index = 1; read_index < *argc; read_index++) {
        char *argument = args[read_index];
        const char *value_text = NULL;
        double *target = NULL;

        if (strcmp(argument, "--help") == 0) {
            printf("Static TRMM precision-map options:\n"
                   "  --portion_dp PERCENT   Random off-diagonal DP portion "
                   "(default: 20)\n"
                   "  --portion_sp PERCENT   Random off-diagonal SP portion "
                   "(default: 30)\n"
                   "  --portion_hp PERCENT   Random off-diagonal HP portion "
                   "(default: 50)\n"
                   "  --adaptive_decision 0  Use the band-size decision map\n"
                   "  --adaptive_decision 1  Use the randomized decision map\n\n");
            args[write_index++] = argument;
            continue;
        } else if (strcmp(argument, "--portion_dp") == 0) {
            target = portion_dp;
        } else if (strcmp(argument, "--portion_sp") == 0) {
            target = portion_sp;
        } else if (strcmp(argument, "--portion_hp") == 0) {
            target = portion_hp;
        } else if (strncmp(argument, "--portion_dp=", 13) == 0) {
            target = portion_dp;
            value_text = argument + 13;
        } else if (strncmp(argument, "--portion_sp=", 13) == 0) {
            target = portion_sp;
            value_text = argument + 13;
        } else if (strncmp(argument, "--portion_hp=", 13) == 0) {
            target = portion_hp;
            value_text = argument + 13;
        } else {
            args[write_index++] = argument;
            continue;
        }

        if (value_text == NULL) {
            if (++read_index >= *argc) {
                fprintf(stderr, "%s requires a percentage value\n", argument);
                return -1;
            }
            value_text = args[read_index];
        }
        if (parse_percentage(value_text, target) != 0) {
            fprintf(stderr, "Invalid percentage for %s: %s\n",
                    argument, value_text);
            return -1;
        }
    }

    args[write_index] = NULL;
    *argc = write_index;

    if (fabs(*portion_dp + *portion_sp + *portion_hp - 100.0) > 1.0e-9) {
        fprintf(stderr,
                "--portion_dp + --portion_sp + --portion_hp must equal 100%% "
                "(received %.12g%%)\n",
                *portion_dp + *portion_sp + *portion_hp);
        return -1;
    }
    return 0;
}

static uint64_t decision_rng_next(uint64_t *state)
{
    uint64_t value = *state;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    return value * UINT64_C(2685821657736338717);
}

static int initialize_random_A_decisions(hicma_parsec_params_t *params,
                                         double portion_dp,
                                         double portion_sp,
                                         double portion_hp,
                                         uint64_t seed)
{
    const int nt = params->NT;
    const size_t off_diagonal_tiles = (size_t)nt * (size_t)(nt - 1) / 2;
    size_t count_dp = (size_t)llround(off_diagonal_tiles * portion_dp / 100.0);
    size_t count_sp = (size_t)llround(off_diagonal_tiles * portion_sp / 100.0);
    uint16_t *choices;
    size_t position = 0;

    (void)portion_hp;

    if (count_dp > off_diagonal_tiles) {
        count_dp = off_diagonal_tiles;
    }
    if (count_sp > off_diagonal_tiles - count_dp) {
        count_sp = off_diagonal_tiles - count_dp;
    }

    choices = (uint16_t *)malloc(off_diagonal_tiles * sizeof(uint16_t));
    if (off_diagonal_tiles != 0 && choices == NULL) {
        return -1;
    }

    for (; position < count_dp; position++) {
        choices[position] = DENSE_DP;
    }
    for (; position < count_dp + count_sp; position++) {
        choices[position] = DENSE_SP;
    }
    for (; position < off_diagonal_tiles; position++) {
        choices[position] = DENSE_HP;
    }

    for (size_t i = off_diagonal_tiles; i > 1; i--) {
        const size_t j = (size_t)(decision_rng_next(&seed) % i);
        const uint16_t temporary = choices[i - 1];
        choices[i - 1] = choices[j];
        choices[j] = temporary;
    }

    position = 0;
    for (int m = 0; m < nt; m++) {
        params->decisions[m * nt + m] = DENSE_DP;
        for (int n = 0; n < m; n++) {
            params->decisions[n * nt + m] = choices[position++];
        }
    }

    free(choices);
    return 0;
}

static int initialize_band_A_decisions(hicma_parsec_params_t *params)
{
    const int nt = params->NT;
    const int band_dp = params->band_size_dense_dp;
    const int band_sp = params->band_size_dense_sp;
    const int band_hp = params->band_size_dense_hp;
    const int band_dense = params->band_size_dense;

    if (band_dp < 0 || band_sp < band_dp || band_hp < band_sp ||
        band_dense < band_hp) {
        if (params->rank == 0) {
            fprintf(stderr,
                    "Band sizes must satisfy 0 <= DP <= SP <= HP <= dense; "
                    "received %d, %d, %d, %d\n",
                    band_dp, band_sp, band_hp, band_dense);
        }
        return -1;
    }

    for (int m = 0; m < nt; m++) {
        for (int n = 0; n <= m; n++) {
            const int distance = m - n;
            uint16_t decision;

            if (distance < band_dp) {
                decision = DENSE_DP;
            } else if (distance < band_sp) {
                decision = DENSE_SP;
            } else if (distance < band_hp) {
                decision = DENSE_HP;
            } else if (distance < band_dense) {
                /* TRMM has no FP8 path, so the remaining dense band is HP. */
                decision = DENSE_HP;
            } else {
                decision = DENSE_HP;
            }
            params->decisions[n * nt + m] = decision;
        }
    }
    return 0;
}

static int initialize_decision_maps(hicma_parsec_params_t *params,
                                    double portion_dp,
                                    double portion_sp,
                                    double portion_hp)
{
    int rc;

    if (params->adaptive_decision == 0) {
        rc = initialize_band_A_decisions(params);
    } else {
        rc = initialize_random_A_decisions(params,
                                           portion_dp,
                                           portion_sp,
                                           portion_hp,
                                           UINT64_C(0x9e3779b97f4a7c15));
    }
    if (rc != 0) {
        return rc;
    }

    for (int n = 0; n < params->KT; n++) {
        for (int m = 0; m < params->NT; m++) {
            params->decisionsB[n * params->NT + m] = DENSE_HP;
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    parsec_context_t *parsec;
    hicma_parsec_params_t params = {0};
    hicma_parsec_data_t data = {0};
    const dplasma_enum_t side = dplasmaLeft;
    const dplasma_enum_t uplo = dplasmaLower;
    const dplasma_enum_t trans = dplasmaNoTrans;
    const dplasma_enum_t diag = dplasmaNonUnit;
    const unsigned long long Aseed = 3872;
    const unsigned long long Bseed = 2873;
    const double alpha = 3.5;
    double portion_dp, portion_sp, portion_hp;
    int ret = 0;

    if (parse_portion_arguments(&argc, &argv,
                                &portion_dp, &portion_sp, &portion_hp) != 0) {
        return 1;
    }

    parse_arguments(&argc, &argv, &params);
    params.adaptive_memory = 1;
    hicma_parsec_params_init(&params, argv);

    if (params.MT != params.NT) {
        if (params.rank == 0) {
            fprintf(stderr,
                    "Static TRMM requires M == N so the A decision-map stride "
                    "matches the square A descriptor\n");
        }
        ret = 1;
        goto free_parameters;
    }
    if (initialize_decision_maps(&params,
                                 portion_dp, portion_sp, portion_hp) != 0) {
        ret = 1;
        goto free_parameters;
    }

    parsec = hicma_parsec_setup_parsec(argc, argv, &params);
    hicma_parsec_params_print_initial(&params);

#if !HAVE_HP_CPU
    if (params.nruns > 0 && params.gpus == 0) {
        if (params.rank == 0) {
            fprintf(stderr,
                    "Static TRMM stores B in HP and therefore requires an "
                    "enabled GPU when HAVE_HP_CPU=0. Use --nruns 0 for a "
                    "CPU-only allocation/free test.\n");
        }
        hicma_parsec_cleanup_parsec(parsec, &params);
        ret = 1;
        goto free_parameters;
    }
#endif

    const int M = params.M;
    const int K = params.K;
    const int MB = params.MB;
    const int NB = params.NB;
    const int KB = params.KB;
    const int rank = params.rank;
    const int nodes = params.nodes;
    const int P = params.P;

    assert(MB == NB);
    assert(MB == KB);

    parsec_matrix_sym_block_cyclic_t dcA;
    parsec_matrix_sym_block_cyclic_init(&dcA, PARSEC_MATRIX_DOUBLE,
            rank, MB, NB, M, M, 0, 0,
            M, M, P, nodes / P, uplo);
    dcA.mat = NULL;
    parsec_data_collection_set_key((parsec_data_collection_t *)&dcA, "dcA_static");

    parsec_matrix_block_cyclic_t dcB;
    parsec_matrix_block_cyclic_init(&dcB, PARSEC_MATRIX_DOUBLE,
            PARSEC_MATRIX_TILE, rank, MB, KB, M, K, 0, 0,
            M, K, P, nodes / P,
            params.KP, params.KQ, 0, 0);
    dcB.mat = NULL;
    parsec_data_collection_set_key((parsec_data_collection_t *)&dcB, "dcB_static");

#if (defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)) && GPU_BUFFER_ONCE
    gpu_temporay_buffer_init(&data, MB, NB, 0, params.kind_of_cholesky);
#endif

    SYNC_TIME_START();
    hicma_parsec_memory_allocation_dense_decision(
            parsec, dplasmaLower, (parsec_tiled_matrix_t *)&dcA,
            params.decisions, 1, 1, Aseed);
    hicma_parsec_memory_allocation_dense_decision(
            parsec, dplasmaUpperLower, (parsec_tiled_matrix_t *)&dcB,
            params.decisionsB, 0, 0, Bseed);
    SYNC_TIME_PRINT(rank, ("Static mixed-precision allocation and initialization\n"));

    if (rank == 0 && params.verbose > 0) {
        printf("A decision mode: %s",
               params.adaptive_decision == 0 ? "band" : "random off-diagonal");
        if (params.adaptive_decision != 0) {
            printf(" (DP %.2f%%, SP %.2f%%, HP %.2f%%)",
                   portion_dp, portion_sp, portion_hp);
        }
        printf("; B precision: HP\n");
    }
    if (params.verbose > 9) {
        print_decisions(&params, params.decisions, uplo,
                        params.NT, params.NT);
        print_decisions(&params, params.decisionsB, dplasmaUpperLower,
                        params.NT, params.KT);
    }

    const double flops = FLOPS_DTRMM(side, (DagDouble_t)M,
                                     (DagDouble_t)K);
    for (int run = 0; run < params.nruns; run++) {
        struct timeval start, end;

#if defined(PARSEC_HAVE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        gettimeofday(&start, NULL);
        if (hicma_parsec_trmm(parsec, side, uplo, trans, diag,
                              alpha, (parsec_tiled_matrix_t *)&dcA,
                              (parsec_tiled_matrix_t *)&dcB,
                              &data, &params) != 0) {
            ret = 1;
        }
#if defined(PARSEC_HAVE_MPI)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        gettimeofday(&end, NULL);

        const double elapsed = (end.tv_sec - start.tv_sec) +
                               (end.tv_usec - start.tv_usec) / 1.0e6;
        const double tflops = elapsed > 0.0 ? flops * 1.0e-12 / elapsed : 0.0;
        if (rank == 0) {
            printf("Static TRMM run %d/%d: %.6f s, %.3f Tflop/s "
                   "(nodes=%d gpus=%d P=%d Q=%d MB=%d M=%d K=%d)\n",
                   run + 1, params.nruns, elapsed, tflops,
                   params.nodes, params.gpus, params.P, params.Q,
                   MB, M, K);
        }
    }

    hicma_parsec_memory_free_dense_decision(
            parsec, dplasmaUpperLower,
            (parsec_tiled_matrix_t *)&dcB, &params);
    hicma_parsec_memory_free_dense_decision(
            parsec, dplasmaLower,
            (parsec_tiled_matrix_t *)&dcA, &params);

#if (defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)) && GPU_BUFFER_ONCE
    gpu_temporay_buffer_fini(&data, params.kind_of_cholesky);
#endif

    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcB);
    parsec_tiled_matrix_destroy((parsec_tiled_matrix_t *)&dcA);
    hicma_parsec_cleanup_parsec(parsec, &params);

free_parameters:
    free(params.rank_array);
    free(params.op_band);
    free(params.op_offband);
    free(params.op_path);
    free(params.op_offpath);
    free(params.gather_time);
    free(params.gather_time_tmp);
    free(params.decisions);
    free(params.decisions_send);
    free(params.decisions_gemm_gpu);
    free(params.decisionsB);
    free(params.norm_tile);
    free(params.norm_tileB);
    return ret;
}
