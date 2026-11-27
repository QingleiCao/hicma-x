/**
 * @file testing_potrf_ir.c
 * @brief Cholesky factorization in mixed precision followed by iterative
 *        refinement to FP64 accuracy (classic IR or GMRES-IR)
 *
 * Every run factorizes (kinds 5, 9 or 10 in the dense layout), converts the
 * factor to FP64 and solves --ir_trials right-hand sides with refinement.
 * One summary line per run and trial (IR_summary ...) gives the precision
 * map, the refinement steps, the errors and the time of every phase.
 *
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec_ir.h"

int main(int argc, char **argv)
{
    hicma_parsec_params_t params;
    starsh_params_t params_kernel;
    hicma_parsec_data_t data;
    hicma_parsec_matrix_analysis_t analysis = {0};
    hicma_parsec_ir_opts_t opt;
    hicma_parsec_ir_t ir;

#if GENOMICS
    fprintf(stderr, "testing_potrf_ir does not support GENOMICS builds\n");
    exit(1);
#endif

    for( int i = 1; i < argc; i++ )
        if( 0 == strcmp(argv[i], "--help") || 0 == strcmp(argv[i], "-h") ) hicma_parsec_ir_print_help();
    if( 0 != hicma_parsec_ir_parse_args(&argc, argv, &opt) ) return EXIT_FAILURE;

    parsec_context_t *parsec = hicma_parsec_init(argc, argv, &params, &params_kernel, &data);
    if( NULL == parsec ) {
        fprintf(stderr, "Failed to initialize HiCMA PaRSEC\n");
        return EXIT_FAILURE;
    }

    hicma_parsec_matrix_generation(parsec, &data, &params, &params_kernel);
    hicma_parsec_ir_init(parsec, &data, &params, &opt, &ir);

    int ret = EXIT_SUCCESS;
    for( int run = 0; run < params.nruns; run++ ) {
        if( 0 != hicma_parsec_ir_factorize(&ir, &params_kernel, &analysis, run) ) ret = EXIT_FAILURE;
        if( 0 != hicma_parsec_ir_solve(&ir, run) ) ret = EXIT_FAILURE;
    }

    /* the summary of hicma_parsec_fini reports the worst final backward error */
    params.result_accuracy = ir.berr_max;
    hicma_parsec_matrix_post_analysis(parsec, &data, &params, &params_kernel, &analysis);
    hicma_parsec_ir_fini(&ir);
    if( 0 == params.rank )
        printf("Iterative refinement testing completed %s.\n", EXIT_SUCCESS == ret ? "successfully" : "with a failure");
    hicma_parsec_fini(parsec, argc, argv, &params, &params_kernel, &data, &analysis);
    return ret;
}
