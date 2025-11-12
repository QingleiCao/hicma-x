/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

/**
 * @file testing_hgemm.c
 * @brief Test program for half-precision GEMM operations using CBLAS
 * 
 * This program benchmarks half-precision matrix-matrix multiplication (GEMM)
 * operations using the CBLAS library with __fp16 data type. It performs
 * C = alpha * A * B + beta * C where A, B, and C are matrices.
 * 
 * Usage:
 *   ./testing_hgemm m           # Square matrices of size m x m
 *   ./testing_hgemm m n k       # Rectangular matrices A(m x k), B(k x n), C(m x n)
 * 
 * Features:
 * - Half-precision (__fp16) matrix operations
 * - Performance benchmarking with multiple runs
 * - GFLOP/s calculation and reporting
 * - Memory allocation and cleanup
 */

#include "cblas.h"
#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
#include "time.h"

/**
 * @brief Main function for half-precision GEMM testing
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return 0 on success, 1 on failure
 */
int main (int argc, char ** argv) {
    struct timeval tstart, tend;
    double t, gflops;

    /* Matrix dimensions */
    int m, n, k;
    
    /* Parse command line arguments for matrix dimensions */
    if( argc == 2 ) {
        /* Square matrix case: A(m x m), B(m x m), C(m x m) */
        m = atoi(argv[1]);
        n = atoi(argv[1]);
        k = atoi(argv[1]);
    } else if( argc == 4 ) {
        /* Rectangular matrix case: A(m x k), B(k x n), C(m x n) */
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    } else {
        /* Invalid number of arguments */
        printf("Usage: ./testing_hgemm m [n k]\n");
        printf("  m: number of rows in matrix A and C\n");
        printf("  n: number of columns in matrix B and C (default: m)\n");
        printf("  k: number of columns in A and rows in B (default: m)\n");
        printf("Examples:\n");
        printf("  ./testing_hgemm 1024        # Square matrices 1024x1024\n");
        printf("  ./testing_hgemm 1024 512 256 # A(1024x256), B(256x512), C(1024x512)\n");
        return 1;
    }

    /* Validate matrix dimensions */
    if (m <= 0 || n <= 0 || k <= 0) {
        fprintf(stderr, "Error: Matrix dimensions must be positive integers\n");
        return 1;
    }

    /* Leading dimensions for matrix storage */
    int lda = m;  /* Leading dimension of matrix A */
    int ldb = k;  /* Leading dimension of matrix B */
    int ldc = m;  /* Leading dimension of matrix C */

    /* Allocate memory for matrices A, B, and C */
    __fp16* A = malloc(sizeof(__fp16) * m * k);
    __fp16* B = malloc(sizeof(__fp16) * k * n);
    __fp16* C = malloc(sizeof(__fp16) * m * n);

    /* Check if memory allocation was successful */
    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for matrices\n");
        /* Clean up any successfully allocated memory */
        if (A) free(A);
        if (B) free(B);
        if (C) free(C);
        return 1;
    }

    /* Initialize matrix A with ones */
    for(int i = 0; i < m*k; i++){
        A[i] = (__fp16)1.0;
    }
    
    /* Initialize matrix B with ones */
    for(int i = 0; i < k*n; i++){
        B[i] = (__fp16)1.0;
    }
    
    /* Initialize matrix C with ones */
    for(int i = 0; i < m*n; i++){
        C[i] = (__fp16)1.0;
    }

    /* GEMM operation parameters */
    __fp16 alpha = (__fp16)1.0;  /* Scaling factor for A * B */
    __fp16 beta = (__fp16)0.0;   /* Scaling factor for C */

    /* Print test configuration */
    printf("Half-precision GEMM Test Configuration:\n");
    printf("  Matrix A: %d x %d (leading dimension: %d)\n", m, k, lda);
    printf("  Matrix B: %d x %d (leading dimension: %d)\n", k, n, ldb);
    printf("  Matrix C: %d x %d (leading dimension: %d)\n", m, n, ldc);
    printf("  Alpha: %f, Beta: %f\n", (float)alpha, (float)beta);
    printf("  Total operations: %d\n", m * n * k * 2);
    printf("  Starting benchmark with 3 runs...\n\n");

    /* Perform GEMM operation multiple times for benchmarking */
    for( int i = 0; i < 3; i++ ) {
        /* Start timing */
        gettimeofday(&tstart, NULL);
        
        /* Execute half-precision GEMM: C = alpha * A * B + beta * C */
        ss_fjcblas_gemm_r16(
                CblasRowMajor,    /* Matrix storage order */
                CblasNoTrans,     /* Don't transpose matrix A */
                CblasNoTrans,     /* Don't transpose matrix B */
                m,                /* Number of rows in A and C */
                k,                /* Number of columns in A and rows in B */
                n,                /* Number of columns in B and C */
                alpha,            /* Scaling factor for A * B */
                (const __fp16 *)A, lda,  /* Matrix A and its leading dimension */
                (const __fp16 *)B, ldb,  /* Matrix B and its leading dimension */
                beta,             /* Scaling factor for C */
                C, ldc);          /* Matrix C and its leading dimension */
        
        /* End timing */
        gettimeofday(&tend, NULL);

        /* Calculate execution time in seconds */
        t = (tend.tv_sec - tstart.tv_sec) + (tend.tv_usec - tstart.tv_usec) / 1000000.0;
        
        /* Calculate performance in GFLOP/s */
        /* Each element of C requires k multiply-add operations, so total operations = m * n * k * 2 */
        gflops = (double)m * n * k * 2.0 / t / 1.0e9;
        
        /* Print results for this run */
        printf("Run %d: %d %d %d : %.6f seconds %.3f Gflop/s\n", 
               i + 1, m, n, k, t, gflops);
    }

    /* Print summary statistics */
    printf("\nBenchmark completed successfully.\n");
    printf("Matrix dimensions: A(%dx%d), B(%dx%d), C(%dx%d)\n", m, k, k, n, m, n);

    /* Clean up allocated memory */
    free(A);
    free(B);
    free(C);

    return 0;
}
