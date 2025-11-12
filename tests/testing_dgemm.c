/**
 * @file testing_dgemm.c
 * @brief Test program for double precision general matrix-matrix multiplication (DGEMM)
 * 
 * This program benchmarks the performance of BLAS DGEMM operation by:
 * - Allocating matrices A, B, and C with proper memory management
 * - Initializing them with deterministic test patterns for reproducibility
 * - Running multiple iterations of DGEMM for statistical accuracy
 * - Measuring and reporting performance in GFlop/s with detailed timing
 * - Providing comprehensive error handling and memory cleanup
 * 
 * The DGEMM operation computes: C = alpha * A * B + beta * C
 * where A is m x k, B is k x n, and C is m x n
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

/* Type definitions for better maintainability */
#define FLOAT   double
#define GEMM    cblas_dgemm

/* Forward declaration of BLAS DGEMM function */
void GEMM(CBLAS_ORDER order, CBLAS_TRANSPOSE transa, CBLAS_TRANSPOSE transb, 
           int m, int n, int k, FLOAT alpha, const FLOAT *a, int lda, 
           const FLOAT *b, int ldb, FLOAT beta, FLOAT *c, int ldc);

/**
 * @brief Validates command line arguments and matrix dimensions
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @param m Pointer to store matrix dimension m
 * @param n Pointer to store matrix dimension n
 * @param k Pointer to store matrix dimension k
 * @return 0 if validation passes, non-zero otherwise
 */
int validateArguments(int argc, char **argv, int *m, int *n, int *k) {
    if (argc == 2) {
        /* Single dimension: m = n = k */
        *m = *n = *k = atoi(argv[1]);
    } else if (argc == 4) {
        /* Three dimensions: m, n, k */
        *m = atoi(argv[1]);
        *n = atoi(argv[2]);
        *k = atoi(argv[3]);
    } else {
        /* Invalid number of arguments - show usage */
        printf("Usage: %s m [n k]\n", argv[0]);
        printf("  m: matrix dimension (required)\n");
        printf("  n: matrix dimension (optional, defaults to m)\n");
        printf("  k: matrix dimension (optional, defaults to m)\n");
        printf("Example: %s 1024\n", argv[0]);
        printf("Example: %s 1024 512 2048\n", argv[0]);
        return 1;
    }
    
    /* Validate matrix dimensions are positive */
    if (*m <= 0 || *n <= 0 || *k <= 0) {
        printf("Error: Matrix dimensions must be positive integers\n");
        return 1;
    }
    
    return 0;
}

/**
 * @brief Allocates memory for matrices A, B, and C with error handling
 * 
 * @param m Matrix dimension m
 * @param n Matrix dimension n
 * @param k Matrix dimension k
 * @param a Pointer to store allocated matrix A
 * @param b Pointer to store allocated matrix B
 * @param c Pointer to store allocated matrix C
 * @return 0 if allocation succeeds, non-zero otherwise
 */
int allocateMatrices(int m, int n, int k, FLOAT **a, FLOAT **b, FLOAT **c) {
    /* Allocate memory for matrix A (m x k) */
    *a = (FLOAT *)malloc(m * k * sizeof(FLOAT));
    if (*a == NULL) {
        printf("Error: Out of memory for A matrix (%d x %d)\n", m, k);
        return 1;
    }
    
    /* Allocate memory for matrix B (k x n) */
    *b = (FLOAT *)malloc(k * n * sizeof(FLOAT));
    if (*b == NULL) {
        printf("Error: Out of memory for B matrix (%d x %d)\n", k, n);
        free(*a);  /* Clean up previously allocated memory */
        return 1;
    }
    
    /* Allocate memory for matrix C (m x n) */
    *c = (FLOAT *)malloc(m * n * sizeof(FLOAT));
    if (*c == NULL) {
        printf("Error: Out of memory for C matrix (%d x %d)\n", m, n);
        free(*a);  /* Clean up previously allocated memory */
        free(*b);
        return 1;
    }
    
    return 0;
}

/**
 * @brief Initializes matrices A and B with deterministic test patterns
 * 
 * The patterns ensure reproducible results and provide good numerical properties
 * for testing matrix multiplication algorithms.
 * 
 * @param m Matrix dimension m
 * @param n Matrix dimension n
 * @param k Matrix dimension k
 * @param a Matrix A to initialize (m x k)
 * @param b Matrix B to initialize (k x n)
 */
void initializeMatrices(int m, int n, int k, FLOAT *a, FLOAT *b) {
    /* Initialize matrix A with test pattern: (i + j + 1.0) / (m * k) */
    /* This pattern ensures all elements are in [0, 1] range for numerical stability */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            a[i * k + j] = (FLOAT)(i + j + 1.0) / (FLOAT)(m * k);
        }
    }
    
    /* Initialize matrix B with test pattern: (i + j + 1.0) / (k * n) */
    /* Similar pattern for consistency and numerical stability */
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            b[i * n + j] = (FLOAT)(i + j + 1.0) / (FLOAT)(k * n);
        }
    }
}

/**
 * @brief Performs DGEMM benchmark with timing and performance measurement
 * 
 * @param m Matrix dimension m
 * @param n Matrix dimension n
 * @param k Matrix dimension k
 * @param a Matrix A (m x k)
 * @param b Matrix B (k x n)
 * @param c Matrix C (m x n) - result matrix
 * @param alpha Scaling factor for A * B
 * @param beta Scaling factor for C
 * @param nb_runs Number of benchmark runs
 */
void runDGEMMBenchmark(int m, int n, int k, FLOAT *a, FLOAT *b, FLOAT *c,
                       FLOAT alpha, FLOAT beta, int nb_runs) {
    struct timeval tstart, tend;
    double t, gflops;
    
    printf("Testing DGEMM with matrix dimensions: %d x %d x %d\n", m, n, k);
    printf("Number of runs: %d\n", nb_runs);
    printf("Format: m n k : time(seconds) performance(GFlop/s)\n");
    printf("DGEMM operation: C = %.1f * A * B + %.1f * C\n", alpha, beta);
    
    /* Run DGEMM multiple times for benchmarking and statistical accuracy */
    for (int i = 0; i < nb_runs; i++) {
        /* Start timing */
        gettimeofday(&tstart, NULL);
        
        /* Perform DGEMM operation: C = alpha * A * B + beta * C */
        /* Using CblasNoTrans for no transpose on both A and B matrices */
        GEMM(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, m, b, k, beta, c, m);
        
        /* End timing */
        gettimeofday(&tend, NULL);
        
        /* Calculate elapsed time in seconds with microsecond precision */
        t = (tend.tv_sec - tstart.tv_sec) + 
            (tend.tv_usec - tstart.tv_usec) / 1000000.0;
        
        /* Calculate performance in GFlop/s */
        /* DGEMM performs 2*m*n*k floating point operations */
        gflops = (double)m * n * k * 2.0 / t / 1.0e9;
        
        /* Print results for this run */
        printf("Run %d: %d %d %d : %.6f seconds %.3f GFlop/s\n", 
               i + 1, m, n, k, t, gflops);
    }
    
    printf("DGEMM testing completed successfully.\n");
}

/**
 * @brief Main function - entry point of the DGEMM testing program
 * 
 * This program provides a comprehensive benchmark for BLAS DGEMM operations,
 * measuring performance in GFlop/s across multiple runs for statistical accuracy.
 * 
 * Command line usage:
 *   Single dimension: ./testing_dgemm <size>
 *   Three dimensions: ./testing_dgemm <m> <n> <k>
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return Exit status (0 for success, non-zero for failure)
 */
int main(int argc, char **argv)
{
    FLOAT *a, *b, *c;
    FLOAT alpha = 1.0;      /* Scaling factor for A * B */
    FLOAT beta = 1.0;       /* Scaling factor for C */
    const int nb_runs = 3;  /* Number of benchmark runs for statistical accuracy */
    int m, n, k;            /* Matrix dimensions */
    
    /* Validate command line arguments and matrix dimensions */
    if (validateArguments(argc, argv, &m, &n, &k) != 0) {
        return EXIT_FAILURE;
    }
    
    /* Allocate memory for matrices with proper error handling */
    if (allocateMatrices(m, n, k, &a, &b, &c) != 0) {
        return EXIT_FAILURE;
    }
    
    /* Initialize matrices with deterministic test patterns */
    initializeMatrices(m, n, k, a, b);
    
    /* Run DGEMM benchmark with timing and performance measurement */
    runDGEMMBenchmark(m, n, k, a, b, c, alpha, beta, nb_runs);
    
    /* Clean up allocated memory */
    free(a);
    free(b);
    free(c);
    
    return EXIT_SUCCESS;
}
