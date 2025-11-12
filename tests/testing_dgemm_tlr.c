/**
 * @file testing_dgemm_tlr.c
 * @brief Test program for double precision general matrix-matrix multiplication (DGEMM) using TLR format
 * 
 * This program benchmarks the performance of DGEMM operations using both dense and TLR (Tensor Low Rank)
 * matrix formats. It compares traditional BLAS DGEMM with HCORE TLR DGEMM for various matrix sizes
 * and rank configurations.
 * 
 * The program supports:
 * - Dense matrix DGEMM using standard BLAS
 * - TLR matrix DGEMM using HCORE library
 * - Performance comparison between both approaches
 * - Configurable matrix dimensions and rank parameters
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"

// Use work array while using HCORE functions.
int use_scratch = 1;

#define FLOAT   double
#define GEMM    dgemm_
void GEMM(char *, char *, int *, int *, int *, FLOAT *, FLOAT *, int *, FLOAT *, int *, FLOAT *, FLOAT *, int *);

#define LOOP    1

/**
 * @brief Validates command line arguments and matrix dimensions
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @param type Pointer to store matrix type (0: dense, 1: TLR)
 * @param m Pointer to store matrix dimension
 * @param tlr_rk Pointer to store TLR rank (if applicable)
 * @param maxrank Pointer to store maximum rank (if applicable)
 * @return 0 if validation passes, non-zero otherwise
 */
int validateArguments(int argc, char **argv, int *type, int *m, int *tlr_rk, int *maxrank) {
    if (argc != 3 && argc != 5) {
        printf("Usage: %s 0/1(0: dense; 1: tlr) matrix_size [rank maxrank]\n", argv[0]);
        printf("  First argument: 0 for dense DGEMM, 1 for TLR DGEMM\n");
        printf("  Second argument: matrix dimension (required)\n");
        printf("  Third argument: TLR rank (required for TLR mode)\n");
        printf("  Fourth argument: maximum rank (required for TLR mode)\n");
        printf("Examples:\n");
        printf("  %s 0 1024          # Dense DGEMM with 1024x1024 matrix\n", argv[0]);
        printf("  %s 1 1024 64 128   # TLR DGEMM with 1024x1024 matrix, rank 64, max rank 128\n", argv[0]);
        return 1;
    }
    
    *type = atoi(argv[1]);
    *m = atoi(argv[2]);
    
    /* Validate matrix type */
    if (*type != 0 && *type != 1) {
        printf("Error: Matrix type must be 0 (dense) or 1 (TLR)\n");
        return 1;
    }
    
    /* Validate matrix dimension */
    if (*m <= 0) {
        printf("Error: Matrix dimension must be positive\n");
        return 1;
    }
    
    /* For TLR mode, validate rank parameters */
    if (*type == 1) {
        if (argc != 5) {
            printf("Error: TLR mode requires rank and maxrank parameters\n");
            return 1;
        }
        *tlr_rk = atoi(argv[3]);
        *maxrank = atoi(argv[4]);
        
        if (*tlr_rk <= 0 || *maxrank <= 0) {
            printf("Error: Rank and maxrank must be positive\n");
            return 1;
        }
        if (*tlr_rk > *maxrank) {
            printf("Error: Rank cannot exceed maxrank\n");
            return 1;
        }
    }
    
    return 0;
}

/**
 * @brief Allocates memory for dense matrices A, B, and C
 * 
 * @param m Matrix dimension
 * @param a Pointer to store allocated matrix A
 * @param b Pointer to store allocated matrix B
 * @param c Pointer to store allocated matrix C
 * @return 0 if allocation succeeds, non-zero otherwise
 */
int allocateDenseMatrices(int m, FLOAT **a, FLOAT **b, FLOAT **c) {
    /* Allocate memory for matrix A (m x m) */
    *a = (FLOAT *)malloc(m * m * sizeof(FLOAT));
    if (*a == NULL) {
        printf("Error: Out of memory for A matrix (%d x %d)\n", m, m);
        return 1;
    }
    
    /* Allocate memory for matrix B (m x m) */
    *b = (FLOAT *)malloc(m * m * sizeof(FLOAT));
    if (*b == NULL) {
        printf("Error: Out of memory for B matrix (%d x %d)\n", m, m);
        free(*a);
        return 1;
    }
    
    /* Allocate memory for matrix C (m x m) */
    *c = (FLOAT *)malloc(m * m * sizeof(FLOAT));
    if (*c == NULL) {
        printf("Error: Out of memory for C matrix (%d x %d)\n", m, m);
        free(*a);
        free(*b);
        return 1;
    }
    
    return 0;
}

/**
 * @brief Initializes dense matrices with random values
 * 
 * @param m Matrix dimension
 * @param a Matrix A to initialize
 * @param b Matrix B to initialize
 * @param c Matrix C to initialize
 */
void initializeDenseMatrices(int m, FLOAT *a, FLOAT *b, FLOAT *c) {
    for (int j = 0; j < m; j++) {
        for (int i = 0; i < m; i++) {
            a[i + (size_t)j * (size_t)m] = rand() / (double)RAND_MAX - 0.5;
            b[i + (size_t)j * (size_t)m] = rand() / (double)RAND_MAX - 0.5;
            c[i + (size_t)j * (size_t)m] = rand() / (double)RAND_MAX - 0.5;
        }
    }
}

/**
 * @brief Performs dense DGEMM benchmark
 * 
 * @param m Matrix dimension
 * @param a Matrix A
 * @param b Matrix B
 * @param c Matrix C
 * @param alpha Scaling factor for A * B
 * @param beta Scaling factor for C
 */
void runDenseDGEMMBenchmark(int m, FLOAT *a, FLOAT *b, FLOAT *c, FLOAT alpha, FLOAT beta) {
    double gflops = 2.0 * (double)m * (double)m * (double)m;
    
    printf("Running dense DGEMM benchmark with matrix size %d x %d\n", m, m);
    
    /* Warm-up run */
    GEMM("N", "N", &m, &m, &m, &alpha, a, &m, b, &m, &beta, c, &m);
    
    /* Benchmark run with timing */
    double dstart = MPI_Wtime();
    
    for (int i = 0; i < LOOP; i++) {
        GEMM("N", "N", &m, &m, &m, &alpha, a, &m, b, &m, &beta, c, &m);
    }
    
    double dstop = MPI_Wtime();
    
    printf("DGEMM Performance N = %6d : %10.4f GF, time %10.6f (s)\n", 
           m, gflops / (dstop - dstart) * (double)LOOP * 1.e-9, (dstop - dstart) / (double)LOOP);
}

/**
 * @brief Allocates memory for TLR matrices and work arrays
 * 
 * @param m Matrix dimension
 * @param tlr_rk TLR rank
 * @param Au Pointer to store allocated U matrix for A
 * @param Av Pointer to store allocated V matrix for A
 * @param Bu Pointer to store allocated U matrix for B
 * @param Bv Pointer to store allocated V matrix for B
 * @param Cu Pointer to store allocated U matrix for C
 * @param Cv Pointer to store allocated V matrix for C
 * @param Ar Pointer to store allocated rank for A
 * @param Br Pointer to store allocated rank for B
 * @param Cr Pointer to store allocated rank for C
 * @param p_elem_work Pointer to store allocated work array
 * @return 0 if allocation succeeds, non-zero otherwise
 */
int allocateTLRMatrices(int m, int tlr_rk, FLOAT **Au, FLOAT **Av, FLOAT **Bu, FLOAT **Bv,
                        FLOAT **Cu, FLOAT **Cv, FLOAT **Ar, FLOAT **Br, FLOAT **Cr,
                        double **p_elem_work) {
    /* Allocate U and V matrices for TLR format */
    *Au = (FLOAT *)malloc(m * m * sizeof(FLOAT));
    *Av = (FLOAT *)malloc(m * m * sizeof(FLOAT));
    *Bu = (FLOAT *)malloc(m * m * sizeof(FLOAT));
    *Bv = (FLOAT *)malloc(m * m * sizeof(FLOAT));
    *Cu = (FLOAT *)malloc(m * m * sizeof(FLOAT));
    *Cv = (FLOAT *)malloc(m * m * sizeof(FLOAT));
    
    /* Allocate rank scalars */
    *Ar = malloc(sizeof(FLOAT));
    *Br = malloc(sizeof(FLOAT));
    *Cr = malloc(sizeof(FLOAT));
    
    /* Allocate work array for HCORE operations */
    *p_elem_work = malloc(8 * m * m * sizeof(double));
    
    /* Check allocation success */
    if (!*Au || !*Av || !*Bu || !*Bv || !*Cu || !*Cv || !*Ar || !*Br || !*Cr || !*p_elem_work) {
        printf("Error: Out of memory for TLR matrices\n");
        return 1;
    }
    
    return 0;
}

/**
 * @brief Performs TLR DGEMM benchmark
 * 
 * @param m Matrix dimension
 * @param tlr_rk TLR rank
 * @param maxrank Maximum rank
 * @param Au U matrix for A
 * @param Av V matrix for A
 * @param Bu U matrix for B
 * @param Bv V matrix for B
 * @param Cu U matrix for C
 * @param Cv V matrix for C
 * @param Ar Rank for A
 * @param Br Rank for B
 * @param Cr Rank for C
 * @param p_elem_work Work array
 */
void runTLRDGEMMBenchmark(int m, int tlr_rk, int maxrank,
                          FLOAT *Au, FLOAT *Av, FLOAT *Bu, FLOAT *Bv,
                          FLOAT *Cu, FLOAT *Cv, FLOAT *Ar, FLOAT *Br, FLOAT *Cr,
                          double *p_elem_work) {
    int ldamu = m;
    int ldamv = m;
    int rk = 0;
    double acc = 1.0e-8;
    flop_counter flops;
    
    printf("Running TLR DGEMM benchmark with matrix size %d x %d, rank %d, max rank %d\n", 
           m, m, tlr_rk, maxrank);
    
    /* Set ranks for TLR matrices */
    *Ar = tlr_rk;
    *Br = tlr_rk;
    *Cr = tlr_rk;
    
    /* Warm-up run */
    HCORE_dgemm(PlasmaNoTrans, PlasmaTrans, 
                m, /* mb */ 
                m, /* nb */
                (FLOAT)-1.0, 
                Au, Av, Ar, ldamu,
                Bu, Bv, Br, ldamv,
                (FLOAT)1.0,
                Cu, Cv, Cr, ldamu,
                rk, maxrank, acc, p_elem_work, &flops);
    
    /* Benchmark run with timing */
    for (int i = 0; i < LOOP; i++) {
        double dstart = MPI_Wtime();
        
        *Ar = tlr_rk;
        *Br = tlr_rk;
        *Cr = tlr_rk;
        
        HCORE_dgemm(PlasmaNoTrans, PlasmaTrans, 
                    m, /* mb */ 
                    m, /* nb */
                    (double)-1.0, 
                    Au, Av, Ar, ldamu,
                    Bu, Bv, Br, ldamv,
                    (double)1.0,
                    Cu, Cv, Cr, ldamu,
                    rk, maxrank, acc, p_elem_work, &flops);
        
        double dstop = MPI_Wtime();
        
        printf("TLR DGEMM Performance N = %6d rank = %d maxrank= %d %10.4f GF, time %10.6f (s)\n",
               m, tlr_rk, maxrank, 
               (36. * m * tlr_rk * tlr_rk + 157 * tlr_rk * tlr_rk) / (dstop - dstart) * 1.e-9, 
               (dstop - dstart));
    }
}

/**
 * @brief Main function - entry point of the TLR DGEMM testing program
 * 
 * This program provides a comprehensive benchmark for DGEMM operations using both dense and TLR
 * matrix formats, allowing performance comparison between traditional BLAS and modern TLR approaches.
 * 
 * Command line usage:
 *   Dense mode: ./testing_dgemm_tlr 0 <matrix_size>
 *   TLR mode: ./testing_dgemm_tlr 1 <matrix_size> <rank> <maxrank>
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return Exit status (0 for success, non-zero for failure)
 */
int main(int argc, char **argv) {
    MPI_Init(NULL, NULL);
    
    int type, m, tlr_rk, maxrank;
    
    /* Validate command line arguments */
    if (validateArguments(argc, argv, &type, &m, &tlr_rk, &maxrank) != 0) {
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    
    printf("Configuration: type=%d, matrix_size=%d, rank=%d, maxrank=%d\n", 
           type, m, tlr_rk, maxrank);
    
    if (type == 0) {
        /* Run dense DGEMM benchmark */
        FLOAT *a, *b, *c;
        FLOAT alpha = 1.0;
        FLOAT beta = 1.0;
        
        /* Allocate dense matrices */
        if (allocateDenseMatrices(m, &a, &b, &c) != 0) {
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        
        /* Initialize matrices with random values */
        initializeDenseMatrices(m, a, b, c);
        
        /* Run dense DGEMM benchmark */
        runDenseDGEMMBenchmark(m, a, b, c, alpha, beta);
        
        /* Clean up dense matrices */
        free(a);
        free(b);
        free(c);
        
    } else {
        /* Run TLR DGEMM benchmark */
        FLOAT *Au, *Av, *Bu, *Bv, *Cu, *Cv;
        FLOAT *Ar, *Br, *Cr;
        double *p_elem_work;
        
        /* Allocate TLR matrices */
        if (allocateTLRMatrices(m, tlr_rk, &Au, &Av, &Bu, &Bv, &Cu, &Cv, 
                               &Ar, &Br, &Cr, &p_elem_work) != 0) {
            MPI_Finalize();
            return EXIT_FAILURE;
        }
        
        /* Initialize TLR matrices using SVD-based approach */
        /* Note: This is a simplified initialization - in practice, you would load real TLR data */
        printf("Initializing TLR matrices with synthetic data...\n");
        
        /* Set ranks */
        *Ar = tlr_rk;
        *Br = tlr_rk;
        *Cr = tlr_rk;
        
        /* Run TLR DGEMM benchmark */
        runTLRDGEMMBenchmark(m, tlr_rk, maxrank, Au, Av, Bu, Bv, Cu, Cv, 
                            Ar, Br, Cr, p_elem_work);
        
        /* Clean up TLR matrices */
        free(Au);
        free(Av);
        free(Bu);
        free(Bv);
        free(Cu);
        free(Cv);
        free(Ar);
        free(Br);
        free(Cr);
        free(p_elem_work);
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
