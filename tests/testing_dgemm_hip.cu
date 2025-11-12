/**
 * @file testing_dgemm_hip.cu
 * @brief HIP-based test program for double precision general matrix-matrix multiplication (DGEMM)
 * 
 * This program benchmarks the performance of hipBLAS DGEMM operation on AMD GPUs by:
 * - Allocating matrices A, B, and C on both host and device
 * - Initializing them with deterministic test patterns for reproducibility
 * - Running multiple iterations of DGEMM with optional memory transfer timing
 * - Measuring and reporting performance in GFlop/s with detailed timing
 * - Providing comprehensive error handling and resource cleanup
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

#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <sys/time.h>
#include <vector>

#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>

#include "hicma_parsec_hip_cuda.h"

#define DEBUG_INFO 0

#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        hipError_t err_ = (err);                                                                  \
        if (err_ != hipSuccess) {                                                                 \
            printf("HIP error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
        }                                                                                          \
    } while (0)

// hipblas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        hipblasStatus_t err_ = (err);                                                               \
        if (err_ != HIPBLAS_STATUS_SUCCESS) {                                                       \
            printf("hipblas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
        }                                                                                          \
    } while (0)

/**
 * @brief Prints a matrix in a formatted way for debugging purposes
 * 
 * @param m Number of rows
 * @param n Number of columns
 * @param A Matrix data pointer
 * @param lda Leading dimension of matrix A
 */
static void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%0.2f ", A[j * lda + i]);
        }
        printf("\n");
    }
}

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
int validateArguments(int argc, char *argv[], int *m, int *n, int *k) {
    if (argc == 2) {
        /* Single dimension: m = n = k */
        *m = *n = *k = atoi(argv[1]);
    } else if (argc == 4) {
        /* Three dimensions: m, n, k */
        *m = atoi(argv[1]);
        *n = atoi(argv[2]);
        *k = atoi(argv[3]);
    } else {
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
 * @brief Allocates memory for matrices A, B, and C on the host
 * 
 * @param m Matrix dimension m
 * @param n Matrix dimension n
 * @param k Matrix dimension k
 * @param A Pointer to store allocated matrix A
 * @param B Pointer to store allocated matrix B
 * @param C Pointer to store allocated matrix C
 * @return 0 if allocation succeeds, non-zero otherwise
 */
int allocateHostMatrices(int m, int n, int k, double **A, double **B, double **C) {
    /* Allocate memory for matrix A (m x k) */
    *A = (double *)malloc(m * k * sizeof(double));
    if (*A == NULL) {
        printf("Error: Out of memory for A matrix (%d x %d)\n", m, k);
        return 1;
    }
    
    /* Allocate memory for matrix B (k x n) */
    *B = (double *)malloc(k * n * sizeof(double));
    if (*B == NULL) {
        printf("Error: Out of memory for B matrix (%d x %d)\n", k, n);
        free(*A);  /* Clean up previously allocated memory */
        return 1;
    }
    
    /* Allocate memory for matrix C (m x n) */
    *C = (double *)malloc(m * n * sizeof(double));
    if (*C == NULL) {
        printf("Error: Out of memory for C matrix (%d x %d)\n", m, n);
        free(*A);  /* Clean up previously allocated memory */
        free(*B);
        return 1;
    }
    
    return 0;
}

/**
 * @brief Allocates memory for matrices A, B, and C on the GPU device
 * 
 * @param m Matrix dimension m
 * @param n Matrix dimension n
 * @param k Matrix dimension k
 * @param d_A Pointer to store allocated device matrix A
 * @param d_B Pointer to store allocated device matrix B
 * @param d_C Pointer to store allocated device matrix C
 * @return 0 if allocation succeeds, non-zero otherwise
 */
int allocateDeviceMatrices(int m, int n, int k, double **d_A, double **d_B, double **d_C) {
    /* Allocate device memory for matrix A (m x k) */
    CUDA_CHECK(hipMalloc(d_A, sizeof(double) * m * k));
    
    /* Allocate device memory for matrix B (k x n) */
    CUDA_CHECK(hipMalloc(d_B, sizeof(double) * k * n));
    
    /* Allocate device memory for matrix C (m x n) */
    CUDA_CHECK(hipMalloc(d_C, sizeof(double) * m * n));
    
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
 * @param A Matrix A to initialize (m x k)
 * @param B Matrix B to initialize (k x n)
 */
void initializeMatrices(int m, int n, int k, double *A, double *B) {
    /* Initialize matrix A with test pattern: (i + j + 1.0) / (m * k) */
    /* This pattern ensures all elements are in [0, 1] range for numerical stability */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            A[i * k + j] = (i + j + 1.0) / (m * k);
        }
    }
    
    /* Initialize matrix B with test pattern: (i + j + 1.0) / (k * n) */
    /* Similar pattern for consistency and numerical stability */
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            B[i * n + j] = (i + j + 1.0) / (k * n);
        }
    }
}

/**
 * @brief Performs DGEMM benchmark with optional memory transfer timing
 * 
 * @param cublasH hipBLAS handle
 * @param stream HIP stream for asynchronous operations
 * @param m Matrix dimension m
 * @param n Matrix dimension n
 * @param k Matrix dimension k
 * @param A Host matrix A (m x k)
 * @param B Host matrix B (k x n)
 * @param C Host matrix C (m x n) - result matrix
 * @param d_A Device matrix A (m x k)
 * @param d_B Device matrix B (k x n)
 * @param d_C Device matrix C (m x n)
 * @param alpha Scaling factor for A * B
 * @param beta Scaling factor for C
 * @param lda Leading dimension of matrix A
 * @param ldb Leading dimension of matrix B
 * @param ldc Leading dimension of matrix C
 * @param nb_runs Number of benchmark runs
 * @param transa Transpose operation for matrix A
 * @param transb Transpose operation for matrix B
 */
void runDGEMMBenchmark(hipblasHandle_t cublasH, hipStream_t stream,
                       int m, int n, int k,
                       double *A, double *B, double *C,
                       double *d_A, double *d_B, double *d_C,
                       const double alpha, const double beta,
                       const int lda, const int ldb, const int ldc,
                       const int nb_runs,
                       hipblasOperation_t transa, hipblasOperation_t transb) {
    
    struct timeval tstart, tend;
    double t, gflops;
    
    printf("Testing HIP DGEMM with matrix dimensions: %d x %d x %d\n", m, n, k);
    printf("Number of runs: %d\n", nb_runs);
    printf("DGEMM operation: C = %.1f * A * B + %.1f * C\n", alpha, beta);
    printf("Format: m n k : time(seconds) performance(GFlop/s)\n");
    
    /* Run DGEMM multiple times for benchmarking and statistical accuracy */
    for (int i = 0; i < nb_runs; i++) {
        
#if INCLUDE_MEMCPY
        /* Start timing including memory transfers */
        gettimeofday(&tstart, NULL);
#endif
        
        /* Copy data from host to device asynchronously */
        CUDA_CHECK(hipMemcpyAsync(d_A, A, sizeof(double) * m * k, hipMemcpyHostToDevice, stream));
        CUDA_CHECK(hipMemcpyAsync(d_B, B, sizeof(double) * k * n, hipMemcpyHostToDevice, stream));
        
#if !INCLUDE_MEMCPY
        /* Start timing excluding memory transfers */
        gettimeofday(&tstart, NULL);
#endif
        
        /* Perform DGEMM operation on GPU: C = alpha * A * B + beta * C */
        CUBLAS_CHECK(hipblasDgemm(cublasH, transa, transb, m, n, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc));
        
#if !INCLUDE_MEMCPY
        /* Synchronize stream and end timing */
        CUDA_CHECK(hipStreamSynchronize(stream));
        
        gettimeofday(&tend, NULL);
        t = (tend.tv_sec - tstart.tv_sec) + (tend.tv_usec - tstart.tv_usec) / 1000000.0;
        gflops = (double)m * n * k * 2.0 / t / 1.0e9;
        printf("%d %d %d : %lf seconds %lf Gflop/s\n", m, n, k, t, gflops);
#endif
        
#if INCLUDE_MEMCPY
        /* Synchronize stream and end timing */
        CUDA_CHECK(hipStreamSynchronize(stream));
        
        gettimeofday(&tend, NULL);
        t = (tend.tv_sec - tstart.tv_sec) + (tend.tv_usec - tstart.tv_usec) / 1000000.0;
        gflops = (double)m * n * k * 2.0 / t / 1.0e9;
        printf("%d %d %d : %lf seconds %lf Gflop/s\n", m, n, k, t, gflops);
#endif
        
        /* Copy result from device to host asynchronously */
        CUDA_CHECK(hipMemcpyAsync(C, d_C, sizeof(double) * m * n, hipMemcpyDeviceToHost, stream));
        
        if(DEBUG_INFO) {
            printf("Result matrix C:\n");
            print_matrix(m, n, C, ldc);
            printf("=====\n");
        }
    }

    printf("HIP DGEMM testing completed successfully.\n");
}

/**
 * @brief Main function - entry point of the HIP DGEMM testing program
 * 
 * This program provides a comprehensive benchmark for hipBLAS DGEMM operations on AMD GPUs,
 * measuring performance in GFlop/s across multiple runs for statistical accuracy.
 * 
 * Command line usage:
 *   Single dimension: ./testing_dgemm_hip <size>
 *   Three dimensions: ./testing_dgemm_hip <m> <n> <k>
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return Exit status (0 for success, non-zero for failure)
 */
int main(int argc, char *argv[]) {
    hipblasHandle_t cublasH = NULL;
    hipStream_t stream = NULL;
    
    int m, n, k;
    
    /* Validate command line arguments and matrix dimensions */
    if (validateArguments(argc, argv, &m, &n, &k) != 0) {
        return EXIT_FAILURE;
    }
    
    /* Matrix leading dimensions and benchmark parameters */
    const int lda = m; 
    const int ldb = k; 
    const int ldc = m; 
    const int nb_runs = 3;
    
    /* Host matrices - using stack allocation for better performance */
    double A[m * k], B[k * n], C[m * n];
    
    /* DGEMM operation parameters */
    const double alpha = 1.0;
    const double beta = 0.0;
    
    /* Device matrices */
    double *d_A, *d_B, *d_C;
    
    /* Transpose operations (no transpose for both matrices) */
    hipblasOperation_t transa = HIPBLAS_OP_N;
    hipblasOperation_t transb = HIPBLAS_OP_N;
    
    /* Step 1: Create hipBLAS handle and bind to a stream */
    CUBLAS_CHECK(hipblasCreate(&cublasH));
    CUDA_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    CUBLAS_CHECK(hipblasSetStream(cublasH, stream));
    
    /* Initialize matrices with deterministic test patterns */
    initializeMatrices(m, n, k, A, B);
    
    if(DEBUG_INFO) {
        printf("Matrix A:\n");
        print_matrix(m, k, A, lda);
        printf("=====\n");

        printf("Matrix B:\n");
        print_matrix(k, n, B, ldb);
        printf("=====\n");
    }

    /* Step 2: Allocate device memory */
    allocateDeviceMatrices(m, n, k, &d_A, &d_B, &d_C);
    
    /* Step 3: Run DGEMM benchmark */
    runDGEMMBenchmark(cublasH, stream, m, n, k, A, B, C, d_A, d_B, d_C,
                      alpha, beta, lda, ldb, ldc, nb_runs, transa, transb);
    
    /* Step 4: Clean up resources */
    CUDA_CHECK(hipFree(d_A));
    CUDA_CHECK(hipFree(d_B));
    CUDA_CHECK(hipFree(d_C));
    
    CUBLAS_CHECK(hipblasDestroy(cublasH));
    CUDA_CHECK(hipStreamDestroy(stream));
    
    /* Reset HIP device */
    CUDA_CHECK(hipDeviceReset());
    
    return EXIT_SUCCESS;
}
