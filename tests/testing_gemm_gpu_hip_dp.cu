#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <time.h>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>

/**
 * @file testing_gemm_gpu_hip_dp.cu
 * @brief HIP-based GPU GEMM performance testing for double precision matrices
 * 
 * This program benchmarks HIPBLAS GEMM operations on GPU using HIP runtime.
 * It tests both NT (NoTranspose, Transpose) and TN (Transpose, NoTranspose) 
 * matrix multiplication patterns for performance evaluation.
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 */

#define CHUNKSIZE 32 

/**
 * @brief Initialize host matrix with random double precision values
 * @param A Pointer to matrix data pointer
 * @param M Number of rows
 * @param N Number of columns  
 * @param seed Random seed for reproducible results
 */
static void init_matrix_host(void **A, int M, int N, int seed) {
    srand(seed);
    
    // Allocate pinned host memory for optimal GPU transfer
    hipError_t hip_err = hipHostMalloc((void**)A, M * N * sizeof(double));
    if (hip_err != hipSuccess) {
        fprintf(stderr, "Failed to allocate host memory: %s\n", hipGetErrorString(hip_err));
        exit(EXIT_FAILURE);
    }
    
    double *_A = (double *)(*A);
    
    // Fill matrix with random values in range [0, 1)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            _A[i * N + j] = (double)rand() / RAND_MAX;
        }
    }
}

/**
 * @brief Main function for HIP GPU GEMM performance testing
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return 0 on success, non-zero on failure
 */
int main(int argc, char *argv[]) {
    // Matrix dimensions and performance parameters
    int N = 2048;           // Matrix size (N x N)
    int nb_runs = 10;       // Number of benchmark runs
    
    // HIPBLAS handle for library operations
    hipblasHandle_t handle;
    hipblasStatus_t status = hipblasCreate(&handle);
    if (status != HIPBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to create HIPBLAS handle\n");
        return EXIT_FAILURE;
    }
    
    // Create HIP stream for asynchronous operations
    hipStream_t stream;
    hipError_t hip_err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "Failed to create HIP stream: %s\n", hipGetErrorString(hip_err));
        hipblasDestroy(handle);
        return EXIT_FAILURE;
    }
    
    // Associate stream with HIPBLAS handle
    hipblasSetStream(handle, stream);
    
    // Matrix data pointers
    void *h_A, *h_B, *h_C;  // Host matrices
    void *d_A, *d_B, *d_C;  // Device matrices
    
    // Initialize host matrices with random data
    init_matrix_host(&h_A, N, N, 2782);
    init_matrix_host(&h_B, N, N, 2782);
    init_matrix_host(&h_C, N, N, 2782);
    
    // Performance measurement variables
    double start_time, end_time;
    struct timeval tstart, tend;
    double alpha = 1.0;  // GEMM alpha parameter
    double beta = 0.0;   // GEMM beta parameter
    
    // Calculate theoretical FLOPS for performance comparison
    double gflops = 2.0 * N * N * N / (1e12);  // 2*M*N*K operations
    
    // Allocate device memory for matrices
    hip_err = hipMalloc(&d_A, N * N * sizeof(double));
    if (hip_err != hipSuccess) {
        fprintf(stderr, "Failed to allocate device memory for A: %s\n", hipGetErrorString(hip_err));
        goto cleanup;
    }
    
    hip_err = hipMalloc(&d_B, N * N * sizeof(double));
    if (hip_err != hipSuccess) {
        fprintf(stderr, "Failed to allocate device memory for B: %s\n", hipGetErrorString(hip_err));
        (void)hipFree(d_A);
        goto cleanup;
    }
    
    hip_err = hipMalloc(&d_C, N * N * sizeof(double));
    if (hip_err != hipSuccess) {
        fprintf(stderr, "Failed to allocate device memory for C: %s\n", hipGetErrorString(hip_err));
        (void)hipFree(d_A);
        (void)hipFree(d_B);
        goto cleanup;
    }
    
    // Asynchronously copy host matrices to device
    hip_err = hipMemcpyAsync(d_A, h_A, N * N * sizeof(double), hipMemcpyHostToDevice, stream);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "Failed to copy A to device: %s\n", hipGetErrorString(hip_err));
        goto cleanup;
    }
    
    hip_err = hipMemcpyAsync(d_B, h_B, N * N * sizeof(double), hipMemcpyHostToDevice, stream);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "Failed to copy B to device: %s\n", hipGetErrorString(hip_err));
        goto cleanup;
    }
    
    hip_err = hipMemcpyAsync(d_C, h_C, N * N * sizeof(double), hipMemcpyHostToDevice, stream);
    if (hip_err != hipSuccess) {
        fprintf(stderr, "Failed to copy C to device: %s\n", hipGetErrorString(hip_err));
        goto cleanup;
    }
    
    printf("Starting HIP GPU GEMM performance benchmark...\n");
    printf("Matrix size: %d x %d, Runs: %d\n\n", N, N, nb_runs);
    
    // Performance benchmark loop
    for (int i = 0; i < nb_runs; ++i) {
        printf("Run %d/%d:\n", i + 1, nb_runs);
        
        // Test NT pattern: A * B^T
        (void)hipDeviceSynchronize();
        (void)hipStreamSynchronize(stream);
        gettimeofday(&tstart, NULL);
        start_time = tstart.tv_sec + tstart.tv_usec / 1.0e6;
        
        // Perform GEMM: C = alpha * A * B^T + beta * C
        status = hipblasGemmEx(handle,
                               HIPBLAS_OP_N,      // A operation: No transpose
                               HIPBLAS_OP_T,      // B operation: Transpose
                               N, N, N,           // Matrix dimensions M, N, K
                               &alpha,            // Alpha scalar
                               d_A, HIPBLAS_R_64F, N,  // Input matrix A
                               d_B, HIPBLAS_R_64F, N,  // Input matrix B
                               &beta,             // Beta scalar
                               d_C, HIPBLAS_R_64F, N,  // Output matrix C
                               HIPBLAS_R_64F,     // Compute type
                               HIPBLAS_GEMM_DEFAULT);   // Algorithm
        
        if (status != HIPBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "GEMM NT operation failed\n");
            goto cleanup;
        }
        
        (void)hipStreamSynchronize(stream);
        (void)hipDeviceSynchronize();
        gettimeofday(&tend, NULL);
        end_time = tend.tv_sec + tend.tv_usec / 1.0e6;
        
        double nt_time = end_time - start_time;
        double nt_perf = gflops / nt_time;
        printf("  GEMM NT: %d %d %d : %.3f TFLOPS (%.6f s)\n", 
               N, N, N, nt_perf, nt_time);
        
        // Test TN pattern: A^T * B
        (void)hipDeviceSynchronize();
        (void)hipStreamSynchronize(stream);
        gettimeofday(&tstart, NULL);
        start_time = tstart.tv_sec + tstart.tv_usec / 1.0e6;
        
        // Perform GEMM: C = alpha * A^T * B + beta * C
        status = hipblasGemmEx(handle,
                               HIPBLAS_OP_T,      // A operation: Transpose
                               HIPBLAS_OP_N,      // B operation: No transpose
                               N, N, N,           // Matrix dimensions M, N, K
                               &alpha,            // Alpha scalar
                               d_A, HIPBLAS_R_64F, N,  // Input matrix A
                               d_B, HIPBLAS_R_64F, N,  // Input matrix B
                               &beta,             // Beta scalar
                               d_C, HIPBLAS_R_64F, N,  // Output matrix C
                               HIPBLAS_R_64F,     // Compute type
                               HIPBLAS_GEMM_DEFAULT);   // Algorithm
        
        if (status != HIPBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "GEMM TN operation failed\n");
            goto cleanup;
        }
        
        (void)hipStreamSynchronize(stream);
        (void)hipDeviceSynchronize();
        gettimeofday(&tend, NULL);
        end_time = tend.tv_sec + tend.tv_usec / 1.0e6;
        
        double tn_time = end_time - start_time;
        double tn_perf = gflops / tn_time;
        printf("  GEMM TN: %d %d %d : %.3f TFLOPS (%.6f s)\n\n", 
               N, N, N, tn_perf, tn_time);
    }
    
    printf("Benchmark completed successfully!\n");
    
cleanup:
    // Clean up allocated resources
    if (h_A) (void)hipHostFree(h_A);
    if (h_B) (void)hipHostFree(h_B);
    if (h_C) (void)hipHostFree(h_C);
    if (d_A) (void)hipFree(d_A);
    if (d_B) (void)hipFree(d_B);
    if (d_C) (void)hipFree(d_C);
    
    // Clean up HIPBLAS and stream
    hipblasDestroy(handle);
    (void)hipStreamDestroy(stream);
    
    return EXIT_SUCCESS;
}
