#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_fp8.h>

//#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_fp8.hpp>

/**
 * @file testing_gemm_gpu_dp.cu
 * @brief Double precision GPU GEMM performance testing program
 * 
 * This program specifically tests double precision (FP64) GEMM operations using NVIDIA cuBLAS.
 * It focuses on measuring performance of different transpose patterns (NT and TN) for
 * double precision matrix multiplication.
 * 
 * The program performs:
 * - NT operation: C = A * B^T (No transpose on A, Transpose on B)
 * - TN operation: C = A^T * B (Transpose on A, No transpose on B)
 * 
 * Usage: ./testing_gemm_gpu_dp [matrix_size] [nb_runs]
 *        matrix_size: Size of the square matrix (default: 2048)
 *        nb_runs: Number of benchmark runs (default: 10)
 */

#define CHUNKSIZE 32 

/**
 * @brief Initialize host matrix with random double precision values
 * @param A Pointer to store the allocated matrix
 * @param M Number of rows
 * @param N Number of columns
 * @param seed Random seed for reproducible results
 */
static void init_matrix_host( void **A, int M, int N, int seed ) {
    srand( seed );  // Set random seed for reproducible results
    cudaMallocHost((void**)A, M * N * sizeof(double));  // Allocate pinned host memory
    
    double *_A = (double *)(*A);
    for( int i = 0; i < M; i++ ) {
        for( int j = 0; j < N; j++ ) {
            _A[i*N+j] = (double)rand()/RAND_MAX;  // Generate random values in [0,1)
        }
    }
}

int main(int argc, char *argv[]) {
    // Default parameters
    int N = 2048;        // Matrix size (square matrix N x N)
    int nb_runs = 10;    // Number of benchmark runs
    
    // Parse command line arguments if provided
    if (argc >= 2) {
        N = atoi(argv[1]);
        if (N <= 0) {
            fprintf(stderr, "Error: Matrix size must be positive\n");
            return 1;
        }
    }
    if (argc >= 3) {
        nb_runs = atoi(argv[2]);
        if (nb_runs <= 0) {
            fprintf(stderr, "Error: Number of runs must be positive\n");
            return 1;
        }
    }
    
    printf("Starting double precision GEMM test with matrix size %d x %d\n", N, N);
    printf("Number of benchmark runs: %d\n", nb_runs);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Create CUDA stream for asynchronous execution
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(handle, stream);

    // Host memory pointers
    void *h_A, *h_B, *h_C;

    // Initialize host matrices with different seeds for variety
    init_matrix_host(&h_A, N, N, 2782);
    init_matrix_host(&h_B, N, N, 2782);
    init_matrix_host(&h_C, N, N, 2782);

    // Allocate device memory
    void *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(double));
    cudaMalloc(&d_B, N * N * sizeof(double));
    cudaMalloc(&d_C, N * N * sizeof(double));

    // Copy host matrices to device asynchronously
    cudaMemcpyAsync(d_A, h_A, N*N*sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, N*N*sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_C, h_C, N*N*sizeof(double), cudaMemcpyHostToDevice, stream);

    // GEMM parameters
    double alpha = 1.0;  // Scaling factor for the product
    double beta = 0.0;   // Scaling factor for matrix C
    double gflops = 2.0 * N * N * N / (1e12);  // Theoretical GFLOPs for this operation

    printf("Starting benchmark runs...\n");
    printf("----------------------------------------\n");

    // Main benchmark loop
    for (int i = 0; i < nb_runs; ++i) {
        double start_time, end_time;
        struct timeval tstart, tend;
        
        /* Test NT operation: C = A * B^T */
        cudaDeviceSynchronize();      // Ensure all previous operations are complete
        cudaStreamSynchronize(stream); // Synchronize the stream
        gettimeofday(&tstart, NULL);
        start_time = tstart.tv_sec + tstart.tv_usec / 1.0e6;

        // Perform GEMM operation: C = alpha * A * B^T + beta * C
        cublasGemmEx(handle,
                CUBLAS_OP_N,      // No transpose on A
                CUBLAS_OP_T,      // Transpose on B
                N, N, N,          // Matrix dimensions
                &alpha,           // Scaling factor
                d_A, CUDA_R_64F, N,  // Matrix A, double precision, leading dimension N
                d_B, CUDA_R_64F, N,  // Matrix B, double precision, leading dimension N
                &beta,            // Scaling factor for C
                d_C, CUDA_R_64F, N,  // Matrix C, double precision, leading dimension N
                     CUDA_R_64F,     // Compute type (double precision)
                     CUBLAS_GEMM_DEFAULT);  // Algorithm selection

        cudaStreamSynchronize(stream);  // Wait for GEMM operation to complete
        cudaDeviceSynchronize();        // Ensure all operations are complete
        gettimeofday(&tend, NULL);
        end_time = tend.tv_sec + tend.tv_usec / 1.0e6;
        
        // Calculate and display performance for NT operation
        double nt_time = end_time - start_time;
        double nt_tflops = gflops / nt_time;
        printf("GEMM NT: %d %d %d : %.3f TFLOPS (%.6f seconds)\n", 
               N, N, N, nt_tflops, nt_time);

       /* Test TN operation: C = A^T * B */
        cudaDeviceSynchronize();      // Ensure all previous operations are complete
        cudaStreamSynchronize(stream); // Synchronize the stream
        gettimeofday(&tstart, NULL); 
        start_time = tstart.tv_sec + tstart.tv_usec / 1.0e6;
        
        // Perform GEMM operation: C = alpha * A^T * B + beta * C
        cublasGemmEx(handle, 
                CUBLAS_OP_T,      // Transpose on A
                CUBLAS_OP_N,      // No transpose on B
                N, N, N,          // Matrix dimensions
                &alpha,           // Scaling factor
                d_A, CUDA_R_64F, N,  // Matrix A, double precision, leading dimension N
                d_B, CUDA_R_64F, N,  // Matrix B, double precision, leading dimension N
                &beta,            // Scaling factor for C
                d_C, CUDA_R_64F, N,  // Matrix C, double precision, leading dimension N
                     CUDA_R_64F,     // Compute type (double precision)
                     CUBLAS_GEMM_DEFAULT);  // Algorithm selection
        
        cudaStreamSynchronize(stream);  // Wait for GEMM operation to complete
        cudaDeviceSynchronize();        // Ensure all operations are complete
        gettimeofday(&tend, NULL);
        end_time = tend.tv_sec + tend.tv_usec / 1.0e6;
        
        // Calculate and display performance for TN operation
        double tn_time = end_time - start_time;
        double tn_tflops = gflops / tn_time;
        printf("GEMM TN: %d %d %d : %.3f TFLOPS (%.6f seconds)\n\n", 
               N, N, N, tn_tflops, tn_time);
    }

    printf("----------------------------------------\n");
    printf("Benchmark completed successfully\n");

    // Clean up allocated resources
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
