#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <sys/time.h>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
 * @file testing_dgemv.cu
 * @brief CUDA cuBLAS DGEMV performance testing program
 * 
 * This program tests the performance of double-precision general matrix-vector
 * multiplication (DGEMV) using NVIDIA cuBLAS library. It measures execution
 * time and computes GFLOP/s for various matrix sizes.
 * 
 * Usage: ./testing_dgemv [matrix_size]
 *        matrix_size: Size of the square matrix (default: 10000)
 */

#define INCLUDE_MEMCPY    0

// Utility macro to check CUDA and cuBLAS results
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    const cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("cuBLAS Error: %s:%d, ", __FILE__, __LINE__); \
        printf("status: %d\n", status); \
        exit(1); \
    } \
}

/**
 * @brief Initialize matrix and vectors with random values
 * @param A Matrix to initialize
 * @param x Input vector to initialize
 * @param y Output vector to initialize
 * @param m Number of rows in matrix
 * @param n Number of columns in matrix
 */
void initialize_data(double *A, double *x, double *y, int m, int n) {
    // Initialize matrix A with random values
    for (int i = 0; i < m * n; i++) {
        A[i] = (double)rand() / RAND_MAX;
    }
    
    // Initialize input vector x with random values
    for (int i = 0; i < n; i++) {
        x[i] = (double)rand() / RAND_MAX;
    }
    
    // Initialize output vector y with zeros
    for (int i = 0; i < m; i++) {
        y[i] = 0.0;
    }
}

/**
 * @brief Print performance results
 * @param m Matrix rows
 * @param n Matrix columns
 * @param time Execution time in seconds
 * @param gflops Computed GFLOP/s
 */
void print_results(int m, int n, double time, double gflops) {
    printf("Matrix size: %d x %d | Time: %.6f seconds | Performance: %.2f GFLOP/s\n", 
           m, n, time, gflops);
}

int main(int argc, char **argv) {
    cublasHandle_t handle;
    cudaStream_t stream = NULL;

    // Default matrix dimensions
    int n = 10000; // Size of the vector
    int m = 10000; // Size of the matrix rows, for simplicity square matrix is used

    // Parse command line arguments
    if (2 == argc) {
        m = atoi(argv[1]);
        if (m <= 0) {
            printf("Error: Matrix size must be positive\n");
            printf("Usage: ./testing_dgemv [matrix_size]\n");
            return 1;
        }
    } else if(argc > 2) {
        printf("Error: Too many arguments\n");
        printf("Usage: ./testing_dgemv [matrix_size]\n");
        return 1;
    }
    n = m; // Use square matrix for simplicity

    const int nb_runs = 10;  // Number of benchmark runs
    struct timeval tstart, tend;
    double t, gflops;

    // Host memory pointers
    double *A, *x, *y;
    // Device memory pointers
    double *d_A, *d_x, *d_y;
    
    // DGEMV parameters: y = alpha * A * x + beta * y
    double alpha = 1.0;
    double beta = 0.0;
    size_t bytes = m * n * sizeof(double);

    printf("Starting DGEMV performance test with matrix size %d x %d\n", m, n);
    printf("Number of benchmark runs: %d\n", nb_runs);

    // Allocate host memory
    A = (double*)malloc(bytes);
    x = (double*)malloc(n * sizeof(double));
    y = (double*)malloc(m * sizeof(double));
    
    if (!A || !x || !y) {
        printf("Error: Failed to allocate host memory\n");
        return 1;
    }

    // Initialize host data with random values
    srand(42); // Fixed seed for reproducible results
    initialize_data(A, x, y, m, n);

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_x, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, m * sizeof(double)));

    // Create cuBLAS context
    CHECK_CUBLAS(cublasCreate(&handle));

    // Create CUDA stream for asynchronous execution
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    printf("Starting benchmark runs...\n");
    printf("----------------------------------------\n");

    // Main benchmark loop
    for( int i = 0; i < nb_runs; i++ ) {
#if INCLUDE_MEMCPY
        // Start timing including memory transfers
        gettimeofday(&tstart, NULL);
#endif

        // Copy host memory to device asynchronously
        CHECK_CUDA(cudaMemcpyAsync(d_A, A, bytes, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_y, y, m * sizeof(double), cudaMemcpyHostToDevice, stream));

#if !INCLUDE_MEMCPY
        // Start timing excluding memory transfers
        gettimeofday(&tstart, NULL);
#endif

        // Perform DGEMV operation: y = alpha * A^T * x + beta * y
        // Note: Using CUBLAS_OP_T for transposition since A is stored in row-major format
        // but cuBLAS expects column-major format
        CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T, n, m, &alpha, d_A, n, d_x, 1, &beta, d_y, 1));

#if !INCLUDE_MEMCPY
        // Synchronize stream and measure computation time only
        CHECK_CUDA(cudaStreamSynchronize(stream));

        gettimeofday(&tend, NULL);
        t = (tend.tv_sec - tstart.tv_sec) + (tend.tv_usec - tstart.tv_usec) / 1000000.0;
        gflops = (double)m * n * 2.0 / t / 1.0e9; // 2 FLOPs per element (multiply + add)
        print_results(m, n, t, gflops);
#endif

        // Copy result back to host
        CHECK_CUDA(cudaMemcpy(y, d_y, m * sizeof(double), cudaMemcpyDeviceToHost));

#if INCLUDE_MEMCPY
        // Synchronize stream and measure total time including memory transfers
        CHECK_CUDA(cudaStreamSynchronize(stream));

        gettimeofday(&tend, NULL);
        t = (tend.tv_sec - tstart.tv_sec) + (tend.tv_usec - tstart.tv_usec) / 1000000.0;
        gflops = (double)m * n * 2.0 / t / 1.0e9;
        print_results(m, n, t, gflops);
#endif
    }

    printf("----------------------------------------\n");
    printf("Benchmark completed successfully\n");

    // Cleanup: Destroy cuBLAS context
    CHECK_CUBLAS(cublasDestroy(handle));

    // Cleanup: Free all allocated memory
    free(A); 
    free(x); 
    free(y);
    CHECK_CUDA(cudaFree(d_A)); 
    CHECK_CUDA(cudaFree(d_x)); 
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}

