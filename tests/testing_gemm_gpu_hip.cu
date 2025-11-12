#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <time.h>

#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>

#include "hicma_parsec_hip_cuda.h"

/**
 * @file testing_gemm_gpu_hip.cu
 * @brief HIP-based GPU GEMM performance testing program with mixed precision support
 * 
 * This program tests the performance of various precision GEMM operations using AMD HIP
 * and hipBLAS libraries. It supports multiple data types including FP64, FP32, FP16, and BF16,
 * with comprehensive benchmarking and validation capabilities.
 * 
 * Supported GEMM types:
 * 1: FP64 (Double precision)
 * 2: FP32 (Single precision)
 * 3: TF32 (Tensor Float 32)
 * 4: FP16 with FP32 accumulation (A16B16C32OP32)
 * 5: FP16 with FP16 accumulation (A16B16C16OP16)
 * 6: BF16 with FP32 accumulation (A16B16C32OP32)
 * 7: BF16 with BF16 accumulation (A16B16C16OP16)
 * 
 * Usage: ./testing_gemm_gpu_hip M/N/K gemm_type seed nb_runs [time_with_copy] [time_with_conversion]
 */

#define CHUNKSIZE 32 

/****************************************************************************************************/
/**
 * @brief CUDA kernel to convert double precision values to bfloat16
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input double precision matrix
 * @param ldf Leading dimension of input matrix
 * @param H Output bfloat16 matrix
 * @param ldh Leading dimension of output matrix
 */
__global__ void double2bf_GPU_kernel( int nrows, int ncols,
                const double *F, int ldf,
                hip_bfloat16 *H, int ldh ) {
        const int tx=hipThreadIdx_x;
        const int ty=hipThreadIdx_y;
        const int idx= hipBlockIdx_x * hipBlockDim_x + tx;
        const int idy= hipBlockIdx_y * hipBlockDim_y + ty;

        if( idx >= nrows || idy >= ncols ) { return; }
        H[idy*ldh+idx]= H[idy*ldh+idx].round_to_bfloat16( (float)(F[idy*ldf+idx]) );
}

/**
 * @brief Launch double to bfloat16 conversion kernel
 * @param nrows Number of rows
 * @param ncols Number of columns
 * @param F Input double precision matrix
 * @param ldf Leading dimension of input matrix
 * @param _H Output bfloat16 matrix
 * @param ldh Leading dimension of output matrix
 * @param stream HIP stream for execution
 */
extern "C"
void double2bf_GPU( int nrows, int ncols,
                const double *F, int ldf,
                void *_H, int ldh,
                hipStream_t stream ) {
        int nBlockx= (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky= (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        hip_bfloat16 *H = (hip_bfloat16 *)_H;
        double2bf_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}
/****************************************************************************************************/

/****************************************************************************************************/
/**
 * @brief CUDA kernel to convert single precision values to bfloat16
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input single precision matrix
 * @param ldf Leading dimension of input matrix
 * @param H Output bfloat16 matrix
 * @param ldh Leading dimension of output matrix
 */
__global__ void float2bf_GPU_kernel( int nrows, int ncols,
                const float *F, int ldf,
                hip_bfloat16 *H, int ldh ) {
        const int tx=hipThreadIdx_x;
        const int ty=hipThreadIdx_y;
        const int idx= hipBlockIdx_x * hipBlockDim_x + tx;
        const int idy= hipBlockIdx_y * hipBlockDim_y + ty;

        if( idx >= nrows || idy >= ncols ) { return; }
        H[idy*ldh+idx]= H[idy*ldh+idx].round_to_bfloat16( F[idy*ldf+idx] );
}

/**
 * @brief Launch float to bfloat16 conversion kernel
 * @param nrows Number of rows
 * @param ncols Number of columns
 * @param F Input single precision matrix
 * @param ldf Leading dimension of input matrix
 * @param _H Output bfloat16 matrix
 * @param ldh Leading dimension of output matrix
 * @param stream HIP stream for execution
 */
extern "C"
void float2bf_GPU( int nrows, int ncols,
                const float *F, int ldf,
                void *_H, int ldh,
                hipStream_t stream ) {
        int nBlockx= (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky= (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        hip_bfloat16 *H = (hip_bfloat16 *)_H;
        float2bf_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}
/****************************************************************************************************/

/**
 * @brief CUDA kernel to convert bfloat16 values to single precision
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param H Input bfloat16 matrix
 * @param ldh Leading dimension of input matrix
 * @param F Output single precision matrix
 * @param ldf Leading dimension of output matrix
 */
__global__ void bf2float_GPU_kernel( int nrows, int ncols,
                const hip_bfloat16 *H, int ldh,
                float *F, int ldf ) {
        const int tx=hipThreadIdx_x;
        const int ty=hipThreadIdx_y;
        const int idx= hipBlockIdx_x * hipBlockDim_x + tx;
        const int idy= hipBlockIdx_y * hipBlockDim_y + ty;

        if( idx >= nrows || idy >= ncols ) { return; }
        F[idy * ldf + idx] = (float)H[idy * ldh + idx];
}

/**
 * @brief Launch bfloat16 to float conversion kernel
 * @param nrows Number of rows
 * @param ncols Number of columns
 * @param _H Input bfloat16 matrix
 * @param ldh Leading dimension of input matrix
 * @param F Output single precision matrix
 * @param ldf Leading dimension of output matrix
 * @param stream HIP stream for execution
 */
extern "C"
void bf2float_GPU( int nrows, int ncols,
                const void *_H, int ldh,
                float *F, int ldf,
                hipStream_t stream ) {
        int nBlockx= (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky= (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        hip_bfloat16 *H = (hip_bfloat16 *)_H;
        bf2float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, H, ldh, F, ldf);
}
/****************************************************************************************************/

/****************************************************************************************************/
/**
 * @brief CUDA kernel to convert double precision values to half precision
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input double precision matrix
 * @param ldf Leading dimension of input matrix
 * @param H Output half precision matrix
 * @param ldh Leading dimension of output matrix
 */
__global__ void double2half_GPU_kernel( int nrows, int ncols,
                const double *F, int ldf,
                __half *H, int ldh ) {
        const int tx=hipThreadIdx_x;
        const int ty=hipThreadIdx_y;
        const int idx= hipBlockIdx_x * hipBlockDim_x + tx;
        const int idy= hipBlockIdx_y * hipBlockDim_y + ty;

        if( idx >= nrows || idy >= ncols ) { return; }
        H[idy*ldh+idx]= __float2half( (float)(F[idy*ldf+idx]) );
}

/**
 * @brief Launch double to half conversion kernel
 * @param nrows Number of rows
 * @param ncols Number of columns
 * @param F Input double precision matrix
 * @param ldf Leading dimension of input matrix
 * @param _H Output half precision matrix
 * @param ldh Leading dimension of output matrix
 * @param stream HIP stream for execution
 */
extern "C"
void double2half_GPU( int nrows, int ncols,
                const double *F, int ldf,
                void *_H, int ldh,
                hipStream_t stream ) {
        int nBlockx= (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky= (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        __half *H = (__half *)_H;
        double2half_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}
/****************************************************************************************************/

/****************************************************************************************************/
/**
 * @brief CUDA kernel to convert single precision values to half precision
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param F Input single precision matrix
 * @param ldf Leading dimension of input matrix
 * @param H Output half precision matrix
 * @param ldh Leading dimension of output matrix
 */
__global__ void float2half_GPU_kernel( int nrows, int ncols,
                const float *F, int ldf,
                __half *H, int ldh ) {
        const int tx=hipThreadIdx_x;
        const int ty=hipThreadIdx_y;
        const int idx= hipBlockIdx_x * hipBlockDim_x + tx;
        const int idy= hipBlockIdx_y * hipBlockDim_y + ty;

        if( idx >= nrows || idy >= ncols ) { return; }
        H[idy*ldh+idx]= __float2half_rn( F[idy*ldf+idx] );
}

/**
 * @brief Launch float to half conversion kernel
 * @param nrows Number of rows
 * @param ncols Number of columns
 * @param F Input single precision matrix
 * @param ldf Leading dimension of input matrix
 * @param _H Output half precision matrix
 * @param ldh Leading dimension of output matrix
 * @param stream HIP stream for execution
 */
extern "C"
void float2half_GPU( int nrows, int ncols,
                const float *F, int ldf,
                void *_H, int ldh,
                hipStream_t stream ) {
        int nBlockx= (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky= (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        __half *H = (__half *)_H;
        float2half_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}
/****************************************************************************************************/

/**
 * @brief CUDA kernel to convert half precision values to single precision
 * @param nrows Number of rows in the matrix
 * @param ncols Number of columns in the matrix
 * @param H Input half precision matrix
 * @param ldh Leading dimension of input matrix
 * @param F Output single precision matrix
 * @param ldf Leading dimension of output matrix
 */
__global__ void half2float_GPU_kernel( int nrows, int ncols,
                const __half *H, int ldh,
                float *F, int ldf ) {
        const int tx=hipThreadIdx_x;
        const int ty=hipThreadIdx_y;
        const int idx= hipBlockIdx_x * hipBlockDim_x + tx;
        const int idy= hipBlockIdx_y * hipBlockDim_y + ty;

        if( idx >= nrows || idy >= ncols ) { return; }

        F[idy * ldf + idx] = __half2float(H[idy * ldh + idx]);
}

/**
 * @brief Launch half to float conversion kernel
 * @param nrows Number of rows
 * @param ncols Number of columns
 * @param _H Input half precision matrix
 * @param ldh Leading dimension of input matrix
 * @param F Output single precision matrix
 * @param ldf Leading dimension of output matrix
 * @param stream HIP stream for execution
 */
extern "C"
void half2float_GPU( int nrows, int ncols,
                const void *_H, int ldh,
                float *F, int ldf,
                hipStream_t stream ) {
        int nBlockx= (nrows+CHUNKSIZE-1)/CHUNKSIZE;
        int nBlocky= (ncols+CHUNKSIZE-1)/CHUNKSIZE;
        dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
        dim3 dimGrid(nBlockx, nBlocky);
        __half *H = (__half *)_H;
        half2float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, H, ldh, F, ldf);
}
/****************************************************************************************************/

/**
 * @brief Calculate Frobenius norm of the difference between two matrices
 * @param A First matrix
 * @param A_ref Reference matrix (double precision)
 * @param M Number of rows
 * @param N Number of columns
 * @param gemm_type Data type of matrix A
 * @return Frobenius norm of the difference
 */
static double frobenius_norm_diff(void *A, double *A_ref, int M, int N, int gemm_type) {
    double norm = 0.0;
    switch( gemm_type ) {
        case 1:  // Double precision
            {
                double *_A = (double *)A;
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        double diff = _A[i * N + j] - A_ref[i * N + j];
                        norm += diff * diff;
                    }
                }
            }
            break;

        default:  // Single precision
            {
                float *_A = (float *)A;
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        double diff = _A[i * N + j] - A_ref[i * N + j];
                        norm += diff * diff;
                    }
                }
            }
            break;
    }

    return sqrt(norm);
}

/**
 * @brief Calculate Frobenius norm of a matrix
 * @param A Input matrix
 * @param M Number of rows
 * @param N Number of columns
 * @param gemm_type Data type of matrix A
 * @return Frobenius norm of the matrix
 */
static double frobenius_norm(void *A, int M, int N, int gemm_type) {
    double norm = 0.0;
    switch( gemm_type ) {
        case 1:  // Double precision
            {
                double *_A = (double *)A;
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        double diff = _A[i * N + j];
                        norm += diff * diff;
                    }
                }
            }
            break;

        default:  // Single precision
            {
                float *_A = (float *)A;
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        double diff = _A[i * N + j];
                        norm += diff * diff;
                    }
                }
            }
            break;

    }

    return sqrt(norm);
}

/**
 * @brief Print matrix contents for debugging
 * @param A Matrix to print
 * @param M Number of rows
 * @param N Number of columns
 * @param gemm_type Data type of matrix A
 */
static void print_matrix(void *A, int M, int N, int gemm_type) {
    switch( gemm_type ) {
        case 1:  // Double precision
            {
                double *_A = (double *)A;
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        printf("%lf ", _A[i * N + j]);
                    }
                    printf("\n");
                }
            }
            break;

        default:  // Single precision
            {
                float *_A = (float *)A;
                for (int i = 0; i < M; i++) {
                    for (int j = 0; j < N; j++) {
                        printf("%f ", _A[i * N + j]);
                    }
                    printf("\n");
                }
            }
            break;

    }
}

/**
 * @brief Parse integer command line arguments
 * @param arg String argument to parse
 * @return Parsed integer value
 */
static int parse_arg(const char *arg) {
    int value;
    if (sscanf(arg, "%d", &value) != 1) {
        fprintf(stderr, "Invalid argument: %s\n", arg);
        exit(EXIT_FAILURE);
    }
    return value;
}

/**
 * @brief Initialize host matrices with random values
 * @param A Pointer to store the allocated matrix
 * @param A_ref Pointer to store the reference matrix (double precision)
 * @param M Number of rows
 * @param N Number of columns
 * @param gemm_type Data type of matrix A
 * @param seed Random seed for reproducible results
 */
static void init_matrix_host( void **A, void **A_ref, int M, int N, int gemm_type, int seed ) {
    (void)hipHostMalloc((void**)A_ref, M * N * sizeof(double));  // Allocate pinned host memory for reference
    double *_A_ref = (double *)(*A_ref);
    srand( seed );

    switch( gemm_type ) {
        case 1:  // Double precision
            {
                (void)hipHostMalloc((void**)A, M * N * sizeof(double));
                double *_A = (double *)(*A);
                for( int i = 0; i < M; i++ ) {
                    for( int j = 0; j < N; j++ ) {
                        _A[i*N+j] = (double)rand()/RAND_MAX;  // Generate random values in [0,1)
                        _A_ref[i*N+j] = (double)_A[i*N+j];    // Store reference values
                    }
                }
            }
            break;

        default:  // Single precision
            {
                (void)hipHostMalloc((void**)A, M * N * sizeof(float));
                float *_A = (float *)(*A);
                for( int i = 0; i < M; i++ ) {
                    for( int j = 0; j < N; j++ ) {
                        _A[i*N+j] = (float)rand()/(float)RAND_MAX;   // Generate random values in [0,1)
                        _A_ref[i*N+j] = (double)_A[i*N+j];    // Store reference values
                    }
                }
            }
            break;

    }
}

/**
 * @brief Allocate device memory for matrices
 * @param A Pointer to store device matrix A
 * @param A_ref Pointer to store device reference matrix
 * @param M Number of rows
 * @param N Number of columns
 * @param gemm_type Data type of matrix A
 */
static void allocate_memory_device( void **A, void **A_ref, int M, int N, int gemm_type ) {
    (void)hipMalloc(A_ref, M * N * sizeof(double));  // Always allocate double precision for reference
    switch( gemm_type ) {
        case 1:  // Double precision
            (void)hipMalloc(A, M * N * sizeof(double));
            break;

        default:  // Single precision
            (void)hipMalloc(A, M * N * sizeof(float));
            break;
    }
}

/**
 * @brief Copy data between host and device with proper data type handling
 * @param d_A Device matrix pointer
 * @param h_A Host matrix pointer
 * @param M Number of rows
 * @param N Number of columns
 * @param flag Memory copy direction
 * @param gemm_type Data type of matrix A
 * @param stream HIP stream for execution
 */
static void my_cuda_copy( void *d_A, void *h_A, int M, int N,
        enum hipMemcpyKind flag, int gemm_type, hipStream_t stream) {
    switch( gemm_type ) {
        case 1:  // Double precision
            (void)hipMemcpyAsync(d_A, h_A, M * N * sizeof(double), flag, stream);
            break;

        default:  // Single precision
            (void)hipMemcpyAsync(d_A, h_A, M * N * sizeof(float), flag, stream);
            break;
    }
}

/**
 * @brief Perform GEMM operation with specified precision and parameters
 * @param handle hipBLAS handle
 * @param M Number of rows in matrix A and C
 * @param N Number of columns in matrix B and C
 * @param K Number of columns in matrix A and rows in matrix B
 * @param d_A Device matrix A
 * @param d_B Device matrix B
 * @param d_C Device matrix C
 * @param gemm_type Data type and computation mode
 */
static void my_gemm(hipblasHandle_t handle, int M, int N, int K, void *d_A, void *d_B, void *d_C, int gemm_type ) {
    switch( gemm_type ) {
        case 1:  // Double precision
            {
                double alpha = 1.0;
                double beta = 0.0;
                hipblasGemmEx(handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,  // No transpose on A and B
                        M, N, K,                      // Matrix dimensions
                        &alpha,                       // Scaling factor
                        d_A, HIPBLAS_R_64F, M,        // Matrix A, double precision, leading dimension M
                        d_B, HIPBLAS_R_64F, K,        // Matrix B, double precision, leading dimension K
                        &beta,                        // Scaling factor for C
                        d_C, HIPBLAS_R_64F, M,        // Matrix C, double precision, leading dimension M
                        HIPBLAS_R_64F,                // Compute type (double precision)
                        HIPBLAS_GEMM_DEFAULT);        // Algorithm selection
            }
            break;

        case 2:  // Single precision
            {
                float alpha = 1.0f;
                float beta = 0.0f;
                hipblasGemmEx(handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,  // No transpose on A and B
                        M, N, K,                      // Matrix dimensions
                        &alpha,                       // Scaling factor
                        d_A, HIPBLAS_R_32F, M,        // Matrix A, single precision, leading dimension M
                        d_B, HIPBLAS_R_32F, K,        // Matrix B, single precision, leading dimension K
                        &beta,                        // Scaling factor for C
                        d_C, HIPBLAS_R_32F, M,        // Matrix C, single precision, leading dimension M
                        HIPBLAS_R_32F,                // Compute type (single precision)
                        HIPBLAS_GEMM_DEFAULT);        // Algorithm selection
            }
            break;

        case 3:  // TF32 (Tensor Float 32)
            {
                float alpha = 1.0f;
                float beta = 0.0f;
                hipblasGemmEx(handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,  // No transpose on A and B
                        M, N, K,                      // Matrix dimensions
                        &alpha,                       // Scaling factor
                        d_A, HIPBLAS_R_32F, M,        // Matrix A, single precision, leading dimension M
                        d_B, HIPBLAS_R_32F, K,        // Matrix B, single precision, leading dimension K
                        &beta,                        // Scaling factor for C
                        d_C, HIPBLAS_R_32F, M,        // Matrix C, single precision, leading dimension M
                        CUBLAS_COMPUTE_32F_FAST_TF32, // Compute type (TF32)
                        HIPBLAS_GEMM_DEFAULT);        // Algorithm selection
            }
            break;

        case 4:  // FP16 with FP32 accumulation
            {
                float alpha = 1.0f;
                float beta = 0.0f;
                hipblasGemmEx(handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,  // No transpose on A and B
                        M, N, K,                      // Matrix dimensions
                        &alpha,                       // Scaling factor
                        d_A, HIPBLAS_R_16F, M,        // Matrix A, half precision, leading dimension M
                        d_B, HIPBLAS_R_16F, K,        // Matrix B, half precision, leading dimension K
                        &beta,                        // Scaling factor for C
                        d_C, HIPBLAS_R_32F, M,        // Matrix C, single precision, leading dimension M
                        HIPBLAS_R_32F,                // Compute type (single precision)
                        HIPBLAS_GEMM_DEFAULT);        // Algorithm selection
            }
            break;

        case 5:  // FP16 with FP16 accumulation
            {
                hipblasHalf alpha = (hipblasHalf)1.0f;
                hipblasHalf beta = (hipblasHalf)0.0f;
#if 0
                hipblasStatus_t status = hipblasGemmEx(handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,  // No transpose on A and B
                        M, N, K,                      // Matrix dimensions
                        &alpha,                       // Scaling factor
                        d_A, HIPBLAS_R_16F, M,        // Matrix A, half precision, leading dimension M
                        d_B, HIPBLAS_R_16F, K,        // Matrix B, half precision, leading dimension K
                        &beta,                        // Scaling factor for C
                        d_C, HIPBLAS_R_16F, M,        // Matrix C, half precision, leading dimension M
                        HIPBLAS_R_16F,                // Compute type (half precision)
                        HIPBLAS_GEMM_DEFAULT);        // Algorithm selection
#else
                hipblasStatus_t status = hipblasHgemm(handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,  // No transpose on A and B
                        M, N, K,                      // Matrix dimensions
                        &alpha,                       // Scaling factor
                        (hipblasHalf *)d_A, M,        // Matrix A, half precision, leading dimension M
                        (hipblasHalf *)d_B, K,        // Matrix B, half precision, leading dimension K
                        &beta,                        // Scaling factor for C
                        (hipblasHalf *)d_C, M);       // Matrix C, half precision, leading dimension M
#endif
                if( HIPBLAS_STATUS_SUCCESS != status ) {
                    printf("ERROR: hipBLAS operation failed with status %d\n", status);
                }
            }
            break;

        case 6:  // BF16 with FP32 accumulation
            {
                float alpha = 1.0f;
                float beta = 0.0f;
                hipblasGemmEx(handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,  // No transpose on A and B
                        M, N, K,                      // Matrix dimensions
                        &alpha,                       // Scaling factor
                        d_A, HIPBLAS_R_16B, M,        // Matrix A, bfloat16, leading dimension M
                        d_B, HIPBLAS_R_16B, K,        // Matrix B, bfloat16, leading dimension K
                        &beta,                        // Scaling factor for C
                        d_C, HIPBLAS_R_32F, M,        // Matrix C, single precision, leading dimension M
                        HIPBLAS_R_32F,                // Compute type (single precision)
                        HIPBLAS_GEMM_DEFAULT);        // Algorithm selection
            }
            break;

        case 7:  // BF16 with BF16 accumulation
            {
                float alpha = 1.0f;
                float beta = 0.0f;
                hipblasGemmEx(handle,
                        HIPBLAS_OP_N, HIPBLAS_OP_N,  // No transpose on A and B
                        M, N, K,                      // Matrix dimensions
                        &alpha,                       // Scaling factor
                        d_A, HIPBLAS_R_16B, M,        // Matrix A, bfloat16, leading dimension M
                        d_B, HIPBLAS_R_16B, K,        // Matrix B, bfloat16, leading dimension K
                        &beta,                        // Scaling factor for C
                        d_C, HIPBLAS_R_16B, M,        // Matrix C, bfloat16, leading dimension M
                        HIPBLAS_R_32F,                // Compute type (single precision)
                        HIPBLAS_GEMM_DEFAULT);        // Algorithm selection
            }
            break;

    }
}

/* GEMM type definitions:
 * 1: FP64 (Double precision)
 * 2: FP32 (Single precision)
 * 3: TP32 (Tensor Float 32)
 * 4: FP16: A16B16C32OP32 (Half precision with FP32 accumulation)
 * 5: FP16: A16B16C16OP16 (Half precision with FP16 accumulation)
 * 6: BF16: A16B16C32OP32 (Bfloat16 with FP32 accumulation)
 * 7: BF16: A16B16C16OP16 (Bfloat16 with BF16 accumulation)
 */

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 7) {
        fprintf(stderr, "Usage: %s M/N/K gemm_type seed nb_runs [time_with_copy, default: 0] [time_with_conversion, default 0]\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    // Parse command line arguments
    int M = parse_arg(argv[1]);
    int N = M; 
    int K = M; 
    int gemm_type = parse_arg(argv[2]);
    int seed = 0;
    if (argc >= 4) {
        seed = parse_arg(argv[3]);
    }
    int nb_runs = 3;
    if (argc >= 5) {
        nb_runs = parse_arg(argv[4]);
    }
    int time_with_copy = 0;
    if (argc >= 6) {
        time_with_copy = parse_arg(argv[5]);
    }
    int time_with_conversion = 0;
    if (argc >= 7) {
        time_with_conversion = parse_arg(argv[6]);
    }

    // Validate input parameters
    if( 0 == time_with_conversion && 1 == time_with_copy ) {
        printf("No support for time_with_copy=1 when time_with_conversion=0!\n");
        return 0;
    }

    printf("Starting HIP-based GEMM test with matrix size %d x %d\n", M, N);
    printf("GEMM type: %d, Seed: %d, Runs: %d\n", gemm_type, seed, nb_runs);
    printf("Time with copy: %d, Time with conversion: %d\n", time_with_copy, time_with_conversion);

    // Create hipBLAS handle
    hipblasHandle_t handle;
    hipblasCreate(&handle);
    
    // Create HIP stream for asynchronous execution
    hipStream_t stream;
    (void)hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
    hipblasSetStream(handle, stream);

    // Host and device memory pointers
    void *h_A, *h_B, *h_C, *h_A_ref, *h_B_ref, *h_C_ref;
    void *d_A, *d_B, *d_C, *d_C0, *d_A_ref, *d_B_ref, *d_C_ref;

    // Initialize host matrices
    init_matrix_host( &h_A, &h_A_ref, M, K, gemm_type, seed );
    init_matrix_host( &h_B, &h_B_ref, K, N, gemm_type, seed );
    init_matrix_host( &h_C, &h_C_ref, M, N, gemm_type, seed );

#if 0
    printf("Original matrix A:\n");
    print_matrix( h_A, M, N, gemm_type );
    printf("Reference matrix A:\n");
    print_matrix( h_A_ref, M, N, 1 );
#endif

    // Allocate device memory
    allocate_memory_device( (void **)&d_A, (void **)&d_A_ref, M, K, gemm_type );
    allocate_memory_device( (void **)&d_B, (void **)&d_B_ref, K, N, gemm_type );
    allocate_memory_device( (void **)&d_C, (void **)&d_C_ref, M, N, gemm_type );
    (void)hipMalloc((void **)&d_C0, M * N * sizeof(float)/2);  // Temporary storage for half precision results

    // Copy reference matrices to device
    my_cuda_copy( d_A_ref, h_A_ref, M, K, hipMemcpyHostToDevice, 1, stream);
    my_cuda_copy( d_B_ref, h_B_ref, K, N, hipMemcpyHostToDevice, 1, stream);
    
    // Copy input matrices to device if not timing with copy
    if( !time_with_copy ) {
        my_cuda_copy( d_A, h_A, M, K, hipMemcpyHostToDevice, gemm_type, stream );
        my_cuda_copy( d_B, h_B, K, N, hipMemcpyHostToDevice, gemm_type, stream );
    }

    // Perform data type conversions if not timing with conversion
    if( 4 == gemm_type || 5 == gemm_type ) {
        if( !time_with_conversion ) {
            double2half_GPU( M, K, (double *)d_A_ref, M, d_A, M, stream );
            double2half_GPU( K, M, (double *)d_B_ref, K, d_B, K, stream );
        }
    } else if( 6 == gemm_type || 7 == gemm_type ) {
        if( !time_with_conversion ) {
            double2bf_GPU( M, K, (double *)d_A_ref, M, d_A, M, stream );
            double2bf_GPU( K, M, (double *)d_B_ref, K, d_B, K, stream );
        }
    }

    printf("Starting benchmark runs...\n");
    printf("----------------------------------------\n");

    // Main benchmark loop
    double start_time, end_time;
    struct timeval tstart, tend;

    for (int i = 0; i < nb_runs; ++i) {
        (void)hipDeviceSynchronize();
        gettimeofday(&tstart, NULL);
        start_time = tstart.tv_sec + tstart.tv_usec / 1.0e6;

        // Copy host matrices to device if timing with copy
        if( time_with_copy ) {
            my_cuda_copy( d_A, h_A, M, K, hipMemcpyHostToDevice, gemm_type, stream );
            my_cuda_copy( d_B, h_B, K, N, hipMemcpyHostToDevice, gemm_type, stream );
        }

        (void)hipStreamSynchronize(stream);
        
        // Perform GEMM operation based on precision type
        if( 5 == gemm_type ) {  // FP16 with FP16 accumulation
            if( time_with_conversion ) {
                double2half_GPU( M, K, (double *)d_A_ref, M, d_A, M, stream );
                double2half_GPU( K, M, (double *)d_B_ref, K, d_B, K, stream );
            }
            my_gemm( handle, M, N, K, d_A, d_B, d_C0, gemm_type );
            if( time_with_conversion ) {
                half2float_GPU( M, N, d_C0, M, (float *)d_C, M, stream );
            }
        } else if( 7 == gemm_type ) {  // BF16 with BF16 accumulation
            if( time_with_conversion ) {
                double2bf_GPU( M, K, (double *)d_A_ref, M, d_A, M, stream );
                double2bf_GPU( K, M, (double *)d_B_ref, K, d_B, K, stream );
            }
            my_gemm( handle, M, N, K, d_A, d_B, d_C0, gemm_type );
            if( time_with_conversion ) {
                bf2float_GPU( M, N, d_C0, M, (float *)d_C, M, stream );
            }
        } else {  // Other precision types
            my_gemm( handle, M, N, K, d_A, d_B, d_C, gemm_type );
        }

        // Copy result back to host if timing with copy
        if( time_with_copy ) {
            my_cuda_copy( h_C, d_C, M, N, hipMemcpyDeviceToHost, gemm_type, stream );
        }

        (void)hipStreamSynchronize(stream);
        (void)hipDeviceSynchronize();
        gettimeofday(&tend, NULL);
        end_time = tend.tv_sec + tend.tv_usec / 1.0e6;

        // Perform data type conversions if not timing with conversion
        if( 5 == gemm_type ) {
            if( !time_with_conversion ) {
                half2float_GPU( M, N, d_C0, M, (float *)d_C, M, stream );
            }
        } else if( 7 == gemm_type ) {
            if( !time_with_conversion ) {
                bf2float_GPU( M, N, d_C0, M, (float *)d_C, M, stream );
            }
        }

        // Copy result back to host if not timing with copy
        if( !time_with_copy ) {
            my_cuda_copy( h_C, d_C, M, N, hipMemcpyDeviceToHost, gemm_type, stream );
        }

        // Calculate reference result using double precision
        my_gemm( handle, M, N, K, d_A_ref, d_B_ref, d_C_ref, 1 );
        my_cuda_copy( h_C_ref, d_C_ref, M, N, hipMemcpyDeviceToHost, 1, stream );
        (void)hipStreamSynchronize(stream);

        // Calculate Frobenius norms for validation
        double norm_diff = frobenius_norm_diff(h_C, (double *)h_C_ref, M, N, gemm_type);
        double norm_C = frobenius_norm( h_C, M, N, gemm_type );
        double norm_C_ref = frobenius_norm( h_C_ref, M, N, 1 );

        // Calculate and display performance metrics
        double gflops = 2.0 * M * N * K / (1e12);
        if( i > 0 ) {
            printf("GEMM: %d %d %d %d %d %d %d : %.3f TFLOPS : %.6e %.6e %.2e\n",
                M, N, K, gemm_type, seed, time_with_copy, time_with_conversion, 
                gflops / (end_time - start_time), norm_C, norm_C_ref, norm_diff/norm_C_ref);
        }

        //printf("Frobenius_norm: C %g C_ref %g ||C-C_ref||/||C_ref||%g\n", norm_C, norm_C_ref, norm_diff);
    }

#if 0
    printf("Result matrix C:\n");
    print_matrix( h_C, M, N, gemm_type );
    printf("Reference matrix C:\n");
    print_matrix( h_C_ref, M, N, 1 );
#endif

    printf("----------------------------------------\n");
    printf("Benchmark completed successfully\n");

    // Clean up allocated resources
    (void)hipHostFree(h_A);
    (void)hipHostFree(h_B);
    (void)hipHostFree(h_C);
    (void)hipHostFree(h_A_ref);
    (void)hipHostFree(h_B_ref);
    (void)hipHostFree(h_C_ref);
    (void)hipFree(d_A);
    (void)hipFree(d_B);
    (void)hipFree(d_C);
    (void)hipFree(d_C0);
    (void)hipFree(d_A_ref);
    (void)hipFree(d_B_ref);
    (void)hipFree(d_C_ref);
    hipblasDestroy(handle);

    return 0;
}
