/**
 * @file gemm_f16_wrapper.c
 * @brief HICMA GEMM F16 (Half Precision General Matrix Multiply) wrapper implementation
 * 
 * This file provides a wrapper function for half-precision (FP16) GEMM operations
 * using Intel MKL (Math Kernel Library). It performs matrix multiplication with
 * FP16 input matrices and FP32 output, providing a bridge between half-precision
 * computation and single-precision results.
 * 
 * The function uses MKL's cblas_gemm_f16f16f32 routine which:
 * - Takes FP16 input matrices A and B
 * - Performs the computation: C = α * op(A) * op(B) + β * C
 * - Returns results in FP32 matrix C
 * 
 * This is useful for memory-constrained applications where input data can be
 * stored in half precision but higher precision is needed for the output.
 */

// #include "hicma_parsec.h"
#include <stdio.h>

#ifdef HAVE_MKL
#define MKL_ILP64
#include "mkl.h"
#endif

/**
 * @brief Perform half-precision GEMM operation using MKL
 * 
 * This function performs general matrix multiplication using half-precision (FP16)
 * input matrices and single-precision (FP32) output. It serves as a wrapper around
 * Intel MKL's optimized cblas_gemm_f16f16f32 routine.
 * 
 * @param[in] Layout Matrix layout (CblasRowMajor or CblasColMajor)
 * @param[in] TransA Transpose operation for matrix A (CblasNoTrans, CblasTrans, CblasConjTrans)
 * @param[in] TransB Transpose operation for matrix B (CblasNoTrans, CblasTrans, CblasConjTrans)
 * @param[in] M Number of rows of matrix op(A) and C
 * @param[in] N Number of columns of matrix op(B) and C
 * @param[in] K Number of columns of matrix op(A) and rows of matrix op(B)
 * @param[in] alpha Scalar multiplier for the matrix product
 * @param[in] A Input matrix A (FP16 precision)
 * @param[in] lda Leading dimension of matrix A
 * @param[in] B Input matrix B (FP16 precision)
 * @param[in] ldb Leading dimension of matrix B
 * @param[in] beta Scalar multiplier for matrix C
 * @param[in,out] C Output matrix C (FP32 precision, input/output)
 * @param[in] ldc Leading dimension of matrix C
 */
void my_gemm_f16(const int Layout, const int TransA, const int TransB,
                 const int M, const int N, const int K,
                 const float alpha,
                 const void* A, const int lda,
                 const void* B, const int ldb,
                 const float beta,
                 float* C, const int ldc)
{
#ifdef HAVE_MKL
       /* Call MKL's optimized half-precision GEMM routine */
       cblas_gemm_f16f16f32(Layout, TransA, TransB,
              M, N, K,
              alpha, A, lda,
                     B, ldb,
              beta,  C, ldc);
       // printf("cblas_gemm_f16f16f32\n");
#else
       /* MKL not available - provide a stub implementation */
       fprintf(stderr, "Warning: MKL not available, gemm_f16_wrapper is not functional\n");
       /* Zero out the output matrix as a safe fallback */
       for (int i = 0; i < M; i++) {
           for (int j = 0; j < N; j++) {
               C[i * ldc + j] = 0.0f;
           }
       }
#endif
}
