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
//#endif

//#if defined(PARSEC_HAVE_DEV_HIP_SUPPORT)
//#include "hicma_parsec_hip_cuda.h"
//#endif

/**
 * @file testing_gemm_gpu.cu
 * @brief Comprehensive GPU GEMM performance testing program with mixed precision support
 * 
 * This program tests the performance of various precision GEMM operations using NVIDIA cuBLAS
 * and cuBLASLt libraries. It supports multiple data types including FP64, FP32, FP16, BF16, FP8,
 * and INT8, with comprehensive benchmarking and validation capabilities.
 * 
 * Supported GEMM types:
 * 1: FP64 (Double precision)
 * 2: FP32 (Single precision)
 * 3: TF32 (Tensor Float 32)
 * 4: FP16 with FP32 accumulation (A16B16C32OP32)
 * 5: FP16 with FP16 accumulation (A16B16C16OP16)
 * 6: BF16 with FP32 accumulation (A16B16C32OP32)
 * 7: BF16 with BF16 accumulation (A16B16C16OP16)
 * 8: FP8 with FP32 accumulation (A8B8C32OP32)
 * 9: INT8 with INT32 accumulation (A8B8C32OP32)
 * 
 * Usage: ./testing_gemm_gpu M/N/K gemm_type nb_runs seed [time_with_copy] [time_with_conversion]
 */

namespace mixed_kernels {
#define CHECK_CUBLAS(statement)                        \
    do {                                                 \
        auto status = statement;                           \
        if (status != CUBLAS_STATUS_SUCCESS) {             \
            printf("Failed at %s:%d\n", __FILE__, __LINE__); \
        }                                                  \
        \
    } while (0)

#define CHECK_CUDA(statement)                        \
    do {                                                 \
        auto status = statement;                           \
        if (status != cudaSuccess) {             \
            printf("Failed at %s:%d\n", __FILE__, __LINE__); \
        }                                                  \
        \
    } while (0)

    namespace cublasLtFp8RowMajorNTNMeta {
        namespace {
            struct Impl {
                constexpr static int searchAlgoNum = 1;
                const uint64_t m, n, k;
                const int64_t lda, ldb, ldc;
                const std::size_t workspaceSize;
                void *workspace;
                cublasLtHandle_t handle{nullptr};
                const cublasOperation_t transa{CUBLAS_OP_T}, transb{CUBLAS_OP_N};
                cublasLtMatmulPreference_t preference{nullptr};
                cublasLtMatrixLayout_t Adesc{nullptr}, Bdesc{nullptr};
                cublasLtMatmulDesc_t operationDesc{nullptr};
                struct {
                    cublasLtMatrixLayout_t Cdesc{nullptr}, Ddesc{nullptr};
                    cublasLtMatmulHeuristicResult_t heuristicResult[searchAlgoNum];
                    int algoNum = searchAlgoNum;
                } fp8fp8fp16{}, fp8fp8fp32{};
            };
        }  // namespace

        cublasStatus_t CUBLASWINAPI
            create(void **instance, uint64_t m, uint64_t n, uint64_t k, int64_t lda,
                    int64_t ldb, int64_t ldc, std::size_t workspaceSize /* 32MB for H100 is a good choice */,
                    void *workspace);

        cublasStatus_t CUBLASWINAPI destroy(void *instance);

        static cublasStatus_t CUBLASWINAPI
            matmul(void *instance, cudaStream_t stream, double alpha, const void *A,
                    const void *B, double beta, void *C, cudaDataType Ctype, int algoIdx = 0);

        static cublasStatus_t CUBLASWINAPI
            getAlgoNum(void *instance, cudaDataType Ctype, int *algoNum);
    }  // namespace cublasLtFp8RowMajorNTNMeta

    cublasStatus_t cublasLtFp8RowMajorNTNMeta::create(
            void **instance, uint64_t m, uint64_t n, uint64_t k, int64_t lda,
            int64_t ldb, int64_t ldc, std::size_t workspaceSize, void *workspace) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        if (deviceProp.major < 9) {
            return CUBLAS_STATUS_SUCCESS;
        }
        auto impl = new Impl{m, n, k, lda, ldb, ldc, workspaceSize, workspace};
        CHECK_CUBLAS(cublasLtCreate(&impl->handle));
        CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&impl->preference));
        CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
                    impl->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &workspaceSize, sizeof(workspaceSize)));
        CHECK_CUBLAS(
                cublasLtMatrixLayoutCreate(&impl->Adesc, CUDA_R_8F_E4M3, k, m, lda));
        CHECK_CUBLAS(
                cublasLtMatrixLayoutCreate(&impl->Bdesc, CUDA_R_8F_E4M3, k, n, ldb));
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&impl->operationDesc,
                    CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
                    impl->operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &impl->transa,
                    sizeof(impl->transa)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
                    impl->operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &impl->transb,
                    sizeof(impl->transb)));

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp16.Cdesc, CUDA_R_16F,
                    n, m, ldc));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp16.Ddesc, CUDA_R_16F,
                    n, m, ldc));
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
                    impl->handle, impl->operationDesc, impl->Bdesc, impl->Adesc,
                    impl->fp8fp8fp16.Cdesc, impl->fp8fp8fp16.Ddesc, impl->preference, impl->fp8fp8fp16.algoNum,
                    impl->fp8fp8fp16.heuristicResult, &impl->fp8fp8fp16.algoNum));

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp32.Cdesc, CUDA_R_32F,
                    n, m, ldc));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp32.Ddesc, CUDA_R_32F,
                    n, m, ldc));
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
                    impl->handle, impl->operationDesc, impl->Bdesc, impl->Adesc,
                    impl->fp8fp8fp32.Cdesc, impl->fp8fp8fp32.Ddesc, impl->preference, impl->fp8fp8fp32.algoNum,
                    impl->fp8fp8fp32.heuristicResult, &impl->fp8fp8fp32.algoNum));
        *instance = impl;
        return CUBLAS_STATUS_SUCCESS;
    }

    cublasStatus_t cublasLtFp8RowMajorNTNMeta::matmul(
            void *instance, cudaStream_t stream, double alpha, const void *A,
            const void *B, double beta, void *C, cudaDataType Ctype, int algoIdx) {
        auto impl = static_cast<Impl *>(instance);
        float alphaFP32 = alpha;
        float betaFP32 = beta;
        switch (Ctype) {
            case CUDA_R_32F:
                return cublasLtMatmul(impl->handle, impl->operationDesc, &alphaFP32, B,
                        impl->Bdesc, A, impl->Adesc, &betaFP32, C,
                        impl->fp8fp8fp32.Cdesc, C, impl->fp8fp8fp32.Ddesc,
                        &impl->fp8fp8fp32.heuristicResult[algoIdx].algo,
                        impl->workspace, impl->workspaceSize, stream);
            case CUDA_R_16F:
                return cublasLtMatmul(impl->handle, impl->operationDesc, &alphaFP32, B,
                        impl->Bdesc, A, impl->Adesc, &betaFP32, C,
                        impl->fp8fp8fp16.Cdesc, C, impl->fp8fp8fp16.Ddesc,
                        &impl->fp8fp8fp16.heuristicResult[algoIdx].algo,
                        impl->workspace, impl->workspaceSize, stream);
            default:
                return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }

    cublasStatus_t cublasLtFp8RowMajorNTNMeta::getAlgoNum(
            void *instance, cudaDataType Ctype, int *algoNum) {
        auto impl = static_cast<Impl *>(instance);
        switch (Ctype) {
            case CUDA_R_32F:
                *algoNum = impl->fp8fp8fp32.algoNum;
                return CUBLAS_STATUS_SUCCESS;
            case CUDA_R_16F:
                *algoNum = impl->fp8fp8fp16.algoNum;
                return CUBLAS_STATUS_SUCCESS;
            default:
                return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }

    cublasStatus_t cublasLtFp8RowMajorNTNMeta::destroy(void *instance) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        if (deviceProp.major < 9) {
            return CUBLAS_STATUS_SUCCESS;
        }
        auto impl = static_cast<Impl *>(instance);
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->fp8fp8fp32.Ddesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->fp8fp8fp32.Cdesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->fp8fp8fp16.Ddesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->fp8fp8fp16.Cdesc));
        CHECK_CUBLAS(cublasLtMatmulDescDestroy(impl->operationDesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->Adesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->Bdesc));
        CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(impl->preference));
        CHECK_CUBLAS(cublasLtDestroy(impl->handle));
        return CUBLAS_STATUS_SUCCESS;
    }

    namespace cublasLtFp8ColMajorTNNMeta {
        namespace {
            struct Impl {
                constexpr static int searchAlgoNum = 1;
                const uint64_t m, n, k;
                const int64_t lda, ldb, ldc;
                const std::size_t workspaceSize;
                void *workspace;
                cublasLtHandle_t handle{nullptr};
                const cublasOperation_t transa{CUBLAS_OP_T}, transb{CUBLAS_OP_N};
                cublasLtMatmulPreference_t preference{nullptr};
                cublasLtMatrixLayout_t Adesc{nullptr}, Bdesc{nullptr};
                cublasLtMatmulDesc_t operationDesc{nullptr};
                struct {
                    cublasLtMatrixLayout_t Cdesc{nullptr}, Ddesc{nullptr};
                    cublasLtMatmulHeuristicResult_t heuristicResult[searchAlgoNum];
                    int algoNum = searchAlgoNum;
                } fp8fp8fp16{}, fp8fp8fp32{};
            };
        }  // namespace

        cublasStatus_t CUBLASWINAPI
            create(void **instance, uint64_t m, uint64_t n, uint64_t k, int64_t lda,
                    int64_t ldb, int64_t ldc, std::size_t workspaceSize /* 32MB for H100 is a good choice */,
                    void *workspace);

        cublasStatus_t CUBLASWINAPI destroy(void *instance);

        static cublasStatus_t CUBLASWINAPI
            matmul(void *instance, cudaStream_t stream, double alpha, const void *A,
                    const void *B, double beta, void *C, cudaDataType Ctype, int algoIdx = 0);

        static cublasStatus_t CUBLASWINAPI
            getAlgoNum(void *instance, cudaDataType Ctype, int *algoNum);
    }  // namespace cublasLtFp8RowMajorNTNMeta

    cublasStatus_t cublasLtFp8ColMajorTNNMeta::create(
            void **instance, uint64_t m, uint64_t n, uint64_t k, int64_t lda,
            int64_t ldb, int64_t ldc, std::size_t workspaceSize, void *workspace) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        if (deviceProp.major < 9) {
            return CUBLAS_STATUS_SUCCESS;
        }
        auto impl = new Impl{m, n, k, lda, ldb, ldc, workspaceSize, workspace};
        CHECK_CUBLAS(cublasLtCreate(&impl->handle));
        CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&impl->preference));
        CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(
                    impl->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &workspaceSize, sizeof(workspaceSize)));
        CHECK_CUBLAS(
                cublasLtMatrixLayoutCreate(&impl->Adesc, CUDA_R_8F_E4M3, k, m, lda));
        CHECK_CUBLAS(
                cublasLtMatrixLayoutCreate(&impl->Bdesc, CUDA_R_8F_E4M3, k, n, ldb));
        CHECK_CUBLAS(cublasLtMatmulDescCreate(&impl->operationDesc,
                    CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
                    impl->operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &impl->transa,
                    sizeof(impl->transa)));
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(
                impl->operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &impl->transb,
                sizeof(impl->transb)));

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp16.Cdesc, CUDA_R_16F,
                                                m, n, ldc));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp16.Ddesc, CUDA_R_16F,
                                                m, n, ldc));
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
                impl->handle, impl->operationDesc, impl->Adesc, impl->Bdesc,
                impl->fp8fp8fp16.Cdesc, impl->fp8fp8fp16.Ddesc, impl->preference, impl->fp8fp8fp16.algoNum,
                impl->fp8fp8fp16.heuristicResult, &impl->fp8fp8fp16.algoNum));
        if (impl->fp8fp8fp16.algoNum == 0) {
            CHECK_CUBLAS(CUBLAS_STATUS_NOT_SUPPORTED);
        }

        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp32.Cdesc, CUDA_R_32F,
                                                m, n, ldc));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&impl->fp8fp8fp32.Ddesc, CUDA_R_32F,
                                                m, n, ldc));
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(
                impl->handle, impl->operationDesc, impl->Adesc, impl->Bdesc,
                impl->fp8fp8fp32.Cdesc, impl->fp8fp8fp32.Ddesc, impl->preference, impl->fp8fp8fp32.algoNum,
                impl->fp8fp8fp32.heuristicResult, &impl->fp8fp8fp32.algoNum));
        if (impl->fp8fp8fp32.algoNum == 0) {
            CHECK_CUBLAS(CUBLAS_STATUS_NOT_SUPPORTED);
        }
        *instance = impl;
        return CUBLAS_STATUS_SUCCESS;
    }

    cublasStatus_t cublasLtFp8ColMajorTNNMeta::matmul(
            void *instance, cudaStream_t stream, double alpha, const void *A,
            const void *B, double beta, void *C, cudaDataType Ctype, int algoIdx) {
        auto impl = static_cast<Impl *>(instance);
        float alphaFP32 = alpha;
        float betaFP32 = beta;
        switch (Ctype) {
            case CUDA_R_32F:
                return cublasLtMatmul(impl->handle, impl->operationDesc, &alphaFP32, A,
                                      impl->Adesc, B, impl->Bdesc, &betaFP32, C,
                                      impl->fp8fp8fp32.Cdesc, C, impl->fp8fp8fp32.Ddesc,
                                      &impl->fp8fp8fp32.heuristicResult[algoIdx].algo,
                                      impl->workspace, impl->workspaceSize, stream);
            case CUDA_R_16F:
                return cublasLtMatmul(impl->handle, impl->operationDesc, &alphaFP32, A,
                                      impl->Adesc, B, impl->Bdesc, &betaFP32, C,
                                      impl->fp8fp8fp16.Cdesc, C, impl->fp8fp8fp16.Ddesc,
                                      &impl->fp8fp8fp16.heuristicResult[algoIdx].algo,
                                      impl->workspace, impl->workspaceSize, stream);
            default:
                return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }

    cublasStatus_t cublasLtFp8ColMajorTNNMeta::getAlgoNum(
            void *instance, cudaDataType Ctype, int *algoNum) {
        auto impl = static_cast<Impl *>(instance);
        switch (Ctype) {
            case CUDA_R_32F:
                *algoNum = impl->fp8fp8fp32.algoNum;
                return CUBLAS_STATUS_SUCCESS;
            case CUDA_R_16F:
                *algoNum = impl->fp8fp8fp16.algoNum;
                return CUBLAS_STATUS_SUCCESS;
            default:
                return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    }

    cublasStatus_t cublasLtFp8ColMajorTNNMeta::destroy(void *instance) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        if (deviceProp.major < 9) {
            return CUBLAS_STATUS_SUCCESS;
        }
        auto impl = static_cast<Impl *>(instance);
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->fp8fp8fp32.Ddesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->fp8fp8fp32.Cdesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->fp8fp8fp16.Ddesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->fp8fp8fp16.Cdesc));
        CHECK_CUBLAS(cublasLtMatmulDescDestroy(impl->operationDesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->Adesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(impl->Bdesc));
        CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(impl->preference));
        CHECK_CUBLAS(cublasLtDestroy(impl->handle));
        return CUBLAS_STATUS_SUCCESS;
    }
}  // namespace mixed_kernels

#define CHUNKSIZE 32


/****************************************************************************************************/
__global__ void double2int8_GPU_kernel(int nrows, int ncols,
                                      const double *S, int lds,
                                      int8_t *T, int ldt) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    if (idx >= nrows || idy >= ncols) { return; }

    T[idy*ldt+idx]= (int8_t)S[idy*lds+idx];
    //printf("double2fp8_GPU %d %d : %g\n", idx, idy, S[idy*lds+idx]);
}

extern "C"
void double2int8_GPU(int nrows, int ncols,
                    const double *S, int lds,
                    void *T, int ldt,
                    cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    int8_t *_T = (int8_t *) T;
    double2int8_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, S, lds, _T, ldt);
}
/****************************************************************************************************/

/****************************************************************************************************/
__global__ void double2fp8_GPU_kernel(int nrows, int ncols,
                                      const double *S, int lds,
                                      uint8_t *T, int ldt) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    if (idx >= nrows || idy >= ncols) { return; }

    //auto s = __nv_fp8_e4m3(S[idy * lds + idx]);
    //T[idy * ldt + idx] = *(uint8_t *)&s;
    T[idy*ldt+idx]= (uint8_t)__nv_cvt_float_to_fp8( (float)S[idy*lds+idx], __NV_SATFINITE, __NV_E4M3 );
    //printf("double2fp8_GPU %d %d : %g\n", idx, idy, S[idy*lds+idx]);
}

extern "C"
void double2fp8_GPU(int nrows, int ncols,
                    const double *S, int lds,
                    void *T, int ldt,
                    cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    uint8_t *_T = (uint8_t *) T;
    double2fp8_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, S, lds, _T, ldt);
}
/****************************************************************************************************/

/****************************************************************************************************/
__global__ void float2fp8_GPU_kernel(int nrows, int ncols,
                                     const float *S, int lds,
                                     uint8_t *T, int ldt) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    if (idx >= nrows || idy >= ncols) { return; }

    //auto s = __nv_fp8_e4m3(S[idy * lds + idx]);
    //T[idy * ldt + idx] = *(uint8_t *)&s;
    T[idy*ldt+idx]= (uint8_t)__nv_cvt_float_to_fp8( (float)S[idy*lds+idx], __NV_SATFINITE, __NV_E4M3 );
    //printf("float2fp8_GPU %d %d : %g\n", idx, idy, S[idy*lds+idx]);
}

extern "C"
void float2fp8_GPU(int nrows, int ncols,
                   const float *S, int lds,
                   void *T, int ldt,
                   cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    uint8_t *_T = (uint8_t *) T;
    float2fp8_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, S, lds, _T, ldt);
}
/****************************************************************************************************/

/****************************************************************************************************/
__global__ void half2fp8_GPU_kernel(int nrows, int ncols,
                                    const __half *S, int lds,
                                    uint8_t *T, int ldt) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    if (idx >= nrows || idy >= ncols) { return; }

    T[idy * ldt + idx] = (uint8_t) __nv_cvt_halfraw_to_fp8(S[idy * lds + idx], __NV_SATFINITE, __NV_E4M3);
    //printf("half2fp8_GPU %d %d : %g\n", idx, idy, S[idy*lds+idx]);
}

extern "C"
void half2fp8_GPU(int nrows, int ncols,
                  const __half *S, int lds,
                  void *T, int ldt,
                  cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    __half *_S = (__half *) S;
    uint8_t *_T = (uint8_t *) T;
    half2fp8_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, _S, lds, _T, ldt);
}

/****************************************************************************************************/

inline __device__ float half_to_float(uint16_t h) {
    float f;
#ifndef USE_ROCM
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
#else
    asm volatile("v_cvt_f32_f16 %0, %1;" : "=v"(f) : "v"(h));
#endif
    return f;
}


/****************************************************************************************************/
__global__ void fp82float_GPU_kernel(int nrows, int ncols,
                                     const uint8_t *S, int lds,
                                     float *T, int ldt) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    if (idx >= nrows || idy >= ncols) { return; }

    __half_raw res = __nv_cvt_fp8_to_halfraw(S[idy * lds + idx], __NV_E4M3);
    T[idy * ldt + idx] = __half2float(res.x);
    //printf("float2fp8_GPU %d %d : %g\n", idx, idy, S[idy*lds+idx]);
}

extern "C"
void fp82float_GPU(int nrows, int ncols,
                   const void *S, int lds,
                   float *T, int ldt,
                   cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    uint8_t *_S = (uint8_t *) S;
    fp82float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, _S, lds, T, ldt);
}
/****************************************************************************************************/

/****************************************************************************************************/
__global__ void double2bf_GPU_kernel(int nrows, int ncols,
                                     const double *F, int ldf,
                                     __nv_bfloat16 *H, int ldh) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    if (idx >= nrows || idy >= ncols) { return; }

    H[idy * ldh + idx] = __double2bfloat16(F[idy * ldf + idx]);
    //printf("double2half_GPU %d %d : %g\n", idx, idy, F[idy*nrows+idx]);
}

extern "C"
void double2bf_GPU(int nrows, int ncols,
                   const double *F, int ldf,
                   void *_H, int ldh,
                   cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    __nv_bfloat16 *H = (__nv_bfloat16 *) _H;
    double2bf_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}
/****************************************************************************************************/
/****************************************************************************************************/
__global__ void float2bf_GPU_kernel(int nrows, int ncols,
                                    const float *F, int ldf,
                                    __nv_bfloat16 *H, int ldh) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    if (idx >= nrows || idy >= ncols) { return; }

    H[idy * ldh + idx] = __float2bfloat16(F[idy * ldf + idx]);
    //printf("double2half_GPU %d %d : %g\n", idx, idy, F[idy*nrows+idx]);
}

extern "C"
void float2bf_GPU(int nrows, int ncols,
                  const float *F, int ldf,
                  void *_H, int ldh,
                  cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    __nv_bfloat16 *H = (__nv_bfloat16 *) _H;
    float2bf_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}
/****************************************************************************************************/

__global__ void bf2float_GPU_kernel(int nrows, int ncols,
                                    const __nv_bfloat16 *H, int ldh,
                                    float *F, int ldf) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    if (idx >= nrows || idy >= ncols) { return; }

    F[idy * ldf + idx] = __bfloat162float(H[idy * ldh + idx]);
    //printf("half2float_GPU %d %d : %g\n", idx, idy, F[idy*nrows+idx]);
}

extern "C"
void bf2float_GPU(int nrows, int ncols,
                  const void *_H, int ldh,
                  float *F, int ldf,
                  cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    __nv_bfloat16 *H = (__nv_bfloat16 *) _H;
    bf2float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, H, ldh, F, ldf);
}
/****************************************************************************************************/
/****************************************************************************************************/
__global__ void double2half_GPU_kernel(int nrows, int ncols,
                                       const double *F, int ldf,
                                       __half *H, int ldh) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    if (idx >= nrows || idy >= ncols) { return; }

    H[idy * ldh + idx] = __double2half(F[idy * ldf + idx]);
    //printf("double2half_GPU %d %d : %g\n", idx, idy, F[idy*nrows+idx]);
}

extern "C"
void double2half_GPU(int nrows, int ncols,
                     const double *F, int ldf,
                     void *_H, int ldh,
                     cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    __half *H = (__half *) _H;
    double2half_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}
/****************************************************************************************************/
/****************************************************************************************************/
__global__ void float2half_GPU_kernel(int nrows, int ncols,
                                      const float *F, int ldf,
                                      __half *H, int ldh) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    if (idx >= nrows || idy >= ncols) { return; }

    H[idy * ldh + idx] = __float2half_rn(F[idy * ldf + idx]);
    //printf("float2half_GPU %d %d : %g\n", idx, idy, F[idy*nrows+idx]);
}

extern "C"
void float2half_GPU(int nrows, int ncols,
                    const float *F, int ldf,
                    void *_H, int ldh,
                    cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    __half *H = (__half *) _H;
    float2half_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, F, ldf, H, ldh);
}
/****************************************************************************************************/
/****************************************************************************************************/

__global__ void half2float_GPU_kernel(int nrows, int ncols,
                                      const __half *H, int ldh,
                                      float *F, int ldf) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + tx;
    const int idy = blockIdx.y * blockDim.y + ty;

    if (idx >= nrows || idy >= ncols) { return; }

    F[idy * ldf + idx] = __half2float(H[idy * ldh + idx]);
    //printf("half2float_GPU %d %d : %g\n", idx, idy, F[idy*nrows+idx]);
}

extern "C"
void half2float_GPU(int nrows, int ncols,
                    const void *_H, int ldh,
                    float *F, int ldf,
                    cudaStream_t stream) {
    int nBlockx = (nrows + CHUNKSIZE - 1) / CHUNKSIZE;
    int nBlocky = (ncols + CHUNKSIZE - 1) / CHUNKSIZE;
    dim3 dimBlock(CHUNKSIZE, CHUNKSIZE);
    dim3 dimGrid(nBlockx, nBlocky);
    __half *H = (__half *) _H;
    half2float_GPU_kernel<<<dimGrid, dimBlock, 0, stream>>>(nrows, ncols, H, ldh, F, ldf);
}
/****************************************************************************************************/



// Function to calculate Frobenius norm
static double frobenius_norm_diff(void *A, double *A_ref, int M, int N, int gemm_type) {
    double norm = 0.0;
    switch (gemm_type) {
        case 1: {
            double *_A = (double *) A;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    double diff = _A[i * N + j] - A_ref[i * N + j];
                    norm += diff * diff;
                }
            }
        }
            break;

        case 9: {
            int32_t *_A = (int32_t *) A;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    double diff = _A[i * N + j] - A_ref[i * N + j];
                    norm += diff * diff;
                }
            }
        }
            break;

        default: {
            float *_A = (float *) A;
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

// Function to calculate Frobenius norm
static double frobenius_norm(void *A, int M, int N, int gemm_type) {
    double norm = 0.0;
    switch (gemm_type) {
        case 1: {
            double *_A = (double *) A;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    double diff = _A[i * N + j];
                    norm += diff * diff;
                }
            }
            break;
        }

        case 9: {
            int32_t *_A = (int32_t *) A;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    //printf("%d %d %d\n", i, j, _A[i * N + j]);
                    double diff = _A[i * N + j];
                    norm += diff * diff;
                }
            }
            break;
        }

        default: {
            float *_A = (float *) A;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    double diff = _A[i * N + j];
                    norm += diff * diff;
                }
            }
            break;
        }

    }

    return sqrt(norm);
}


// Print matrix
static void print_matrix(void *A, int M, int N, int gemm_type) {
    switch (gemm_type) {
        case 1: {
            double *_A = (double *) A;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    printf("%lf ", _A[i * N + j]);
                }
                printf("\n");
            }
        }
            break;

        default: {
            float *_A = (float *) A;
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

// Function to parse integer arguments
static int parse_arg(const char *arg) {
    int value;
    if (sscanf(arg, "%d", &value) != 1) {
        fprintf(stderr, "Invalid argument: %s\n", arg);
        exit(EXIT_FAILURE);
    }
    return value;
}

static void init_matrix_host(void **A, void **A_ref, int M, int N, int gemm_type, int seed) {
    cudaMallocHost((void **) A_ref, M * N * sizeof(double));
    double *_A_ref = (double *) (*A_ref);
    srand(seed);

    switch (gemm_type) {
        case 1: {
            cudaMallocHost((void **) A, M * N * sizeof(double));
            double *_A = (double *) (*A);
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    //_A[i*N+j] = (double)((i+j)/(1.0*M*N));
                    _A[i * N + j] = (double) rand() / RAND_MAX;
                    _A_ref[i * N + j] = (double) _A[i * N + j];
                }
            }
        }
            break;

        case 9: {
            cudaMallocHost((void **) A, M * N * sizeof(int32_t));
            int32_t *_A = (int32_t*) (*A);
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    _A[i * N + j] = 1; 
                    _A_ref[i * N + j] = (double) _A[i * N + j];
                }
            }
        }
            break;

        default: {
            cudaMallocHost((void **) A, M * N * sizeof(float));
            float *_A = (float *) (*A);
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    //_A[i*N+j] = (float)((i+j)/(1.0*M*N));
                    _A[i * N + j] = (float) rand() / RAND_MAX;
                    _A_ref[i * N + j] = (double) _A[i * N + j];
                }
            }
        }
            break;

    }
}


static void allocate_memory_device(void **A, void **A_ref, int M, int N, int gemm_type) {
    cudaMalloc(A_ref, M * N * sizeof(double));
    switch (gemm_type) {
        case 1:
            cudaMalloc(A, M * N * sizeof(double));
            break;

        case 9:
            cudaMalloc(A, M * N * sizeof(int32_t));
            break;

        default:
            cudaMalloc(A, M * N * sizeof(float));
            break;
    }
}

static void my_cuda_copy(void *d_A, void *h_A, int M, int N,
                         enum cudaMemcpyKind flag, int gemm_type, cudaStream_t stream) {
    switch (gemm_type) {
        case 1:
            cudaMemcpyAsync(d_A, h_A, M * N * sizeof(double), flag, stream);
            break;

        case 9:
            cudaMemcpyAsync(d_A, h_A, M * N * sizeof(int32_t), flag, stream);
            break;

        default:
            cudaMemcpyAsync(d_A, h_A, M * N * sizeof(float), flag, stream);
            break;
    }
}

static void
my_gemm(void *cublasLtInstance, cudaStream_t stream, cublasHandle_t handle, int M, int N, int K, void *d_A, void *d_B,
        void *d_C, int gemm_type) {
    switch (gemm_type) {
        case 1: {
                    double alpha = 1.0;
                    double beta = 0.0;
                    cublasGemmEx(handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            M, N, K,
                            &alpha,
                            d_A, CUDA_R_64F, K,
                            d_B, CUDA_R_64F, K,
                            &beta,
                            d_C, CUDA_R_64F, M,
                            CUDA_R_64F,
                            CUBLAS_GEMM_DEFAULT);
                }
                break;

        case 2: {
                    float alpha = 1.0f;
                    float beta = 0.0f;
                    cublasGemmEx(handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            M, N, K,
                            &alpha,
                            d_A, CUDA_R_32F, K,
                            d_B, CUDA_R_32F, K,
                            &beta,
                            d_C, CUDA_R_32F, M,
                            CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT);
                }
                break;

        case 3: {
                    float alpha = 1.0f;
                    float beta = 0.0f;
                    cublasGemmEx(handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            M, N, K,
                            &alpha,
                            d_A, CUDA_R_32F, K,
                            d_B, CUDA_R_32F, K,
                            &beta,
                            d_C, CUDA_R_32F, M,
                            CUBLAS_COMPUTE_32F_FAST_TF32,
                            CUBLAS_GEMM_DEFAULT);
                }
                break;

        case 4: {
                    float alpha = 1.0f;
                    float beta = 0.0f;
                    cublasGemmEx(handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            M, N, K,
                            &alpha,
                            d_A, CUDA_R_16F, K,
                            d_B, CUDA_R_16F, K,
                            //d_A, CUDA_R_32F, M,
                            //d_B, CUDA_R_32F, K,
                            &beta,
                            d_C, CUDA_R_32F, M,
                            CUDA_R_32F,
                            //CUBLAS_COMPUTE_32F_FAST_16F,
                            //On V100, the performace goes down when matrix size bigger than 4096 if using CUBLAS_COMPUTE_32F_FAST_16F
                            CUBLAS_GEMM_DEFAULT);
                }
                break;

        case 5: {
                    /* Here it need to be defined as __half !!! */
                    __half alpha = (__half) 1.0f;
                    __half beta = (__half) 0.0f;
                    cublasStatus_t status = cublasGemmEx(handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            M, N, K,
                            &alpha,
                            d_A, CUDA_R_16F, K,
                            d_B, CUDA_R_16F, K,
                            &beta,
                            d_C, CUDA_R_16F, M,
                            CUDA_R_16F,
                            CUBLAS_GEMM_DEFAULT);
                    if (CUBLAS_STATUS_SUCCESS != status) {
                        printf("ERROR\n");
                    }
                }
                break;

        case 6: {
                    float alpha = 1.0f;
                    float beta = 0.0f;
                    cublasGemmEx(handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            M, N, K,
                            &alpha,
                            d_A, CUDA_R_32F, K,
                            d_B, CUDA_R_32F, K,
                            &beta,
                            d_C, CUDA_R_32F, M,
                            CUBLAS_COMPUTE_32F_FAST_16BF,
                            CUBLAS_GEMM_DEFAULT);
                }
                break;

        case 7: {
                    /* Here it need to be defined as float !!! */
                    float alpha = 1.0f;
            float beta = 0.0f;
            cublasGemmEx(handle,
                         CUBLAS_OP_T, CUBLAS_OP_N,
                         M, N, K,
                         &alpha,
                         d_A, CUDA_R_16BF, K,
                         d_B, CUDA_R_16BF, K,
                         &beta,
                         d_C, CUDA_R_16BF, M,
                         CUDA_R_32F,
                         CUBLAS_GEMM_DEFAULT);
        }
            break;

        case 8: {
            // TODO
            /* Here it need to be defined as float !!! */
            float alpha = 1.0f;
            float beta = 0.0f;
            CHECK_CUBLAS(
                    mixed_kernels::cublasLtFp8ColMajorTNNMeta::matmul(cublasLtInstance, stream, alpha, d_A, d_B, beta,
                                                                      d_C,
                                                                      CUDA_R_32F, 0));
#if 0
            __half alpha = (__half)1.0f;
                __half beta = (__half)0.0f;
                cublasLtMatmul(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        M, N, K,
                        &alpha,
                        d_A, CUDA_R_16BF, M,
                        d_B, CUDA_R_16BF, K,
                        &beta,
                        d_C, CUDA_R_16BF, M,
                        CUDA_R_32F,
                        CUBLAS_GEMM_DEFAULT);
#endif
        }
            break;

        case 9: {
            /* Here it need to be defined as float !!! */
            int32_t alpha = 1;
            int32_t beta = 0;
            cublasGemmEx(handle,
                         CUBLAS_OP_T, CUBLAS_OP_N,
                         M, N, K,
                         &alpha,
                         d_A, CUDA_R_8I, K,
                         d_B, CUDA_R_8I, K,
                         &beta,
                         d_C, CUDA_R_32I, M,
                         CUDA_R_32I,
                         CUBLAS_GEMM_DEFAULT);
        }
            break;

    }
}

#define NTYPES 9

/* gemm_type
 * 1: FP64
 * 2: FP32
 * 3: TP32
 * 4: FP16: A16B16C32OP32
 * 5: FP16: A16B16C16OP16
 * 6: BF16: A16B16C32OP32
 * 7: BF16: A16B16C16OP16
 * 8: FP8: A8B8C32OP32
 * 9: INT8: A8B8C32OP32
 */
int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 7) {
        fprintf(stderr,
                "Usage: %s M/N/K gemm_type nb_runs seed [time_with_copy, default: 0] [time_with_conversion, default 0]\n",
                argv[0]);
        exit(EXIT_FAILURE);
    }

    const char *name[NTYPES] = {
        "FP64",
        "FP32",
        "TP32",
        "FP16:A16B16C32OP32",
        "FP16:A16B16C16OP16",
        "BF16:A16B16C32OP32",
        "BF16:A16B16C16OP16",
        "FP8:A8B8C32OP32",
        "INT8:A8B8C32OP32"
    };

    int M = parse_arg(argv[1]);
    int N = M;
    int K = M;
    int gemm_type = parse_arg(argv[2]);
    int nb_runs = 3;
    if (argc >= 4) {
        nb_runs = parse_arg(argv[3]);
    }
    int seed = 0;
    if (argc >= 5) {
        seed = parse_arg(argv[4]);
    }
    int time_with_copy = 0;
    if (argc >= 6) {
        time_with_copy = parse_arg(argv[5]);
    }
    int time_with_conversion = 0;
    if (argc >= 7) {
        time_with_conversion = parse_arg(argv[6]);
    }

    if (0 == time_with_conversion && 1 == time_with_copy) {
        printf("No support! \n");
        return 0;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cublasSetStream(handle, stream);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    void *cublasLtInstance;
    const size_t workspaceSize = 32 * 1024 * 1024;
    void *workspace;
    cudaMalloc(&workspace, workspaceSize);
    mixed_kernels::cublasLtFp8ColMajorTNNMeta::create(&cublasLtInstance, M, N, K, K, K, M, workspaceSize, workspace);

    void *h_A, *h_B, *h_C, *h_A_ref, *h_B_ref, *h_C_ref;
    void *d_A, *d_B, *d_C, *d_C0, *d_A_ref, *d_B_ref, *d_C_ref;

    // Initialize host matrices
    init_matrix_host(&h_A, &h_A_ref, M, K, gemm_type, seed);
    init_matrix_host(&h_B, &h_B_ref, K, N, gemm_type, seed);
    init_matrix_host(&h_C, &h_C_ref, M, N, gemm_type, seed);

    if(0) {
        printf("org\n");
        print_matrix( h_A, M, N, gemm_type );
        printf("ref\n");
        print_matrix( h_A_ref, M, N, 1 );
    }

    // Allocate device memory
    allocate_memory_device((void **) &d_A, (void **) &d_A_ref, M, K, gemm_type);
    allocate_memory_device((void **) &d_B, (void **) &d_B_ref, K, N, gemm_type);
    allocate_memory_device((void **) &d_C, (void **) &d_C_ref, M, N, gemm_type);
    cudaMalloc((void **) &d_C0, M * N * sizeof(float));

    my_cuda_copy(d_C, h_C, M, N, cudaMemcpyHostToDevice, 1, stream);

    // Copy host matrices to device
    my_cuda_copy(d_A_ref, h_A_ref, M, K, cudaMemcpyHostToDevice, 1, stream);
    my_cuda_copy(d_B_ref, h_B_ref, K, N, cudaMemcpyHostToDevice, 1, stream);
    if (!time_with_copy) {
        my_cuda_copy(d_A, h_A, M, K, cudaMemcpyHostToDevice, gemm_type, stream);
        my_cuda_copy(d_B, h_B, K, N, cudaMemcpyHostToDevice, gemm_type, stream);
    }

    if (4 == gemm_type || 5 == gemm_type) {
        if (!time_with_conversion) {
            double2half_GPU(M, K, (double *) d_A_ref, M, d_A, M, stream);
            double2half_GPU(K, M, (double *) d_B_ref, K, d_B, K, stream);
        }
    } else if (6 == gemm_type || 7 == gemm_type) {
        if (!time_with_conversion) {
            double2bf_GPU(M, K, (double *) d_A_ref, M, d_A, M, stream);
            double2bf_GPU(K, M, (double *) d_B_ref, K, d_B, K, stream);
        }
    } else if (8 == gemm_type) {
        if (!time_with_conversion) {
            double2fp8_GPU(M, K, (double *) d_A_ref, M, d_A, M, stream);
            double2fp8_GPU(K, M, (double *) d_B_ref, K, d_B, K, stream);
        }
    } else if (9 == gemm_type) {
        if (!time_with_conversion) {
            double2int8_GPU(M, K, (double *) d_A_ref, M, d_A, M, stream);
            double2int8_GPU(K, M, (double *) d_B_ref, K, d_B, K, stream);
        }
    }

    // Measure GEMM performance
    double start_time, end_time;
    struct timeval tstart;
    struct timeval tend;

    for (int i = 0; i < nb_runs; ++i) {
        cudaDeviceSynchronize();
        gettimeofday(&tstart, NULL);
        start_time = tstart.tv_sec + tstart.tv_usec / 1.0e6;

        // Copy host matrices to device
        if (time_with_copy) {
            my_cuda_copy(d_A, h_A, M, K, cudaMemcpyHostToDevice, gemm_type, stream);
            my_cuda_copy(d_B, h_B, K, N, cudaMemcpyHostToDevice, gemm_type, stream);
        }

        cudaStreamSynchronize(stream);

        switch(gemm_type) {
            case 1: case 2: case 3:
                my_gemm(cublasLtInstance, stream, handle, M, N, K, d_A, d_B, d_C, gemm_type);
                break;

            case 4: 
                if (time_with_conversion) {
                    double2half_GPU(M, K, (double *) d_A_ref, M, d_A, M, stream);
                    double2half_GPU(K, M, (double *) d_B_ref, K, d_B, K, stream);
                }
                my_gemm(cublasLtInstance, stream, handle, M, N, K, d_A, d_B, d_C, gemm_type);
                if (time_with_conversion) {
                    half2float_GPU(M, N, d_C, M, (float *) d_C0, M, stream);
                }
                break;

            case 5: 
                if (time_with_conversion) {
                    double2half_GPU(M, K, (double *) d_A_ref, M, d_A, M, stream);
                    double2half_GPU(K, M, (double *) d_B_ref, K, d_B, K, stream);
                }
                my_gemm(cublasLtInstance, stream, handle, M, N, K, d_A, d_B, d_C0, gemm_type);
                if (time_with_conversion) {
                    half2float_GPU(M, N, d_C0, M, (float *) d_C, M, stream);
                }
                break;

            case 6: 
                if (time_with_conversion) {
                    double2bf_GPU(M, K, (double *) d_A_ref, M, d_A, M, stream);
                    double2bf_GPU(K, M, (double *) d_B_ref, K, d_B, K, stream);
                }
                my_gemm(cublasLtInstance, stream, handle, M, N, K, d_A, d_B, d_C, gemm_type);
                if (time_with_conversion) {
                    bf2float_GPU(M, N, d_C, M, (float *) d_C0, M, stream);
                }
                break;

            case 7: 
                if (time_with_conversion) {
                    double2bf_GPU(M, K, (double *) d_A_ref, M, d_A, M, stream);
                    double2bf_GPU(K, M, (double *) d_B_ref, K, d_B, K, stream);
                }
                my_gemm(cublasLtInstance, stream, handle, M, N, K, d_A, d_B, d_C0, gemm_type);
                if (time_with_conversion) {
                    bf2float_GPU(M, N, d_C0, M, (float *) d_C, M, stream);
                }
                break;

            case 8: 
                if (time_with_conversion) {
                    double2fp8_GPU(M, K, (double *) d_A_ref, M, d_A, M, stream);
                    double2fp8_GPU(K, M, (double *) d_B_ref, K, d_B, K, stream);
                }
                my_gemm(cublasLtInstance, stream, handle, M, N, K, d_A, d_B, d_C, gemm_type);
                break;

            case 9:
                if (time_with_conversion) {
                    double2int8_GPU(M, K, (double *) d_A_ref, M, d_A, M, stream);
                    double2int8_GPU(K, M, (double *) d_B_ref, K, d_B, K, stream);
                }
                my_gemm(cublasLtInstance, stream, handle, M, N, K, d_A, d_B, d_C, gemm_type);
                break;

            default:
                fprintf(stderr, "Error: wrong inputs!\n");
        }

        // Copy result back to host
        if (time_with_copy) {
            my_cuda_copy(h_C, d_C, M, N, cudaMemcpyDeviceToHost, gemm_type, stream);
        }

        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();
        gettimeofday(&tend, NULL);
        end_time = tend.tv_sec + tend.tv_usec / 1.0e6;

        if (5 == gemm_type) {
            if (!time_with_conversion) {
                half2float_GPU(M, N, d_C0, M, (float *) d_C, M, stream);
            }
        } else if (7 == gemm_type) {
            if (!time_with_conversion) {
                bf2float_GPU(M, N, d_C0, M, (float *) d_C, M, stream);
            }
        } else if (8 == gemm_type) {
            // C is in FP32 here
        }

        // Copy result back to host
        if (!time_with_copy) {
            my_cuda_copy(h_C, d_C, M, N, cudaMemcpyDeviceToHost, gemm_type, stream);
        }

        // Calculate reference result (e.g., using CPU-based GEMM implementation)
        my_gemm(cublasLtInstance, stream, handle, M, N, K, d_A_ref, d_B_ref, d_C_ref, 1);
        my_cuda_copy(h_C_ref, d_C_ref, M, N, cudaMemcpyDeviceToHost, 1, stream);
        cudaStreamSynchronize(stream);
        cudaDeviceSynchronize();

        // Calculate Frobenius norm
        double norm_diff = frobenius_norm_diff(h_C, (double *) h_C_ref, M, N, gemm_type);
        double norm_C = frobenius_norm(h_C, M, N, gemm_type);
        double norm_C_ref = frobenius_norm(h_C_ref, M, N, 1);

        double gflops = 2.0 * M * N * K / (1e12);
        if (i > 0 || 1 == nb_runs) {
            printf("GEMM: %s %d %d %d %d %d %d %d : %lf TFLOPS : %.16lf %.16lf %g \n",
                   name[gemm_type-1], M, N, K, gemm_type, seed, time_with_copy, time_with_conversion, gflops / (end_time - start_time),
                   norm_C, norm_C_ref, norm_diff / norm_C_ref);
        }

        //printf("Frobenius_norm: C %g C_ref %g ||C-C_ref||/||C_ref||%g\n", norm_C, norm_C_ref, norm_diff);
    }

#if 0
    printf("org\n");
    print_matrix( h_C, M, N, gemm_type );
    printf("ref\n");
    print_matrix( h_C_ref, M, N, 1 );
#endif

    // Clean up
    cudaFree(workspace);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFreeHost(h_A_ref);
    cudaFreeHost(h_B_ref);
    cudaFreeHost(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C0);
    cudaFree(d_A_ref);
    cudaFree(d_B_ref);
    cudaFree(d_C_ref);
    cublasDestroy(handle);

    return 0;
}
