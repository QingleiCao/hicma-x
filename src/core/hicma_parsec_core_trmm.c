/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"


void hicma_parsec_trmm_core_gemm_lln_cpu(
        hicma_parsec_params_t *params_tlr,
        parsec_execution_stream_t *es,
        dplasma_enum_t transA,
        dplasma_enum_t transB,
        int mb, int nb, int kb,
        double lalpha, void *A, int lda,
        void *B, int ldb,
        double lbeta, void *C, int ldc,
        int m, int n, int k)
{
    CORE_dgemm(transA, transB, mb, nb, kb,
               lalpha, A /* descA((descB->mt-1)-m,k) */, lda,
                       B /* descB(k,n) */,               ldb,
               1.0,    C /* descB((descB->mt-1)-m,n) */, ldc );
}


#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT) || defined(PARSEC_HAVE_DEV_HIP_SUPPORT)

void hicma_parsec_trmm_core_gemm_lln_gpu(
        hicma_parsec_params_t *params_tlr,
        parsec_potrf_workspace_t *ws_gpu,
        parsec_device_cuda_module_t *cuda_device,
        parsec_gpu_task_t *gpu_task,
        parsec_cuda_exec_stream_t *cuda_stream,
        dplasma_enum_t transA,
        dplasma_enum_t transB,
        int mb, int nb, int kb, 
        double lalpha, void *A, int lda,
        void *B, int ldb,
        double lbeta, void *C, int ldc,  
        int m, int n, int k)
{

    void *A_use = A;
    void *A_d, *A_s, *A_h;
    cublasStatus_t status;

    /* Find workspace */
    parsec_potrf_workspace_t *_ws_gpu = (parsec_potrf_workspace_t *)ws_gpu;
    parsec_potrf_stream_workspace_t *stream_found = lookup_gpu_workspace(cuda_device, cuda_stream, _ws_gpu);

    /* Get handle_cublas */
    cublasHandle_t handle = stream_found->handle_cublas;
    //cublasSetStream( handle, cuda_stream->cuda_stream );

    /* Get the temporary buffer on GPU */
    A_d = (double *)stream_found->gpu_buffer_A;
    A_s = (float *)stream_found->gpu_buffer_A;
    A_h = (void *)stream_found->gpu_buffer_A;

    status = cublasDgemm( handle, dplasma_cublas_op(transA), dplasma_cublas_op(transB), //CUBLAS_OP_N,
                 mb, nb, kb,
                 &lalpha, (double*)A, lda,
                         (double*)B, ldb,
                 &lbeta,  (double*)C, ldc );
    PARSEC_CUDA_CHECK_ERROR( "cublasDgemm ", status, {exit(PARSEC_HOOK_RETURN_ERROR);} );
}

#endif
