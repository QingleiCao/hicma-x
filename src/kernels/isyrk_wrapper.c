/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/
#include "hicma_parsec.h"
#include "isyrk_LT.h"
//TODO upper
//#include "isyrk_UN.h"
//#include "isyrk_UT.h"

/* Debug flag for additional output */
static int print_more = 0;
/**
 *******************************************************************************
 *
 * @ingroup dplasma_double
 *
 *  dplasma_isyrk_New - Generates the taskpool that performs mixed precision SYRK
 *
 *    \f[ C = \alpha [ op( A ) \times op( A )' ] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X'
 *
 *  This function supports mixed precision computation with tensor core acceleration:
 *  - Double precision (inttype = 0): Standard double precision SYRK
 *  - 8-bit integer precision (inttype = 8): Tensor core accelerated SYRK
 *
 *  where alpha and beta are real scalars, C is an n-by-n symmetric
 *  matrix and A is an n-by-k matrix in the first case and a k-by-n
 *  matrix in the second case.
 *
 *  WARNING: The computations are not done by this call.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          = dplasmaUpper: Upper triangle of C is stored;
 *          = dplasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is transposed or transposed:
 *          = dplasmaNoTrans: A is not transposed;
 *          = dplasmaTrans:   A is transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha.
 *
 * @param[in] A
 *          A is a LDA-by-ka matrix, where ka is K when trans = dplasmaNoTrans,
 *          and is N otherwise.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          C is a LDC-by-N matrix.
 *          On exit, the array uplo part of the matrix is overwritten
 *          by the uplo part of the updated matrix.
 *
 * @param[in] inttype
 *          Integer type for mixed precision: 0 for double precision, 8 for 8-bit integer
 *
 * @param[in] params_tlr
 *          HiCMA parameters including tensor core settings
 *
 * @param[in] data
 *          HiCMA data structure containing GPU workspace
 *
 *******************************************************************************
 *
 * @return
 *          \retval NULL if incorrect parameters are given.
 *          \retval The parsec taskpool describing the operation that can be
 *          enqueued in the runtime with parsec_context_add_taskpool(). It, then, needs to be
 *          destroy with dplasma_isyrk_Destruct();
 *
 *
 *****************************************************************************/

/**
 * Hook function for GPU evaluation of SYRK tasks
 * @param task The parsec task to evaluate
 * @return PARSEC_HOOK_RETURN_DONE indicating task should be executed
 */
static parsec_hook_return_t evaluate_gpu_syrk(parsec_task_t* task) {
        return PARSEC_HOOK_RETURN_DONE;
}

/**
 * Hook function for GPU evaluation of GEMM tasks
 * @param task The parsec task to evaluate
 * @return PARSEC_HOOK_RETURN_DONE indicating task should be executed
 */
static parsec_hook_return_t evaluate_gpu_gemm(parsec_task_t* task) {
        return PARSEC_HOOK_RETURN_DONE;
}



parsec_taskpool_t*
dplasma_isyrk_New( dplasma_enum_t uplo,
                   dplasma_enum_t trans,
                   double alpha,
                   parsec_tiled_matrix_t* A,
                   double beta,
                   parsec_tiled_matrix_t* C, int inttype, 
                   hicma_parsec_params_t * params_tlr, 
                   hicma_parsec_data_t *data)
{
    parsec_isyrk_LT_taskpool_t* tp;
    void** eval_gpu_syrk;
    void** eval_gpu_gemm;

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    /** Find all available CUDA devices for GPU computation */
    int nb = 0;
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( PARSEC_DEV_CUDA == device->type ) {
            nb++;
        }
    }

    /* Handle case when no CUDA devices are available */
    if(nb == 0) {
        char hostname[256];
        gethostname(hostname, 256);
            //TODO
        /*if( print_more ) {
            fprintf(stderr, "No CUDA device found on rank %d on %s\n",
                    parsec->my_rank, hostname);
        }*/

        /* No half precision on CPU */
        //assert( band_size_dense >= A->lmt );
    }

    /* Create array of CUDA device indices */
    int *dev_index = (int*)malloc(nb * sizeof(int));
    nb = 0;
    for(int i = 0; i < (int)parsec_nb_devices; i++) {
        parsec_device_module_t *device = parsec_mca_device_get(i);
        if( PARSEC_DEV_CUDA == device->type ) {
            dev_index[nb++] = device->device_index;
        }
    }
#endif

    /* Create the mixed precision SYRK taskpool for lower triangular matrices */
    tp = parsec_isyrk_LT_new(uplo, trans,
                                   alpha, A,
                                   beta,  C, inttype, params_tlr);

    /* Find the correct task class IDs for SYRK and GEMM operations */
    int syrk_id, gemm_id, gpu_id = 0;
    for( int i = 0; i < tp->super.nb_task_classes; i++ ) {
	    if( !strcmp(tp->super.task_classes_array[i]->name, "dsyrk") )
		    syrk_id = tp->super.task_classes_array[i]->task_class_id;
	    else if( !strcmp(tp->super.task_classes_array[i]->name, "dgemm") )
		    gemm_id = tp->super.task_classes_array[i]->task_class_id;
    }

#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    /* Debug output for task class IDs */
    if( 0 == params_tlr->rank ) printf("syrk_id= %d gemm_id= %d gpu_id= %d\n", syrk_id, gemm_id, gpu_id);

        /* Set up GPU evaluation hooks for SYRK and GEMM tasks */
        eval_gpu_syrk  = (void *)&tp->super.task_classes_array[syrk_id]->incarnations[gpu_id].evaluate;
        eval_gpu_gemm  = (void *)&tp->super.task_classes_array[gemm_id]->incarnations[gpu_id].evaluate;
        *eval_gpu_syrk  = &evaluate_gpu_syrk;
        *eval_gpu_gemm  = &evaluate_gpu_gemm;

        /* Set GPU workspace and device information */
            tp->_g_ws_gpu = (void *)data->ws_gpu;
            tp->_g_nb_cuda_devices = nb;
            tp->_g_cuda_device_index = dev_index;
#endif
        /* Set up arena for 32-bit integer data type (output of mixed precision operations) */
        parsec_add2arena( &((parsec_isyrk_LT_taskpool_t*)tp)->arenas_datatypes[PARSEC_isyrk_LT_DEFAULT_ADT_IDX],
                            parsec_datatype_int32_t, PARSEC_MATRIX_FULL,
                             1, C->mb, C->nb, C->mb,
                            PARSEC_ARENA_ALIGNMENT_SSE, -1);

        /* Set up arena for 8-bit integer data type (input for tensor core operations) */
        parsec_add2arena( &((parsec_isyrk_LT_taskpool_t*)tp)->arenas_datatypes[PARSEC_isyrk_LT_FULL_I8_ADT_IDX],
                             MPI_BYTE, PARSEC_MATRIX_FULL,
                             1, A->mb, A->nb, A->mb,
                            PARSEC_ARENA_ALIGNMENT_SSE, -1);

    return (parsec_taskpool_t*)tp;
}
/**
 *******************************************************************************
 *
 * @ingroup dplasma_double
 *
 *  dplasma_isyrk_Destruct - Free the data structure associated to a mixed precision SYRK taskpool
 *  created with dplasma_isyrk_New().
 *
 *******************************************************************************
 *
 * @param[in,out] tp
 *          On entry, the taskpool to destroy.
 *          On exit, the taskpool cannot be used anymore.
 *
 * @param[in] uplo
 *          Upper or lower triangular storage (unused, kept for compatibility)
 *
 * @param[in] trans
 *          Transpose operation (unused, kept for compatibility)
 *
 ******************************************************************************/
void
dplasma_isyrk_Destruct( parsec_taskpool_t *tp, dplasma_enum_t uplo, dplasma_enum_t trans)
{

       parsec_isyrk_LT_taskpool_t *isyrk_tp_LT = (parsec_isyrk_LT_taskpool_t*)tp;
#if defined(PARSEC_HAVE_DEV_CUDA_SUPPORT)
    /* Free CUDA device index array if it was allocated */
    if( isyrk_tp_LT->_g_nb_cuda_devices > 0 ) {

        if( NULL != isyrk_tp_LT->_g_cuda_device_index )
            free(isyrk_tp_LT->_g_cuda_device_index);
    }
#endif 
        /* Clean up arena for 32-bit integer data type */
        parsec_del2arena( &isyrk_tp_LT->arenas_datatypes[PARSEC_isyrk_LT_DEFAULT_ADT_IDX] );
        /* Clean up arena for 8-bit integer data type */
        parsec_del2arena( &isyrk_tp_LT->arenas_datatypes[PARSEC_isyrk_LT_FULL_I8_ADT_IDX] );

    /* Free the taskpool */
    parsec_taskpool_free(tp);
}

/**
 *******************************************************************************
 *
 * @ingroup dplasma_double
 *
 *  dplasma_isyrk - Performs mixed precision SYRK operation
 *
 *    \f[ C = \alpha [ op( A ) \times op( A )' ] + \beta C \f],
 *
 *  where op( X ) is one of
 *
 *    op( X ) = X  or op( X ) = X'
 *
 *  This function supports mixed precision computation with tensor core acceleration:
 *  - Double precision (inttype = 0): Standard double precision SYRK
 *  - 8-bit integer precision (inttype = 8): Tensor core accelerated SYRK
 *
 *  where alpha and beta are real scalars, C is an n-by-n symmetric
 *  matrix and A is an n-by-k matrix in the first case and a k-by-n
 *  matrix in the second case.
 *
 *******************************************************************************
 *
 * @param[in,out] parsec
 *          The parsec context of the application that will run the operation.
 *
 * @param[in] uplo
 *          = dplasmaUpper: Upper triangle of C is stored;
 *          = dplasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          Specifies whether the matrix A is transposed or transposed:
 *          = dplasmaNoTrans: A is not transposed;
 *          = dplasmaTrans:   A is transposed.
 *
 * @param[in] alpha
 *          alpha specifies the scalar alpha.
 *
 * @param[in] A
 *          A is a LDA-by-ka matrix, where ka is K when trans = dplasmaNoTrans,
 *          and is N otherwise.
 *
 * @param[in] beta
 *          beta specifies the scalar beta
 *
 * @param[in,out] C
 *          C is a LDC-by-N matrix.
 *          On exit, the array uplo part of the matrix is overwritten
 *          by the uplo part of the updated matrix.
 *
 * @param[in] inttype
 *          Integer type for mixed precision: 0 for double precision, 8 for 8-bit integer
 *
 * @param[in] params_tlr
 *          HiCMA parameters including tensor core settings
 *
 * @param[in] data
 *          HiCMA data structure containing GPU workspace
 *
 *******************************************************************************
 *
 * @return
 *          \retval -i if the ith parameters is incorrect.
 *          \retval 0 on success.
 *
 ******************************************************************************/
int
dplasma_isyrk( parsec_context_t *parsec,
               dplasma_enum_t uplo,
               dplasma_enum_t trans,
               double alpha,
               parsec_tiled_matrix_t *A,
               double beta,
               parsec_tiled_matrix_t *C, int inttype, 
               hicma_parsec_params_t * params_tlr,
               hicma_parsec_data_t *data)
{
    parsec_taskpool_t *parsec_isyrk = NULL;

    /* Validate input arguments for mixed precision SYRK operation */
    if ((uplo != dplasmaLower) && (uplo != dplasmaUpper)) {
        dplasma_error("dplasma_isyrk", "illegal value of uplo");
        return -1;
    }
    if (trans != dplasmaTrans && trans != dplasmaNoTrans ) {
        dplasma_error("dplasma_isyrk", "illegal value of trans");
        return -2;
    }
    /* Matrix C must be square for SYRK operation */
    if ( (C->m != C->n) ) {
        dplasma_error("dplasma_isyrk", "illegal size of matrix C which should be square");
        return -6;
    }
    /* Validate matrix A dimensions based on transpose operation */
    if ( ((trans == dplasmaNoTrans) && (A->m != C->m)) ||
         ((trans != dplasmaNoTrans) && (A->n != C->m)) ) {
        dplasma_error("dplasma_isyrk", "illegal size of matrix A");
        return -4;
    }

    /* Create the mixed precision SYRK taskpool */
    parsec_isyrk = dplasma_isyrk_New(uplo, trans,
                                    alpha, A,
                                    beta, C, inttype, params_tlr, data);

    /* Execute the SYRK operation if taskpool creation was successful */
    if ( parsec_isyrk != NULL )
    {
        /* Add taskpool to parsec context and execute */
        parsec_context_add_taskpool( parsec, parsec_isyrk);
        /* Wait for completion of all tasks */
        dplasma_wait_until_completion(parsec);
        /* Clean up the taskpool */
        dplasma_isyrk_Destruct( parsec_isyrk, uplo, trans);
    }
    return 0;
}
