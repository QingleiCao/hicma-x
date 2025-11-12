/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2023-2025     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec.h"

/**
 * @file hicma_parsec_distribution.c
 * @brief Implementation of symmetric two-dimensional block cyclic band distribution
 *
 * This file implements the distribution system for HICMA PaRSEC that provides
 * efficient data placement strategies for banded symmetric matrices. The system
 * combines band distribution for tiles within the specified band size with
 * standard off-band distribution for tiles outside the band.
 *
 * The implementation is based on the work presented in IPDPS2021:
 * "Leveraging parsec runtime support to tackle challenging 3d data-sparse matrix problems"
 */

/* ============================================================================
 * Static helper functions for symmetric two-dimensional block cyclic band distribution
 * ============================================================================ */

/**
 * @brief Determine the rank for a tile in symmetric two-dimensional block cyclic band distribution
 *
 * This function determines which process should own a specific tile based on its
 * position in the matrix. Tiles within the band_size use band-optimized distribution,
 * while tiles outside the band use standard off-band distribution.
 *
 * @param[in] desc Data collection descriptor
 * @param[in] ... Variable arguments: m (row index), n (column index)
 * @return Process rank that owns the specified tile
 */
static uint32_t hicma_parsec_sym_twoDBC_band_rank_of(parsec_data_collection_t * desc, ...)
{
    unsigned int m, n;
    va_list ap;
    parsec_matrix_sym_block_cyclic_band_t * dc = (parsec_matrix_sym_block_cyclic_band_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Check tile location within band_size */
    if( (unsigned int)abs((int)m-(int)n) < dc->band_size ) {
        /* New index */
        unsigned int m_band = (unsigned int)abs((int)m - (int)n);
        return dc->band.super.super.rank_of(&dc->band.super.super, m_band, m);
    }

    return dc->off_band.super.super.rank_of(&dc->off_band.super.super, m, n);
}

/**
 * @brief Determine the virtual process ID for a tile in symmetric two-dimensional block cyclic band distribution
 *
 * This function determines the virtual process ID for a specific tile based on its
 * position. Similar to rank_of, it uses band-optimized distribution for tiles
 * within the band_size and standard off-band distribution for tiles outside.
 *
 * @param[in] desc Data collection descriptor
 * @param[in] ... Variable arguments: m (row index), n (column index)
 * @return Virtual process ID for the specified tile
 */
static int32_t hicma_parsec_sym_twoDBC_band_vpid_of(parsec_data_collection_t * desc, ...)
{
    unsigned int m, n;
    va_list ap;
    parsec_matrix_sym_block_cyclic_band_t * dc = (parsec_matrix_sym_block_cyclic_band_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

    /* Check tile location within band_size */
    if( (unsigned int)abs((int)m - (int)n) < dc->band_size ) {
        /* The new m in band */
        unsigned int m_band = (unsigned int)abs((int)m - (int)n);
        return dc->band.super.super.vpid_of(&dc->band.super.super, m_band, m);
    }

    return dc->off_band.super.super.vpid_of(&dc->off_band.super.super, m, n);
}

/**
 * @brief Get data pointer for a tile in symmetric two-dimensional block cyclic band distribution
 *
 * This function retrieves the data pointer for a specific tile. It includes
 * a distributed assertion to ensure the calling process owns the tile.
 * The function uses band-optimized distribution for tiles within the band_size
 * and standard off-band distribution for tiles outside.
 *
 * @param[in] desc Data collection descriptor
 * @param[in] ... Variable arguments: m (row index), n (column index)
 * @return Pointer to the tile data
 */
static parsec_data_t* hicma_parsec_sym_twoDBC_band_data_of(parsec_data_collection_t *desc, ...)
{
    unsigned int m, n;
    va_list ap;
    parsec_matrix_sym_block_cyclic_band_t * dc;
    dc = (parsec_matrix_sym_block_cyclic_band_t *)desc;

    /* Get coordinates */
    va_start(ap, desc);
    m = va_arg(ap, unsigned int);
    n = va_arg(ap, unsigned int);
    va_end(ap);

#if defined(DISTRIBUTED)
    assert(desc->myrank == hicma_parsec_sym_twoDBC_band_rank_of(desc, m, n));
#endif

    /* Check tile location within band_size */
    if( (unsigned int)abs((int)m - (int)n) < dc->band_size ) {
        /* The new m in band */
        unsigned int m_band = (unsigned int)abs((int)m - (int)n);
        return dc->band.super.super.data_of(&dc->band.super.super, m_band, m);
    }

    return dc->off_band.super.super.data_of(&dc->off_band.super.super, m, n);
}

/**
 * @brief Determine the rank for a tile using its key in symmetric two-dimensional block cyclic band distribution
 *
 * This function determines the process rank for a tile using its key identifier.
 * It converts the key to coordinates and then calls the coordinate-based rank_of function.
 *
 * @param[in] desc Data collection descriptor
 * @param[in] key Tile key identifier
 * @return Process rank that owns the specified tile
 */
static uint32_t hicma_parsec_sym_twoDBC_band_rank_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return hicma_parsec_sym_twoDBC_band_rank_of(desc, m, n);
}

/**
 * @brief Determine the virtual process ID for a tile using its key in symmetric two-dimensional block cyclic band distribution
 *
 * This function determines the virtual process ID for a tile using its key identifier.
 * It converts the key to coordinates and then calls the coordinate-based vpid_of function.
 *
 * @param[in] desc Data collection descriptor
 * @param[in] key Tile key identifier
 * @return Virtual process ID for the specified tile
 */
static int32_t hicma_parsec_sym_twoDBC_band_vpid_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return hicma_parsec_sym_twoDBC_band_vpid_of(desc, m, n);
}

/**
 * @brief Get data pointer for a tile using its key in symmetric two-dimensional block cyclic band distribution
 *
 * This function retrieves the data pointer for a tile using its key identifier.
 * It converts the key to coordinates and then calls the coordinate-based data_of function.
 *
 * @param[in] desc Data collection descriptor
 * @param[in] key Tile key identifier
 * @return Pointer to the tile data
 */
static parsec_data_t* hicma_parsec_sym_twoDBC_band_data_of_key(parsec_data_collection_t *desc, parsec_data_key_t key)
{
    int m, n;
    parsec_matrix_block_cyclic_key2coords(desc, key, &m, &n);
    return hicma_parsec_sym_twoDBC_band_data_of(desc, m, n);
}

/* ============================================================================
 * Public interface functions
 * ============================================================================ */

/**
 * @brief Initialize symmetric two-dimensional block cyclic band distribution structure
 *
 * This function initializes a symmetric two-dimensional block cyclic band distribution
 * structure. It inherits from off-band distribution and should be called after
 * initialization of off_band.
 *
 * The function performs the following operations:
 * 1. Initializes the base tiled matrix structure with parameters from off_band
 * 2. Sets the band_size parameter for banded matrix operations
 * 3. Assigns the custom distribution functions (rank_of, vpid_of, data_of, etc.)
 *    that implement the hybrid band/off-band distribution strategy
 *
 * @param[in,out] desc Pointer to the symmetric block cyclic band descriptor
 * @param[in] nodes Number of nodes in the distribution
 * @param[in] myrank Rank of the current process
 * @param[in] band_size Size of the band for banded matrix operations
 */
void hicma_parsec_parsec_matrix_sym_block_cyclic_band_init( parsec_matrix_sym_block_cyclic_band_t *desc,
                                         int nodes, int myrank, int band_size ) {
    parsec_tiled_matrix_t *off_band = &desc->off_band.super;
    parsec_data_collection_t *dc = (parsec_data_collection_t*)desc;

    parsec_tiled_matrix_init( &desc->super, off_band->mtype, off_band->storage, off_band->dtype,
                                 nodes, myrank, off_band->mb, off_band->nb, off_band->lm, off_band->ln,
                                 off_band->i, off_band->j, off_band->m, off_band->n );

    desc->band_size  = band_size;
    dc->rank_of      = hicma_parsec_sym_twoDBC_band_rank_of;
    dc->vpid_of      = hicma_parsec_sym_twoDBC_band_vpid_of;
    dc->data_of      = hicma_parsec_sym_twoDBC_band_data_of;
    dc->rank_of_key  = hicma_parsec_sym_twoDBC_band_rank_of_key;
    dc->vpid_of_key  = hicma_parsec_sym_twoDBC_band_vpid_of_key;
    dc->data_of_key  = hicma_parsec_sym_twoDBC_band_data_of_key;
}
