/**
 * @file hicma_parsec.cpp
 * @brief C++ utility functions for HiCMA PaRSEC implementation
 * 
 * This file provides C++ implementations of set operations and MPI communication
 * utilities that are exposed to C code through extern "C" declarations. The functions
 * handle distributed set operations across MPI processes for workload balancing
 * and optimization in HiCMA computations.
 * 
 * Key Features:
 * - Thread-safe set operations for multi-threaded environments
 * - MPI-based distributed set merging across processes
 * - Memory management for C++ containers exposed to C code
 * - Efficient set-to-array conversion for MPI communication
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include <iostream>
#include <set>
#include <vector>
#include <thread>
#include <mpi.h>

extern "C" {

    /**
     * @brief Create an array of sets for multi-threaded operations
     * 
     * Allocates a vector of sets where each thread can independently
     * add elements to its own set. This enables thread-safe concurrent
     * set operations without requiring explicit synchronization.
     * 
     * @param[in] num_threads Number of threads that will access the sets
     * @return void* Pointer to the allocated vector of sets, or nullptr on error
     * 
     * @note The returned pointer must be freed using free_array_set()
     * @note Each thread should only access its own set to avoid race conditions
     */
    void* create_array_set(int num_threads) {
        if (num_threads <= 0) {
            return nullptr;
        }
        try {
            return new std::vector<std::set<int>>(num_threads);
        } catch (...) {
            return nullptr;
        }
    }

    /**
     * @brief Add a value to a specific set in the array
     * 
     * Thread-safe operation that adds a value to a specific set
     * identified by thread ID. This function assumes that each thread
     * only accesses its own set to maintain thread safety.
     * 
     * @param[in] array_set_ptr Pointer to the vector of sets
     * @param[in] tid Thread ID (0-based index into the vector)
     * @param[in] value Integer value to add to the set
     * 
     * @note Thread safety is maintained by having each thread access only its own set
     * @note The tid parameter must be within the valid range [0, num_threads-1]
     */
    void add_to_set_in_array(void* array_set_ptr, int tid, int value) {
        if (array_set_ptr == nullptr || tid < 0) {
            return; // Safe to call with invalid parameters
        }
        try {
            auto* array_set = static_cast<std::vector<std::set<int>>*>(array_set_ptr);
            if (tid < static_cast<int>(array_set->size())) {
                array_set->at(tid).insert(value);
            }
        } catch (...) {
            // Ignore exceptions during set operations to prevent crashes
        }
    }

    /**
     * @brief Merge all sets within a single process
     * 
     * Combines all sets from different threads within the current process
     * into a single merged set. This is useful for consolidating results
     * from parallel computations before MPI communication.
     * 
     * @param[in] array_set_ptr Pointer to the vector of sets
     * @return void* Pointer to the merged set containing all unique elements, or nullptr on error
     * 
     * @note The returned pointer must be freed using free_merged_set()
     * @note This operation is performed locally within a single MPI process
     */
    void* merge_local_sets(void* array_set_ptr) {
        if (array_set_ptr == nullptr) {
            return nullptr;
        }
        try {
            auto* array_set = static_cast<std::vector<std::set<int>>*>(array_set_ptr);
            std::set<int>* merged_set = new std::set<int>;
            
            /* Iterate through all thread sets and merge them */
            for (const auto& s : *array_set) {
                merged_set->insert(s.begin(), s.end());
            }
            
            return merged_set;
        } catch (...) {
            return nullptr;
        }
    }

    /**
     * @brief Convert a merged set to an array and return its size
     * 
     * Converts a set to a dynamically allocated integer array for
     * MPI communication. The function allocates memory for the array
     * and copies all elements from the set.
     * 
     * @param[in] merged_set_ptr Pointer to the merged set
     * @param[out] size Pointer to store the size of the resulting array
     * @return int* Dynamically allocated array containing set elements
     * 
     * @note The returned array must be freed by the caller using free()
     * @note The size parameter is updated with the actual array size
     * @note Elements in the array are sorted (due to std::set ordering)
     */
    int* convert_merged_set_to_array(void* merged_set_ptr, int* size) {
        auto* merged_set = static_cast<std::set<int>*>(merged_set_ptr);
        *size = merged_set->size();

        /* Allocate memory for the array */
        int* array = (int*) malloc(*size * sizeof(int));

        /* Copy elements from the set to the array */
        int index = 0;
        for (int elem : *merged_set) {
            array[index++] = elem;
        }

        return array;
    }

    /**
     * @brief Merge sets across all MPI processes and return unique elements
     * 
     * Performs a distributed merge operation across all MPI processes.
     * Each process contributes its local set, and the function returns
     * an array containing all unique elements from all processes.
     * 
     * The algorithm:
     * 1. Gather the size of each process's local set
     * 2. Calculate displacements for MPI_Allgatherv
     * 3. Gather all local data into a global array
     * 4. Use a set to filter unique elements
     * 5. Convert the result back to an array
     * 
     * @param[in] local_set_ptr Pointer to the local merged set
     * @param[out] num_elements Pointer to store the total number of unique elements
     * @return int* Dynamically allocated array of unique elements from all processes
     * 
     * @note The returned array must be freed by the caller using free()
     * @note This function performs collective MPI communication (Allgather)
     * @note All processes must call this function with consistent local sets
     */
    int* merge_across_processes(void* local_set_ptr, int* num_elements) {
        auto* local_set = static_cast<std::set<int>*>(local_set_ptr);

        /* Convert the local set to a vector for MPI communication */
        std::vector<int> local_vector(local_set->begin(), local_set->end());
        int local_size = local_vector.size();

        /* Get MPI process information */
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        /* Gather the sizes from all processes */
        std::vector<int> all_sizes(size);
        MPI_Allgather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

        /* Calculate total size and displacements for MPI_Allgatherv */
        int total_size = 0;
        std::vector<int> displs(size);
        for (int i = 0; i < size; ++i) {
            displs[i] = total_size;
            total_size += all_sizes[i];
        }

        /* Gather all local vectors into a global array */
        std::vector<int> global_data(total_size);
        MPI_Allgatherv(local_vector.data(), local_size, MPI_INT,
                global_data.data(), all_sizes.data(), displs.data(), MPI_INT,
                MPI_COMM_WORLD);

        /* Use a set to filter unique elements across all processes */
        std::set<int> global_merged_set(global_data.begin(), global_data.end());

        /* Convert the global merged set back to an array and return the size */
        *num_elements = global_merged_set.size();
        int* result = (int*) malloc(*num_elements * sizeof(int));
        int index = 0;
        for (int elem : global_merged_set) {
            result[index++] = elem;
        }

        return result;
    }

    /**
     * @brief Free the memory allocated for an array of sets
     * 
     * Deallocates the memory for a vector of sets that was created
     * by create_array_set(). This function should be called when
     * the array of sets is no longer needed.
     * 
     * @param[in] array_set_ptr Pointer to the vector of sets to free
     * 
     * @note This function should only be called on pointers returned by create_array_set()
     * @note After calling this function, the pointer becomes invalid
     */
    void free_array_set(void* array_set_ptr) {
        if (array_set_ptr == nullptr) {
            return; // Safe to call on null pointer
        }
        try {
            auto* array_set = static_cast<std::vector<std::set<int>>*>(array_set_ptr);
            delete array_set;
        } catch (...) {
            // Ignore exceptions during cleanup to prevent crashes
        }
    }

    /**
     * @brief Free the memory allocated for a merged set
     * 
     * Deallocates the memory for a merged set that was created
     * by merge_local_sets(). This function should be called when
     * the merged set is no longer needed.
     * 
     * @param[in] merged_set_ptr Pointer to the merged set to free
     * 
     * @note This function should only be called on pointers returned by merge_local_sets()
     * @note After calling this function, the pointer becomes invalid
     */
    void free_merged_set(void* merged_set_ptr) {
        if (merged_set_ptr == nullptr) {
            return; // Safe to call on null pointer
        }
        try {
            auto* merged_set = static_cast<std::set<int>*>(merged_set_ptr);
            delete merged_set;
        } catch (...) {
            // Ignore exceptions during cleanup to prevent crashes
        }
    }

}

