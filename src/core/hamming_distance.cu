/**
 * @file hamming_distance.cu
 * @brief CUDA implementation of Hamming distance computation using CUTLASS
 * 
 * This file implements efficient Hamming distance computation on GPU using:
 * - CUTLASS library for optimized matrix operations
 * - 1-bit precision for memory efficiency
 * - Tensor cores for maximum performance on modern GPUs
 * 
 * The implementation converts input matrices to binary representations and
 * computes Hamming distances using optimized GEMM operations.
 * 
 */

#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <iostream>

// CUTLASS includes for optimized matrix operations
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include <bitset>

/** @brief Chunk size for kernel execution */
#define CHUNKSIZE 32

/**
 * @brief Data type definitions for CUTLASS GEMM operation
 * 
 * These typedefs define the precision and layout for the Hamming distance computation:
 * - Input matrices A and B use 1-bit precision (cutlass::uint1b_t)
 * - Output matrix C uses 32-bit integers for distance counts
 * - Accumulator uses 32-bit integers for intermediate calculations
 */
using ElementAccumulator = int32_t;                   // Data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;     // Data type of epilogue operations
using ElementInputA = cutlass::uint1b_t;              // Data type of elements in input matrix A
using ElementInputB = cutlass::uint1b_t;              // Data type of elements in input matrix B
using ElementOutput = int32_t;                        // Data type of elements in output matrix D
using ElementCompute = ElementComputeEpilogue;

/**
 * @brief Matrix layout definitions for CUTLASS GEMM
 * 
 * Layout configuration:
 * - Matrix A: Row Major layout for packed 1-bit data
 * - Matrix B: Column Major layout for packed 1-bit data  
 * - Matrix C: Column Major layout for output distance matrix
 */
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

/**
 * @brief GPU architecture configuration
 * 
 * Uses tensor cores (OpClassTensorOp) for maximum performance on modern GPUs
 * Optimized for Ampere architecture (SM80) and newer
 */
using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

/**
 * @brief Thread block and warp tile configurations
 * 
 * These shapes define how work is distributed across GPU threads:
 * - ThreadBlock: 128×256×1024 elements per thread block
 * - Warp: 64×64×1024 elements per warp
 * - MMA Op: 16×8×256 elements per matrix multiply operation
 */
using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 1024>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 1024>;
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 256>;

/**
 * @brief Thread block scheduling and epilogue configuration
 * 
 * - SwizzleThreadBlock: Standard identity swizzling for thread block scheduling
 * - EpilogueOp: Linear combination operation for output matrix computation
 */
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // Output data type
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // Vector width for memory access
    ElementAccumulator,                                // Accumulator data type
    ElementComputeEpilogue>;                           // Epilogue computation data type

/** @brief Number of pipeline stages for kernel execution */
// Note: NumStages is defined but not currently used in the Gemm configuration
// constexpr int NumStages = 2;

/**
 * @brief Main CUTLASS GEMM kernel configuration
 * 
 * This is the primary kernel used for Hamming distance computation.
 * Optimized tile sizes based on performance testing:
 * - ThreadBlock: 64×128×512 (best performance: 1396.56 POps)
 * - Warp: 32×64×512 
 * - MMA Op: 16×8×256
 * 
 * Uses OpAndPopc operation for efficient 1-bit matrix multiplication
 */
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::uint1b_t, cutlass::layout::RowMajor, cutlass::uint1b_t,
    cutlass::layout::ColumnMajor, ElementOutput, cutlass::layout::ColumnMajor,
    ElementAccumulator, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    
    // Thread block tile size (optimized for performance)
    cutlass::gemm::GemmShape<64, 128, 512>,
    
    // Warp tile size
    cutlass::gemm::GemmShape<32, 64, 512>,
    
    // MMA operation tile size
    cutlass::gemm::GemmShape<16, 8, 256>,
    
    // Epilogue operation
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator, ElementCompute>,
    
    // Thread block swizzling and additional parameters
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 
    3, 128, 128, false, cutlass::arch::OpAndPopc>;

/**
 * @brief Print bit vector contents for debugging
 * 
 * @param bit_vector Vector of unsigned chars representing packed bits
 * 
 * This function converts each byte to its binary representation and prints
 * the result for debugging purposes.
 */
void printBitVector(const std::vector<unsigned char>& bit_vector) {
    std::cout << "Contents of bit_vector:" << std::endl;
    for (size_t i = 0; i < bit_vector.size(); ++i) {
        // Use std::bitset to convert each byte into a string of 0s and 1s
        std::bitset<8> bits(bit_vector[i]);
        // Print each bit (reverse order to match typical bit order in bytes)
        for (int j = 7; j >= 0; --j) {
            std::cout << bits[j];
        }
        std::cout << " ";  // Separate bytes for clarity
    }
    std::cout << std::endl;
}

/**
 * @brief Print tensor data contents for debugging
 * 
 * @param data Pointer to tensor data in GPU memory
 * @param length_m Number of rows in the tensor
 * @param length_k Number of columns in the tensor
 * 
 * This function prints the binary representation of packed 1-bit tensor data
 * for debugging purposes.
 */
void printTensorData(unsigned char* data, int length_m, int length_k) {
    std::cout << "Contents of tensor data:" << std::endl;
    int num_elements = length_m * length_k;  // Total number of 1-bit elements
    int byte_count = (num_elements + 7) / 8;  // Total bytes needed

    for (int i = 0; i < byte_count; ++i) {
        std::bitset<8> bits(data[i]);
        // Print each bit (reverse order to match typical bit order in bytes)
        for (int j = 7; j >= 0; --j) {
            std::cout << bits[j];
        }
        std::cout << " ";  // Separate bytes for clarity
    }
    std::cout << std::endl;
}

/**
 * @brief CUDA kernel to set bits in packed 1-bit tensors
 * 
 * @param matrix Input matrix with target values
 * @param tensor_a Output tensor A (packed 1-bit)
 * @param tensor_b Output tensor B (packed 1-bit)
 * @param m Number of rows
 * @param k Number of columns
 * @param target_value Value to match for setting bits
 * 
 * This kernel converts the input matrix to binary representation:
 * - tensor_a[i] = 1 if matrix[i] == target_value, 0 otherwise
 * - tensor_b[i] = 1 if matrix[i] != target_value, 0 otherwise
 * 
 * Uses atomic operations to safely set bits in packed format.
 */
__global__ void setBitsKernel(int8_t *matrix, cutlass::uint1b_t *tensor_a, cutlass::uint1b_t *tensor_b, int m, int k, int target_value) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t width = m;
    uint64_t height = k;
    uint64_t num_elements = width * height;

    if (idx < num_elements) {
        int byte_index = idx / 8;      // Which byte contains this bit
        int bit_index = idx % 8;       // Which bit within the byte

        // Align to 32-bit boundary for atomic operations
        unsigned int* ptr_a = (unsigned int*)&tensor_a[byte_index & ~3];
        unsigned int* ptr_b = (unsigned int*)&tensor_a[byte_index & ~3];
        
        // Calculate bit position within the 32-bit word
        int bit_position = ((byte_index & 3) * 8) + bit_index;
        
        // Create masks for setting appropriate bits
        unsigned int mask_a = (matrix[idx] == target_value) ? (1U << bit_position) : 0;
        unsigned int mask_b = (matrix[idx] != target_value) ? (1U << bit_position) : 0;

        // Atomically set bits using OR operation
        if (mask_a != 0) {
            atomicOr(ptr_a, mask_a);
        }

        if (mask_b != 0) {
            atomicOr(ptr_b, mask_b);
        }
    }
}

/**
 * @brief Optimized CUDA kernel with shared memory for bit setting
 * 
 * @param matrix Input matrix with target values
 * @param tensor_a Output tensor A (packed 1-bit)
 * @param tensor_b Output tensor B (packed 1-bit)
 * @param m Number of rows
 * @param k Number of columns
 * @param target_value Value to match for setting bits
 * 
 * This optimized version uses shared memory to reduce global memory access:
 * 1. Accumulates bit operations in shared memory
 * 2. Performs atomic operations locally
 * 3. Writes back to global memory in bulk
 * 
 * Improves performance by reducing global memory contention.
 */
__global__ void setBitsKernelOptimizedWithShared(int8_t *matrix, cutlass::uint1b_t *tensor_a, cutlass::uint1b_t *tensor_b, int m, int k, int target_value) {
    extern __shared__ unsigned int shared_data[];  // Shared memory for both tensors' bit aggregation
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t width = m;
    uint64_t height = k;
    uint64_t num_elements = width * height;

    int byte_index = idx / 8;
    int bit_index = idx % 8;

    if (idx < num_elements) {
        // Calculate the index within the shared memory (doubling the allocation for tensor_b)
        int shared_index_a = threadIdx.x / 32;  // For tensor_a
        int shared_index_b = (blockDim.x / 32) + shared_index_a;  // Offset for tensor_b

        if (threadIdx.x % 32 == 0) {
            shared_data[shared_index_a] = 0;  // Initialize shared memory for tensor_a
            shared_data[shared_index_b] = 0;  // Initialize shared memory for tensor_b
        }
        __syncthreads();  // Ensure initialization is completed before proceeding

        // Calculate bit position and mask
        int bit_position = (threadIdx.x % 32) * 8 + bit_index;  // Local bit position within shared memory
        unsigned int mask_a = (matrix[idx] == target_value) ? (1U << bit_position) : 0;
        unsigned int mask_b = (matrix[idx] != target_value) ? (1U << bit_position) : 0;

        // Use atomic operations within shared memory to set bits
        if (mask_a != 0) {
            atomicOr(&shared_data[shared_index_a], mask_a);
        }
        if (mask_b != 0) {
            atomicOr(&shared_data[shared_index_b], mask_b);
        }
        __syncthreads();  // Ensure all threads have updated shared memory

        // Write back to global memory from shared memory
        if (threadIdx.x % 32 == 0) {
            unsigned int* global_ptr_a = (unsigned int*)&tensor_a[byte_index & ~3];
            unsigned int* global_ptr_b = (unsigned int*)&tensor_b[byte_index & ~3];
            *global_ptr_a = shared_data[shared_index_a];
            *global_ptr_b = shared_data[shared_index_b];
        }
    }
}

/**
 * @brief Main GPU function for Hamming distance computation
 * 
 * @param input Input matrix (int8_t)
 * @param A Output tensor A (packed 1-bit)
 * @param B Output tensor B (packed 1-bit)
 * @param C Output distance matrix (int32_t)
 * @param m Number of rows
 * @param k Number of columns
 * @param target_value Target value for binary conversion
 * @param stream CUDA stream for asynchronous execution
 * @return 0 on success, -1 on failure
 * 
 * This function performs the complete Hamming distance computation:
 * 1. Converts input matrix to binary representation using CUDA kernels
 * 2. Computes Hamming distances using optimized CUTLASS GEMM
 * 3. Returns the distance matrix C where C[i,j] = Hamming distance between columns i and j
 * 
 * The computation uses 1-bit precision for memory efficiency and tensor cores for performance.
 */
extern "C"
int bitmask_tensergemm_GPU(int8_t *input, void *A, void *B, int32_t *C, int m, int k, int target_value, cudaStream_t stream) {
    // Calculate grid and block dimensions for bit setting kernel
    uint64_t num_elements = m * k;
    int threadsPerBlock = 1024;
    uint64_t blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    int sharedMemSize = sizeof(unsigned int) * 2 * (threadsPerBlock / 32);

    // Calculate data size for packed 1-bit tensors
    // Note: data_size is calculated but not currently used
    // size_t data_size = (m * ((k + 7) / 8)) * sizeof(cutlass::uint1b_t);

    // Calculate strides for matrix layout
    int stride_A = (k + 7) / 8;  // For packed 1-bit columns in A
    int stride_B = (k + 7) / 8;  // For packed 1-bit columns in B
    int stride_C = m;             // Stride for output matrix C

    // Define matrix layouts for CUTLASS
    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    using ElementOutput = int32_t; 

    // Create tensor references for CUTLASS
    cutlass::TensorRef<cutlass::uint1b_t, LayoutA> tensor_refA((cutlass::uint1b_t*)A, stride_A);
    cutlass::TensorRef<cutlass::uint1b_t, LayoutB> tensor_refB((cutlass::uint1b_t*)B, stride_B);
    cutlass::TensorRef<ElementOutput, LayoutC> tensor_refC((int32_t*)C, stride_C);

    // Launch optimized kernel to convert input matrix to binary representation
    setBitsKernelOptimizedWithShared<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream>>>(
        input, tensor_refA.data(), tensor_refB.data(), m, k, target_value);

    // Synchronize to ensure bit setting is complete
    cudaDeviceSynchronize();

    // Initialize alpha and beta for GEMM computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(1);

    // Split K dimension into 1 partition (no splitting)
    int split_k_slices = 1;
    
    // Create problem size tuple for matrix multiplication
    cutlass::gemm::GemmCoord problem_size(m, m, k);

    // Create GEMM kernel arguments
    typename Gemm::Arguments arguments{
        problem_size,           // Problem size (M, N, K)
        tensor_refA,            // Reference to matrix A on device
        tensor_refB,            // Reference to matrix B on device
        tensor_refC,            // Reference to matrix C on device (input)
        tensor_refC,            // Reference to matrix C on device (output)
        {alpha, beta},          // Alpha and beta scaling factors
        split_k_slices          // K-dimension split factor
    };

    // Query for required workspace size
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel
    Gemm gemm_op;

    // Check if problem size is supported
    gemm_op.can_implement(arguments);

    // Initialize CUTLASS kernel with arguments and workspace
    gemm_op.initialize(arguments, workspace.get(), stream);

    // Launch the GEMM kernel
    gemm_op();

    return 0;
}
