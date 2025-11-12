/**
 * @file testing_2gpu_gemm.cpp
 * @brief Benchmark comparison between explicit memory transfer and unified memory for GEMM operations
 * 
 * This program compares the performance of GEMM (General Matrix Multiply) operations
 * using two different memory management approaches:
 * 1. Explicit memory transfers between GPUs using cudaMemcpyPeer
 * 2. Unified memory management with automatic data migration
 * 
 * The benchmark helps evaluate the trade-offs between explicit control over memory
 * placement and the convenience of unified memory management.
 * 
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>
#include <stdexcept>
#include <iomanip>

// CUDA error checking macro - exits program on error
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// cuBLAS error checking macro - exits program on error
#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << status << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * @brief Allocates managed memory on a specific CUDA device
 * 
 * @param size Size of memory to allocate in bytes
 * @param cudaIndex CUDA device index
 * @return Pointer to allocated memory
 */
double *allocateMemory(size_t size, int cudaIndex) {
    double *ptr;
    CHECK_CUDA(cudaSetDevice(cudaIndex));
    CHECK_CUDA(cudaMallocManaged(&ptr, size));
    
    // Set memory advice for optimal performance
    CHECK_CUDA(cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, cudaIndex));
    CHECK_CUDA(cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, cudaIndex));
    
    return ptr;
}

/**
 * @brief Validates input parameters for the benchmark
 * 
 * @param problemSize Matrix dimension
 * @param iterations Number of iterations to run
 * @param maxProblemSize Maximum allowed problem size to prevent memory issues
 */
void validateInputs(size_t problemSize, int iterations, size_t maxProblemSize = 16384) {
    if (problemSize == 0) {
        throw std::invalid_argument("Problem size must be greater than 0");
    }
    if (problemSize > maxProblemSize) {
        throw std::invalid_argument("Problem size too large: " + std::to_string(problemSize) + 
                                   " (max: " + std::to_string(maxProblemSize) + ")");
    }
    if (iterations <= 0) {
        throw std::invalid_argument("Number of iterations must be positive");
    }
    
    // Check if we have at least 2 GPUs
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 2) {
        throw std::runtime_error("This benchmark requires at least 2 CUDA devices");
    }
}

/**
 * @brief Initializes input matrices with test data
 * 
 * @param A Input matrix A
 * @param B Input matrix B  
 * @param C Output matrix C (initialized to zero)
 * @param size Matrix dimension
 */
void initializeMatrices(double *A, double *B, double *C, size_t size) {
    for (size_t i = 0; i < size * size; ++i) {
        A[i] = 1.0;  // Initialize A with ones
        B[i] = 1.0;  // Initialize B with ones
        C[i] = 0.0;  // Initialize C with zeros
    }
}

/**
 * @brief Runs the benchmark comparing explicit vs unified memory approaches
 * 
 * @param problemSize Matrix dimension (S x S)
 * @param iterations Number of iterations to run for averaging
 */
void run_benchmark(size_t problemSize, int iterations) {
    const size_t buffer_size = problemSize * problemSize * sizeof(double);
    const double alpha = 1.0, beta = 0.0;
    
    // Host memory pointers (unified memory)
    double *A_h, *B_h, *C_h;
    
    // Device memory pointers for explicit transfers
    double *A_d0, *B_d0, *C_d0;  // GPU 0
    double *A_d1, *B_d1, *C_d1;  // GPU 1
    
    // CUDA handles and streams
    cublasHandle_t handle0, handle1;
    cudaStream_t stream0, stream1;
    
    std::cout << "Setting up CUDA environment..." << std::endl;
    
    // Initialize GPU 0
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaStreamCreate(&stream0));
    CHECK_CUBLAS(cublasCreate(&handle0));
    CHECK_CUBLAS(cublasSetStream(handle0, stream0));
    
    // Initialize GPU 1
    CHECK_CUDA(cudaSetDevice(1));
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUBLAS(cublasCreate(&handle1));
    CHECK_CUBLAS(cublasSetStream(handle1, stream1));
    
    std::cout << "Allocating memory..." << std::endl;
    
    // Allocate unified memory (accessible from both GPUs)
    CHECK_CUDA(cudaMallocManaged(&A_h, buffer_size));
    CHECK_CUDA(cudaMallocManaged(&B_h, buffer_size));
    CHECK_CUDA(cudaMallocManaged(&C_h, buffer_size));
    
    // Allocate device-specific memory for explicit transfers
    CHECK_CUDA(cudaSetDevice(0));
    A_d0 = allocateMemory(buffer_size, 0);
    B_d0 = allocateMemory(buffer_size, 0);
    C_d0 = allocateMemory(buffer_size, 0);
    
    CHECK_CUDA(cudaSetDevice(1));
    A_d1 = allocateMemory(buffer_size, 1);
    B_d1 = allocateMemory(buffer_size, 1);
    C_d1 = allocateMemory(buffer_size, 1);
    
    std::cout << "Initializing matrices..." << std::endl;
    
    // Initialize input matrices with test data
    initializeMatrices(A_h, B_h, C_h, problemSize);
    
    // Copy data to GPU 0
    std::cout << "Transferring data to GPU 0..." << std::endl;
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaMemcpy(A_d0, A_h, buffer_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_d0, B_h, buffer_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(C_d0, C_h, buffer_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    std::cout << "Running benchmarks..." << std::endl;
    
    // Benchmark 1: Explicit memory transfers between GPUs
    std::cout << "  Benchmark 1: Explicit memory transfers..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        // Copy data from GPU 0 to GPU 1
        CHECK_CUDA(cudaMemcpyPeer(A_d1, 1, A_d0, 0, buffer_size));
        CHECK_CUDA(cudaMemcpyPeer(B_d1, 1, B_d0, 0, buffer_size));
        CHECK_CUDA(cudaMemcpyPeer(C_d1, 1, C_d0, 0, buffer_size));
        
        // Perform GEMM on GPU 1
        CHECK_CUBLAS(cublasDgemm(handle1, CUBLAS_OP_N, CUBLAS_OP_N, 
                                 problemSize, problemSize, problemSize, 
                                 &alpha, A_d1, problemSize, B_d1, problemSize, 
                                 &beta, C_d1, problemSize));
        
        // Copy result back to GPU 0
        CHECK_CUDA(cudaMemcpyPeer(C_d0, 0, C_d1, 1, buffer_size));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double explicit_time = std::chrono::duration<double>(end - start).count() / iterations;
    
    // Benchmark 2: Unified memory (automatic data migration)
    std::cout << "  Benchmark 2: Unified memory..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        // GEMM using unified memory - CUDA automatically handles data placement
        CHECK_CUBLAS(cublasDgemm(handle1, CUBLAS_OP_N, CUBLAS_OP_N, 
                                 problemSize, problemSize, problemSize, 
                                 &alpha, A_d0, problemSize, B_d0, problemSize, 
                                 &beta, C_d0, problemSize));
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    end = std::chrono::high_resolution_clock::now();
    double unified_time = std::chrono::duration<double>(end - start).count() / iterations;
    
    // Print results in a formatted table
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "BENCHMARK RESULTS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Matrix Size: " << problemSize << " x " << problemSize << std::endl;
    std::cout << "Iterations: " << iterations << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << std::left << std::setw(25) << "Method" 
              << std::setw(15) << "Time/GEMM (s)" 
              << std::setw(15) << "Total Time (s)" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    std::cout << std::left << std::setw(25) << "Explicit Transfer" 
              << std::setw(15) << explicit_time 
              << std::setw(15) << (explicit_time * iterations) << std::endl;
    std::cout << std::left << std::setw(25) << "Unified Memory" 
              << std::setw(15) << unified_time 
              << std::setw(15) << (unified_time * iterations) << std::endl;
    
    // Calculate and display performance difference
    double speedup = explicit_time / unified_time;
    std::cout << std::string(55, '-') << std::endl;
    std::cout << "Speedup (Explicit/Unified): " << std::fixed << std::setprecision(2) 
              << speedup << "x" << std::endl;
    
    if (speedup > 1.0) {
        std::cout << "Unified memory is " << speedup << "x faster" << std::endl;
    } else {
        std::cout << "Explicit transfer is " << (1.0/speedup) << "x faster" << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;
    
    // Cleanup all allocated resources
    std::cout << "Cleaning up resources..." << std::endl;
    
    CHECK_CUDA(cudaFree(A_h));
    CHECK_CUDA(cudaFree(B_h));
    CHECK_CUDA(cudaFree(C_h));
    CHECK_CUDA(cudaFree(A_d0));
    CHECK_CUDA(cudaFree(B_d0));
    CHECK_CUDA(cudaFree(C_d0));
    CHECK_CUDA(cudaFree(A_d1));
    CHECK_CUDA(cudaFree(B_d1));
    CHECK_CUDA(cudaFree(C_d1));
    
    CHECK_CUBLAS(cublasDestroy(handle0));
    CHECK_CUBLAS(cublasDestroy(handle1));
    CHECK_CUDA(cudaStreamDestroy(stream0));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    
    std::cout << "Benchmark completed successfully!" << std::endl;
}

/**
 * @brief Main function - entry point of the program
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return Exit status
 */
int main(int argc, char **argv) {
    try {
        // Check command line arguments
        if (argc != 3) {
            std::cerr << "Usage: " << argv[0] << " <problem_size> <iterations>\n";
            std::cerr << "  problem_size: Matrix dimension (e.g., 1024 for 1024x1024)\n";
            std::cerr << "  iterations:   Number of iterations to run (e.g., 100)\n";
            std::cerr << "\nExample: " << argv[0] << " 1024 100\n";
            return EXIT_FAILURE;
        }
        
        // Parse and validate input parameters
        size_t problemSize = std::stoul(argv[1]);
        int iterations = std::stoi(argv[2]);
        
        // Validate inputs before proceeding
        validateInputs(problemSize, iterations);
        
        // Print benchmark configuration
        std::cout << "2-GPU GEMM Benchmark Configuration:" << std::endl;
        std::cout << "  Problem Size: " << problemSize << " x " << problemSize << std::endl;
        std::cout << "  Iterations: " << iterations << std::endl;
        std::cout << "  Memory per Matrix: " << (problemSize * problemSize * sizeof(double) / (1024*1024)) << " MB" << std::endl;
        std::cout << std::endl;
        
        // Run the benchmark
        run_benchmark(problemSize, iterations);
        
        return EXIT_SUCCESS;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return EXIT_FAILURE;
    }
}
