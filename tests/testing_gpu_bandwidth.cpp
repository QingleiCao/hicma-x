#include <cuda_runtime.h>
#include <iostream>

/**
 * @file testing_gpu_bandwidth.cpp
 * @brief CUDA GPU memory bandwidth testing and benchmarking
 * 
 * This program measures the memory bandwidth between host and GPU device
 * by performing memory transfers and timing the operations. It provides
 * performance metrics for Host-to-Device (H2D) and Device-to-Host (D2H)
 * memory transfers.
 * 
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2023-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2018-2023     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 */

/**
 * @brief Check CUDA error and exit if error occurs
 * @param err CUDA error code
 * @param msg Error message to display
 */
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Main function for GPU bandwidth testing
 * @return 0 on success, non-zero on failure
 */
int main() {
    // Test data size: 100 MB for meaningful bandwidth measurement
    const size_t dataSize = 100 * 1024 * 1024;  // 100 MB
    char *h_data = nullptr, *d_data = nullptr;

    std::cout << "Starting CUDA GPU bandwidth test..." << std::endl;
    std::cout << "Data size: " << dataSize / (1024 * 1024) << " MB" << std::endl << std::endl;

    // Allocate pinned host memory for optimal GPU transfer performance
    checkCudaError(cudaMallocHost(&h_data, dataSize), "Host memory allocation failed");
    
    // Allocate device memory
    checkCudaError(cudaMalloc(&d_data, dataSize), "Device memory allocation failed");

    // Create CUDA events for precise timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");

    // Test Host-to-Device (H2D) bandwidth
    std::cout << "Testing Host-to-Device (H2D) bandwidth..." << std::endl;
    
    // Record start time
    checkCudaError(cudaEventRecord(start, 0), "Failed to record start event");
    
    // Perform H2D memory transfer
    checkCudaError(cudaMemcpy(d_data, h_data, dataSize, cudaMemcpyHostToDevice), 
                   "Host to Device copy failed");
    
    // Record stop time
    checkCudaError(cudaEventRecord(stop, 0), "Failed to record stop event");
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");

    // Calculate H2D bandwidth
    float h2d_milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&h2d_milliseconds, start, stop), 
                   "Failed to compute elapsed time");
    
    double h2d_bandwidth = (dataSize / (h2d_milliseconds / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "  H2D Transfer time: " << h2d_milliseconds << " ms" << std::endl;
    std::cout << "  H2D Bandwidth: " << h2d_bandwidth << " GB/s" << std::endl << std::endl;

    // Test Device-to-Host (D2H) bandwidth
    std::cout << "Testing Device-to-Host (D2H) bandwidth..." << std::endl;
    
    // Record start time for D2H
    checkCudaError(cudaEventRecord(start, 0), "Failed to record start event for D2H");
    
    // Perform D2H memory transfer
    checkCudaError(cudaMemcpy(h_data, d_data, dataSize, cudaMemcpyDeviceToHost), 
                   "Device to Host copy failed");
    
    // Record stop time for D2H
    checkCudaError(cudaEventRecord(stop, 0), "Failed to record stop event for D2H");
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event for D2H");

    // Calculate D2H bandwidth
    float d2h_milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&d2h_milliseconds, start, stop), 
                   "Failed to compute elapsed time for D2H");
    
    double d2h_bandwidth = (dataSize / (d2h_milliseconds / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "  D2H Transfer time: " << d2h_milliseconds << " ms" << std::endl;
    std::cout << "  D2H Bandwidth: " << d2h_bandwidth << " GB/s" << std::endl << std::endl;

    // Test Device-to-Device (D2D) bandwidth if multiple GPUs available
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "Failed to get device count");
    
    double d2d_bandwidth = 0.0;  // Initialize D2D bandwidth variable
    if (deviceCount > 1) {
        std::cout << "Testing Device-to-Device (D2D) bandwidth..." << std::endl;
        
        // Allocate memory on second device
        char *d_data2 = nullptr;
        checkCudaError(cudaSetDevice(1), "Failed to set device 1");
        checkCudaError(cudaMalloc(&d_data2, dataSize), "Failed to allocate memory on device 1");
        
        // Switch back to device 0 for transfer
        checkCudaError(cudaSetDevice(0), "Failed to set device 0");
        
        // Record start time for D2D
        checkCudaError(cudaEventRecord(start, 0), "Failed to record start event for D2D");
        
        // Perform D2D memory transfer
        checkCudaError(cudaMemcpy(d_data2, d_data, dataSize, cudaMemcpyDeviceToDevice), 
                       "Device to Device copy failed");
        
        // Record stop time for D2D
        checkCudaError(cudaEventRecord(stop, 0), "Failed to record stop event for D2D");
        checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event for D2D");

        // Calculate D2D bandwidth
        float d2d_milliseconds = 0;
        checkCudaError(cudaEventElapsedTime(&d2d_milliseconds, start, stop), 
                       "Failed to compute elapsed time for D2D");
        
        d2d_bandwidth = (dataSize / (d2d_milliseconds / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
        std::cout << "  D2D Transfer time: " << d2d_milliseconds << " ms" << std::endl;
        std::cout << "  D2D Bandwidth: " << d2d_bandwidth << " GB/s" << std::endl << std::endl;
        
        // Clean up second device memory
        checkCudaError(cudaSetDevice(1), "Failed to set device 1 for cleanup");
        checkCudaError(cudaFree(d_data2), "Failed to free memory on device 1");
        checkCudaError(cudaSetDevice(0), "Failed to set device 0 after cleanup");
    }

    // Print summary
    std::cout << "Bandwidth Test Summary:" << std::endl;
    std::cout << "=======================" << std::endl;
    std::cout << "H2D Bandwidth: " << h2d_bandwidth << " GB/s" << std::endl;
    std::cout << "D2H Bandwidth: " << d2h_bandwidth << " GB/s" << std::endl;
    if (deviceCount > 1) {
        std::cout << "D2D Bandwidth: " << d2d_bandwidth << " GB/s" << std::endl;
    }
    std::cout << "=======================" << std::endl;

    // Clean up allocated resources
    checkCudaError(cudaEventDestroy(start), "Failed to destroy start event");
    checkCudaError(cudaEventDestroy(stop), "Failed to destroy stop event");
    checkCudaError(cudaFreeHost(h_data), "Host memory free failed");
    checkCudaError(cudaFree(d_data), "Device memory free failed");

    std::cout << "Bandwidth test completed successfully!" << std::endl;
    return EXIT_SUCCESS;
}

