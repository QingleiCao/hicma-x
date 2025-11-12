#include "hicma_parsec.h"
#include "hicma_parsec_io.h"

/**
 * @brief Create MPI datatype for 2D data distribution
 * 
 * Creates a custom MPI datatype for efficient 2D data distribution across
 * a process grid. This function optimizes data transfer patterns for
 * matrix operations in distributed memory systems.
 * 
 * The function sets up a 2D cyclic distribution pattern where matrix blocks
 * are distributed across a process grid in a round-robin fashion. This
 * distribution pattern is commonly used in parallel linear algebra libraries
 * for load balancing and minimizing communication overhead.
 * 
 * @param[in] procs Total number of processes in the MPI communicator
 * @param[in] mpirank Rank of the current process (0-based)
 * @param[in] M Number of rows in the global matrix
 * @param[in] N Number of columns in the global matrix
 * @param[in] distM Row distribution factor (typically MPI_DISTRIBUTE_CYCLIC)
 * @param[in] distN Column distribution factor (typically MPI_DISTRIBUTE_CYCLIC)
 * @param[in] mb Block size for rows (tile height)
 * @param[in] nb Block size for columns (tile width)
 * @param[in] prow Row coordinate in process grid
 * @param[in] pcol Column coordinate in process grid
 * @param[in] oldtype Original MPI datatype (e.g., MPI_FLOAT, MPI_DOUBLE)
 * @param[out] newtype New MPI datatype for 2D distribution
 */
void Creating_MPI_Datatype_2D(int procs, int mpirank,
                int M, int N, int distM, int distN,
                int mb, int nb, int prow, int pcol,
                MPI_Datatype oldtype, MPI_Datatype *newtype)
{
        int pdims[2], distribs[2], dims[2], dargs[2];
        int ierr;

        // Set up process grid dimensions
        pdims[0] = prow;  // Number of processes in row dimension
        pdims[1] = pcol;  // Number of processes in column dimension

        // Set up global matrix dimensions
        dims[0] = M;      // Global number of rows
        dims[1] = N;      // Global number of columns

        // Set up distribution patterns
        distribs[0] = distM;  // Row distribution pattern
        distribs[1] = distN;  // Column distribution pattern

        // Set up block sizes for tiling
        dargs[0] = mb;    // Block size in row dimension
        dargs[1] = nb;    // Block size in column dimension

        // Create 2D cyclic distribution datatype
        // This creates a datatype that describes how the matrix is distributed
        // across the process grid using cyclic distribution
        ierr = MPI_Type_create_darray(procs, mpirank, 2,
                        dims, distribs, dargs, pdims, MPI_ORDER_FORTRAN,
                        oldtype, newtype);

        // Check for MPI errors during datatype creation
        if (ierr != 0) {
                printf("\n ====> Error in MPI_Type_create_darray: %d\n", ierr);
                fflush(stdout);
        }

        // Commit the new datatype to make it available for use
        ierr = MPI_Type_commit(newtype);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_Type_commit: %d\n", ierr);
                fflush(stdout);
        }
}

/**
 * @brief Write single precision matrix to file using MPI I/O
 * 
 * Writes a single precision matrix to a file using MPI I/O for efficient
 * parallel file operations. The function uses custom MPI datatypes for
 * optimal data distribution and transfer patterns.
 * 
 * This function performs collective I/O operations where all processes
 * participate in writing their portion of the distributed matrix to a
 * single file. The file view is set up to handle the distributed data
 * layout efficiently.
 * 
 * @param[in] filename Output filename for the matrix data
 * @param[in] mpirank Rank of the current MPI process
 * @param[in] matrix Single precision matrix data to write
 * @param[in] size Size of the matrix data in elements
 * @param[in] oldtype Original MPI datatype for the matrix elements
 * @param[in] newtype New MPI datatype optimized for distribution
 */
void MPI_Writing_sfile(const char *filename, int mpirank, float *matrix, MPI_Count size, MPI_Datatype oldtype, MPI_Datatype newtype)
{
        int ierr;
        MPI_File infile;
        MPI_Status mpistatus;

        // Get information about the MPI datatype for validation
        MPI_Aint file_type_extent, lb;
        MPI_Count file_type_size;
        MPI_Type_get_extent(newtype, &lb, &file_type_extent);
        MPI_Type_size_x(newtype, &file_type_size);
        int64_t buffer_size = file_type_size / sizeof(float);

        // Validate that the buffer size matches the expected size
        if (buffer_size != size) {
                printf("\n On myrank:%d writing data to: %s parsec size:%lld not matching mpi local:%ld\n", 
                       mpirank, filename, size, buffer_size);
                fflush(stdout);
        }

        // Open the file for writing with collective I/O
        ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                            MPI_INFO_NULL, &infile);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_open: %d\n", ierr);
                fflush(stdout);
        }

        // Set the file view to handle distributed data layout
        ierr = MPI_File_set_view(infile, 0, oldtype, newtype, "native", MPI_INFO_NULL);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_set_view: %d\n", ierr);
                fflush(stdout);
        }

        // Perform collective write operation
        ierr = MPI_File_write_all(infile, matrix, size, oldtype, &mpistatus);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_write_all: %d\n", ierr);
                fflush(stdout);
        }

        // Close the file
        ierr = MPI_File_close(&infile);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_close: %d\n", ierr);
                fflush(stdout);
        }
}


/**
 * @brief Write double precision matrix to file using MPI I/O
 * 
 * Writes a double precision matrix to a file using MPI I/O for efficient
 * parallel file operations. The function uses custom MPI datatypes for
 * optimal data distribution and transfer patterns.
 * 
 * This function performs collective I/O operations where all processes
 * participate in writing their portion of the distributed matrix to a
 * single file. The file view is set up to handle the distributed data
 * layout efficiently.
 * 
 * @param[in] filename Output filename for the matrix data
 * @param[in] mpirank Rank of the current MPI process
 * @param[in] matrix Double precision matrix data to write
 * @param[in] size Size of the matrix data in elements
 * @param[in] oldtype Original MPI datatype for the matrix elements
 * @param[in] newtype New MPI datatype optimized for distribution
 */
void MPI_Writing_dfile(const char *filename, int mpirank, double *matrix, MPI_Count size, MPI_Datatype oldtype, MPI_Datatype newtype)
{
        int ierr;
        MPI_File infile;
        MPI_Status mpistatus;

        // Get information about the MPI datatype for validation
        MPI_Aint file_type_extent, lb;
        MPI_Count file_type_size;
        MPI_Type_get_extent(newtype, &lb, &file_type_extent);
        MPI_Type_size_x(newtype, &file_type_size);
        int64_t buffer_size = file_type_size / sizeof(double);

        // Validate that the buffer size matches the expected size
        if (buffer_size != size) {
                printf("\n On myrank:%d writing data to: %s parsec size:%lld not matching mpi local:%ld\n", 
                       mpirank, filename, size, buffer_size);
                fflush(stdout);
        }

        // Open the file for writing with collective I/O
        ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                            MPI_INFO_NULL, &infile);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_open: %d\n", ierr);
                fflush(stdout);
        }

        // Set the file view to handle distributed data layout
        ierr = MPI_File_set_view(infile, 0, oldtype, newtype, "native", MPI_INFO_NULL);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_set_view: %d\n", ierr);
                fflush(stdout);
        }

        // Perform collective write operation
        ierr = MPI_File_write_all(infile, matrix, size, oldtype, &mpistatus);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_write_all: %d\n", ierr);
                fflush(stdout);
        }

        // Close the file
        ierr = MPI_File_close(&infile);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_close: %d\n", ierr);
                fflush(stdout);
        }
}



/**
 * @brief Read single precision matrix from file using MPI I/O
 * 
 * Reads a single precision matrix from a file using MPI I/O for efficient
 * parallel file operations. The function uses custom MPI datatypes for
 * optimal data distribution and transfer patterns.
 * 
 * This function performs collective I/O operations where all processes
 * participate in reading their portion of the distributed matrix from a
 * single file. The file view is set up to handle the distributed data
 * layout efficiently.
 * 
 * @param[in] filename Input filename containing the matrix data
 * @param[in] mpirank Rank of the current MPI process
 * @param[out] matrix Single precision matrix buffer to read data into
 * @param[in] size Size of the matrix data in elements
 * @param[in] oldtype Original MPI datatype for the matrix elements
 * @param[in] newtype New MPI datatype optimized for distribution
 */
void MPI_Reading_sfile(char *filename, int mpirank, float *matrix, MPI_Count size, MPI_Datatype oldtype, MPI_Datatype newtype)
{
        int ierr;
        MPI_File infile;
        MPI_Status mpistatus;

        // Get information about the MPI datatype for validation
        MPI_Aint file_type_extent, lb;
        MPI_Count file_type_size;
        MPI_Type_get_extent(newtype, &lb, &file_type_extent);
        MPI_Type_size_x(newtype, &file_type_size);
        int64_t buffer_size = file_type_size / sizeof(float);

        // Validate that the buffer size matches the expected size
        if (buffer_size != size) {
                printf("\n On myrank:%d reading data from: %s parsec size:%lld not matching mpi local:%ld\n", 
                       mpirank, filename, size, buffer_size);
                fflush(stdout);
        }

        // Open the file for reading with collective I/O
        ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY,
                            MPI_INFO_NULL, &infile);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_open: %d\n", ierr);
                fflush(stdout);
        }

        // Set the file view to handle distributed data layout
        ierr = MPI_File_set_view(infile, 0, oldtype, newtype, "native", MPI_INFO_NULL);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_set_view: %d\n", ierr);
                fflush(stdout);
        }

        // Perform collective read operation
        ierr = MPI_File_read_all(infile, matrix, size, oldtype, MPI_STATUS_IGNORE);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_read_all: %d\n", ierr);
                fflush(stdout);
        }

        // Close the file
        ierr = MPI_File_close(&infile);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_close: %d\n", ierr);
                fflush(stdout);
        }
}

/**
 * @brief Read double precision matrix from file using MPI I/O
 * 
 * Reads a double precision matrix from a file using MPI I/O for efficient
 * parallel file operations. The function uses custom MPI datatypes for
 * optimal data distribution and transfer patterns.
 * 
 * This function performs collective I/O operations where all processes
 * participate in reading their portion of the distributed matrix from a
 * single file. The file view is set up to handle the distributed data
 * layout efficiently.
 * 
 * @param[in] filename Input filename containing the matrix data
 * @param[in] mpirank Rank of the current MPI process
 * @param[out] matrix Double precision matrix buffer to read data into
 * @param[in] size Size of the matrix data in elements
 * @param[in] oldtype Original MPI datatype for the matrix elements
 * @param[in] newtype New MPI datatype optimized for distribution
 */
void MPI_Reading_dfile(char *filename, int mpirank, double *matrix, MPI_Count size, MPI_Datatype oldtype, MPI_Datatype newtype)
{
        int ierr;
        MPI_File infile;
        MPI_Status mpistatus;

        // Get information about the MPI datatype for validation
        MPI_Aint file_type_extent, lb;
        MPI_Count file_type_size;
        MPI_Type_get_extent(newtype, &lb, &file_type_extent);
        MPI_Type_size_x(newtype, &file_type_size);
        int64_t buffer_size = file_type_size / sizeof(double);

        // Validate that the buffer size matches the expected size
        if (buffer_size != size) {
                printf("\n On myrank:%d reading data from: %s parsec size:%lld not matching mpi local:%ld\n", 
                       mpirank, filename, size, buffer_size);
                fflush(stdout);
        }

        // Open the file for reading with collective I/O
        ierr = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY,
                            MPI_INFO_NULL, &infile);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_open: %d\n", ierr);
                fflush(stdout);
        }

        // Set the file view to handle distributed data layout
        ierr = MPI_File_set_view(infile, 0, oldtype, newtype, "native", MPI_INFO_NULL);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_set_view: %d\n", ierr);
                fflush(stdout);
        }

        // Perform collective read operation
        ierr = MPI_File_read_all(infile, matrix, size, oldtype, MPI_STATUS_IGNORE);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_read_all: %d\n", ierr);
                fflush(stdout);
        }

        // Close the file
        ierr = MPI_File_close(&infile);
        if (ierr != 0) {
                printf("\n ====> Error in MPI_File_close: %d\n", ierr);
                fflush(stdout);
        }
}

/**
 * @brief Read matrix from file
 * 
 * Reads a single precision matrix from a file with specified dimensions.
 * The function assumes the file contains raw binary data in column-major format.
 * 
 * This function reads the matrix row by row from a binary file. It is designed
 * to handle large matrices efficiently by reading data in chunks rather than
 * loading the entire matrix into memory at once.
 * 
 * @param[in] filename Input filename containing the matrix data
 * @param[in] M Number of rows in the matrix
 * @param[in] N Number of columns in the matrix
 * @param[out] matrix Single precision matrix buffer to read data into
 * @return 1 on success, 0 on failure
 */
int read_file(char *filename, size_t M, size_t N, float *matrix)
{
        FILE *f;
        unsigned int i;

        // Open the file in binary read mode
        f = fopen(filename, "rb");
        if (f == NULL) {
                printf("Error: Unable to open file %s for reading\n", filename);
                return 0;
        }

        // Read the matrix row by row
        for (i = 0; i < M; i++) {
                // Seek to the beginning of row i
                // Each row has N elements, each element is 4 bytes (sizeof(float))
                fseek(f, N * i * 4, SEEK_SET);

                // Calculate the index for row-major storage
                unsigned long long index = i * N;

                // Read the entire row at once
                size_t elements_read = fread(&matrix[index], sizeof(float), N, f);
                if (elements_read != N) {
                        printf("Error: Expected to read %zu elements, but read %zu\n", N, elements_read);
                        fclose(f);
                        return 0;
                }

                // Optional progress reporting (currently disabled)
                if (0 && (i % 100000 == 0)) {
                        printf("\n %5.2f %% completed", (100. * (double)i) / ((double)M));
                }
        }

        // Close the file
        fclose(f);
        return 1;
}

/**
 * @brief Write double precision matrix to binary file
 * 
 * Writes a double precision matrix to a binary file in raw format.
 * The matrix is stored in column-major order for compatibility with
 * numerical computing libraries.
 * 
 * This function writes the matrix column by column to ensure proper
 * column-major storage format. The file contains only the raw matrix
 * data without any header information.
 * 
 * @param[in] filename Output filename for the binary matrix data
 * @param[in] M Number of rows in the matrix
 * @param[in] N Number of columns in the matrix
 * @param[in] matrix Double precision matrix data to write
 */
void writeMatrixToBinaryFile(const char *filename, size_t M, size_t N, double *matrix)
{
        FILE *file = fopen(filename, "wb"); // Open file for writing in binary mode
        if (file == NULL) {
                printf("Error: Unable to open file %s for writing\n", filename);
                return;
        }

        // Write matrix data column by column
        // This ensures column-major storage format
        for (size_t j = 0; j < N; j++) {
                size_t elements_written = fwrite(matrix + j * M, sizeof(double), M, file);
                if (elements_written != M) {
                        printf("Error: Expected to write %zu elements, but wrote %zu\n", M, elements_written);
                        fclose(file);
                        return;
                }
        }

        fclose(file);
        printf("Matrix written to binary file %s successfully.\n", filename);
}

/**
 * @brief Write single precision matrix to binary file
 * 
 * Writes a single precision matrix to a binary file in raw format.
 * The matrix is stored in column-major order for compatibility with
 * numerical computing libraries.
 * 
 * This function writes the matrix column by column to ensure proper
 * column-major storage format. The file contains only the raw matrix
 * data without any header information.
 * 
 * @param[in] filename Output filename for the binary matrix data
 * @param[in] M Number of rows in the matrix
 * @param[in] N Number of columns in the matrix
 * @param[in] matrix Single precision matrix data to write
 */
void writeMatrixToBinaryFileSingle(const char *filename, size_t M, size_t N, float *matrix)
{
        FILE *file = fopen(filename, "wb"); // Open file for writing in binary mode
        if (file == NULL) {
                printf("Error: Unable to open file %s for writing\n", filename);
                return;
        }

        // Write matrix data column by column
        // This ensures column-major storage format
        for (size_t j = 0; j < N; j++) {
                size_t elements_written = fwrite(matrix + j * M, sizeof(float), M, file);
                if (elements_written != M) {
                        printf("Error: Expected to write %zu elements, but wrote %zu\n", M, elements_written);
                        fclose(file);
                        return;
                }
        }

        fclose(file);
        printf("Matrix written to binary file %s successfully.\n", filename);
}
