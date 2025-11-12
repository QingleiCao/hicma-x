/**
 * @copyright (c) 2023-2025     Saint Louis University (SLU)
 * @copyright (c) 2023-2025     Massachusetts Institute of Technology (MIT)
 * @copyright (c) 2023-2025     Nvidia Corporation
 * @copyright (c) 2018-2025     King Abdullah University of Science and Technology (KAUST)
 * @copyright (c) 2023-2025     The University of Tennessee and The University of Tennessee Research Foundation
 *                              All rights reserved.
 **/

#include "hicma_parsec_flat_file.h"

/**
 * @file hicma_parsec_flat_file.c
 * @brief Implementation of flat file operations for HICMA PaRSEC
 *
 * This file implements utility functions for reading various types of data files
 * used in HICMA PaRSEC applications. The functions support reading spatial
 * coordinates (2D and 3D), observational data, and temporal data from text files.
 * 
 * The implementation provides:
 * - File line counting for data validation
 * - CSV parsing for coordinate data
 * - Memory management for dynamic data structures
 * - Error handling for file operations
 */

/* ============================================================================
 * File utility functions
 * ============================================================================ */

/**
 * @brief Count the number of lines in a file
 *
 * Counts the total number of lines in a text file by reading character by character
 * and counting newline characters. This function is useful for determining the size
 * of data files before reading them and for validating file integrity.
 *
 * @param[in] filename Name of the file to count lines in
 * @return Number of lines in the file, or 0 on error
 * @note This function is designed for text files and may not work correctly
 *       with binary files or files with different line endings.
 */
int countlines(char *filename)
{
    FILE *fp = fopen(filename,"r");
    int ch = 0;
    int lines = 0;

    if (fp == NULL)
    {
        fprintf(stderr,"cannot open locations file\n");
        return 0;
    }

    while(!feof(fp))
    {
        ch = fgetc(fp);
        if(ch == '\n')
            lines++;
    }

    fclose(fp);

    //Excluding header line
    return (lines);
}

/* ============================================================================
 * Location file reading functions
 * ============================================================================ */

/**
 * @brief Read 2D location file
 *
 * Reads a file containing 2D location coordinates and returns a location structure.
 * The file is expected to contain comma-separated values with X and Y coordinates
 * on each line. Memory is dynamically allocated for the coordinate arrays.
 *
 * @param[in] locs_file Filename of the location file
 * @param[in] n Number of locations to read
 * @return Pointer to location structure with x and y arrays populated, or NULL on error
 * @note The caller is responsible for freeing the returned structure and its arrays
 * @warning The function assumes the file contains at least 'n' lines of valid data
 */
location* readLocsFile(char *locs_file, int n)
{
	FILE *fp;
	int i = 0;
	char *line = NULL;
	size_t len  = 0;
	ssize_t read;
	char *pch;        
	location *locations; 

	fp = fopen(locs_file, "r");
	if (fp == NULL)
	{
		printf("cannot read locations file\n");
		printf("%s: \n",locs_file);
		return NULL;
	}
	else
	{
		//Allocate memory
		locations		= (location *) malloc(sizeof(location*));
		locations->x            = (double *) malloc(n * sizeof(double));
		locations->y            = (double *) malloc(n * sizeof(double));
	}

	while ((read = getline(&line, &len, fp)) != -1) {
		pch = strtok(line, ",");
		while (pch != NULL)
		{
			locations->x[i] = atof(pch);
			pch = strtok (NULL, ",");
			locations->y[i] = atof(pch);
			pch = strtok (NULL, ",");
		}
		i++;
	}
	fclose(fp);
	if (line)
		free(line);
	//zsort_locations(n,locations);
	return locations;
}

/**
 * @brief Read 3D location file
 *
 * Reads a file containing 3D location coordinates and returns a location structure.
 * The file is expected to contain comma-separated values with X, Y, and Z coordinates
 * on each line. Memory is dynamically allocated for the coordinate arrays.
 *
 * @param[in] locs_file Filename of the location file
 * @param[in] n Number of locations to read
 * @return Pointer to location structure with x, y, and z arrays populated, or NULL on error
 * @note The caller is responsible for freeing the returned structure and its arrays
 * @warning The function assumes the file contains at least 'n' lines of valid data
 */
location* readLocsFile3d(char* locs_file, int n)
{
	FILE *fp;
	int i = 0;
	char *line = NULL;
	size_t len  = 0;
	ssize_t read;
	char *pch;
	location *locations;

	fp = fopen(locs_file, "r");
	if (fp == NULL)
	{
		printf("cannot open location file%s\n", locs_file);
		exit(EXIT_FAILURE);
	}
	else
	{
		//Allocate memory
		locations               = (location *) malloc(sizeof(location));
		locations->x            = (double *) malloc(n * sizeof(double));
		locations->y            = (double *) malloc(n * sizeof(double));
		locations->z            = (double *) malloc(n * sizeof(double));      
	}

	while ((read = getline(&line, &len, fp)) != -1) {
		pch = strtok(line, ",");
		while (pch != NULL)
		{
			locations->x[i] = atof(pch);
			pch = strtok (NULL, ",");
			locations->y[i] = atof(pch);
			pch = strtok (NULL, ",");
			locations->z[i] = atof(pch);
			pch = strtok (NULL, ",");
		}
		i++;
	}
	fclose(fp);
	if (line)
		free(line);
	//zsort_locations(n,locations);
	return locations;
}

/* ============================================================================
 * Data file reading functions
 * ============================================================================ */

/**
 * @brief Read observation file
 *
 * Reads a file containing observation data and returns an array of values.
 * The file is expected to contain one observation value per line. Memory
 * is dynamically allocated for the observation array.
 *
 * @param[in] obsfile Filename of the observation file
 * @param[in] n Number of observations to read
 * @return Pointer to array of observation values, or NULL on error
 * @note The caller is responsible for freeing the returned array
 * @warning The function assumes the file contains at least 'n' lines of valid data
 */
double* readObsFile(char* obsfile, int n)
{
    FILE* fp;
    char* line     = NULL;
    size_t len      = 0;
    ssize_t read;
    int count       = 0;

    double* z_vec = (double *) malloc(n * sizeof(double));

	fp = fopen(obsfile, "r");
	if (fp == NULL)
	{
		printf("readObsFile:cannot open observations file: %s\n", obsfile);
		exit(EXIT_FAILURE);
	}

	while ((read = getline(&line, &len, fp)) != -1)
		z_vec[count++]=atof(line);

	fclose(fp);
	free(line);

	return z_vec;
}

/**
 * @brief Read time file
 *
 * Reads a file containing time data and returns an array of values.
 * The file is expected to contain one time value per line. Memory
 * is dynamically allocated for the time array. This function is
 * particularly useful for space-time modeling applications.
 *
 * @param[in] timefile Filename of the time file
 * @param[in] n Number of time points to read
 * @return Pointer to array of time values, or NULL on error
 * @note The caller is responsible for freeing the returned array
 * @warning The function assumes the file contains at least 'n' lines of valid data
 */
double* readTimeFile(char* timefile, int n)
{
    FILE* fp;
    char* line     = NULL;
    size_t len      = 0;
    ssize_t read;
    int count       = 0;

    double* time_vec = (double *) malloc(n * sizeof(double));

    fp = fopen(timefile, "r");
    if (fp == NULL)
    {
        printf("readTimeFile:cannot open time file: %s\n", timefile);
        exit(EXIT_FAILURE);
    }

    while ((read = getline(&line, &len, fp)) != -1)
        time_vec[count++]=atof(line);

    fclose(fp);
    free(line);

    return time_vec;
}

