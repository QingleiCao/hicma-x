###
#
# @copyright (c) 2009-2014 The University of Tennessee and The University
#                          of Tennessee Research Foundation.
#                          All rights reserved.
# @copyright (c) 2012-2014 Inria. All rights reserved.
# @copyright (c) 2012-2014 Bordeaux INP, CNRS (LaBRI UMR 5800), Inria, Univ. Bordeaux. All rights reserved.
# @copyright (c) 2017      King Abdullah University of Science and Technology (KAUST). All rights reserved.
#
###
#
# - Find HCORE include dirs and libraries
# Use this module by invoking find_package with the form:
#  find_package(HCORE
#               [REQUIRED]             # Fail with error if hcore is not found
#               [COMPONENTS <comp1> <comp2> ...] # dependencies
#              )
#
#  HCORE depends on the following libraries:
#   - LAPACK
#   - LAPACKE
#   - BLAS
#   - CBLAS
#
#  COMPONENTS are optional libraries HCORE could be linked with,
#  Use it to drive detection of a specific compilation chain
#  COMPONENTS can be some of the following:
#   - no components are available for now: maybe HCORE in the future?
#
# Results are reported in variables:
#  HCORE_FOUND            - True if headers and requested libraries were found
#  HCORE_LINKER_FLAGS     - list of required linker flags (excluding -l and -L)
#  HCORE_INCLUDE_DIRS     - hcore include directories
#  HCORE_LIBRARY_DIRS     - Link directories for hcore libraries
#  HCORE_LIBRARIES        - hcore libraries
#  HCORE_INCLUDE_DIRS_DEP - hcore + dependencies include directories
#  HCORE_LIBRARY_DIRS_DEP - hcore + dependencies link directories
#  HCORE_LIBRARIES_DEP    - hcore libraries + dependencies
#
# The user can give specific paths where to find the libraries adding cmake
# options at configure (ex: cmake path/to/project -DHCORE_DIR=path/to/hcore):
#  HCORE_DIR              - Where to find the base directory of hcore
#  HCORE_INCDIR           - Where to find the header files
#  HCORE_LIBDIR           - Where to find the library files
# The module can also look for the following environment variables if paths
# are not given as cmake variable: HCORE_DIR, HCORE_INCDIR, HCORE_LIBDIR
#
#=============================================================================
# Copyright 2012-2013 Inria
# Copyright 2012-2013 Emmanuel Agullo
# Copyright 2012-2013 Mathieu Faverge
# Copyright 2012      Cedric Castagnede
# Copyright 2013      Florent Pruvost
# Copyright 2017      Eduardo Gonzalez
#
# Distributed under the OSI-approved BSD License (the "License");
# see accompanying file MORSE-Copyright.txt for details.
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the License for more information.
#=============================================================================
# (To distribute this file outside of Morse, substitute the full
#  License text for the above reference.)


if(NOT HCORE_FOUND)
    set(HCORE_DIR "" CACHE PATH "Installation directory of HCORE library")
    if (NOT HCORE_FIND_QUIETLY)
        message(STATUS "A cache variable, namely HCORE_DIR, has been set to specify the install directory of HCORE")
    endif()
endif(NOT HCORE_FOUND)

# HCORE depends on LAPACKE anyway, try to find it
if (NOT LAPACKE_FOUND)
    if(HCORE_FIND_REQUIRED)
        find_package(LAPACKE REQUIRED)
    else()
        find_package(LAPACKE)
    endif()
endif()
# HCORE depends on CBLAS anyway, try to find it
if (NOT CBLAS_FOUND)
    if(HCORE_FIND_REQUIRED)
        find_package(CBLAS REQUIRED)
    else()
        find_package(CBLAS)
    endif()
endif()
# BLAS and LAPACK are searched by CBLAS and LAPACKE


set(ENV_HCORE_DIR "$ENV{HCORE_DIR}")
set(ENV_HCORE_INCDIR "$ENV{HCORE_INCDIR}")
set(ENV_HCORE_LIBDIR "$ENV{HCORE_LIBDIR}")
set(HCORE_GIVEN_BY_USER "FALSE")
if ( HCORE_DIR OR ( HCORE_INCDIR AND HCORE_LIBDIR) OR ENV_HCORE_DIR OR (ENV_HCORE_INCDIR AND ENV_HCORE_LIBDIR) )
    set(HCORE_GIVEN_BY_USER "TRUE")
endif()

# Optionally use pkg-config to detect include/library dirs (if pkg-config is available)
# -------------------------------------------------------------------------------------
include(FindPkgConfig)
find_package(PkgConfig QUIET)
if(PKG_CONFIG_EXECUTABLE AND NOT HCORE_GIVEN_BY_USER)

    pkg_search_module(HCORE hcore)
    if (NOT HCORE_FIND_QUIETLY)
        if (HCORE_FOUND AND HCORE_LIBRARIES)
            message(STATUS "Looking for HCORE - found using PkgConfig")
            #if(NOT HCORE_INCLUDE_DIRS)
            #    message("${Magenta}HCORE_INCLUDE_DIRS is empty using PkgConfig."
            #        "Perhaps the path to hcore headers is already present in your"
            #        "C(PLUS)_INCLUDE_PATH environment variable.${ColourReset}")
            #endif()
        else()
            message("${Magenta}Looking for HCORE - not found using PkgConfig. "
                "Perhaps you should add the directory containing hcore.pc "
                "to the PKG_CONFIG_PATH environment variable.${ColourReset}")
        endif()
    endif()

    if (HCORE_FIND_VERSION_EXACT)
        if( NOT (HCORE_FIND_VERSION_MAJOR STREQUAL HCORE_VERSION_MAJOR) OR
            NOT (HCORE_FIND_VERSION_MINOR STREQUAL HCORE_VERSION_MINOR) )
            if(NOT HCORE_FIND_QUIETLY)
                message(FATAL_ERROR
                        "HCORE version found is ${HCORE_VERSION_STRING} "
                        "when required is ${HCORE_FIND_VERSION}")
            endif()
        endif()
    else()
        # if the version found is older than the required then error
        if( (HCORE_FIND_VERSION_MAJOR STRGREATER HCORE_VERSION_MAJOR) OR
            (HCORE_FIND_VERSION_MINOR STRGREATER HCORE_VERSION_MINOR) )
            if(NOT HCORE_FIND_QUIETLY)
                message(FATAL_ERROR
                        "HCORE version found is ${HCORE_VERSION_STRING} "
                        "when required is ${HCORE_FIND_VERSION} or newer")
            endif()
        endif()
    endif()

    # if pkg-config is used: these variables are empty
    # the pkg_search_module call will set the following:
    # HCORE_LDFLAGS: all required linker flags
    # HCORE_CFLAGS:  all required cflags
    set(HCORE_INCLUDE_DIRS_DEP "")
    set(HCORE_LIBRARY_DIRS_DEP "")
    set(HCORE_LIBRARIES_DEP "")
    # replace it anyway: we should update it with dependencies given by pkg-config
    set(HCORE_INCLUDE_DIRS_DEP "${HCORE_INCLUDE_DIRS}")
    set(HCORE_LIBRARY_DIRS_DEP "${HCORE_LIBRARY_DIRS}")
    set(HCORE_LIBRARIES_DEP "${HCORE_LIBRARIES}")

endif(PKG_CONFIG_EXECUTABLE AND NOT HCORE_GIVEN_BY_USER)

# if HCORE is not found using pkg-config
if( (NOT PKG_CONFIG_EXECUTABLE) OR (PKG_CONFIG_EXECUTABLE AND NOT HCORE_FOUND) OR (HCORE_GIVEN_BY_USER) )

    if (NOT HCORE_FIND_QUIETLY)
        message(STATUS "Looking for HCORE - PkgConfig not used")
    endif()

    # Looking for include
    # -------------------

    # Add system include paths to search include
    # ------------------------------------------
    unset(_inc_env)
    set(ENV_HCORE_DIR "$ENV{HCORE_DIR}")
    set(ENV_HCORE_INCDIR "$ENV{HCORE_INCDIR}")
    if(ENV_HCORE_INCDIR)
        list(APPEND _inc_env "${ENV_HCORE_INCDIR}")
    elseif(ENV_HCORE_DIR)
        list(APPEND _inc_env "${ENV_HCORE_DIR}")
        list(APPEND _inc_env "${ENV_HCORE_DIR}/include")
        list(APPEND _inc_env "${ENV_HCORE_DIR}/include/hcore")
    else()
        if(WIN32)
            string(REPLACE ":" ";" _inc_env "$ENV{INCLUDE}")
        else()
            string(REPLACE ":" ";" _path_env "$ENV{INCLUDE}")
            list(APPEND _inc_env "${_path_env}")
            string(REPLACE ":" ";" _path_env "$ENV{C_INCLUDE_PATH}")
            list(APPEND _inc_env "${_path_env}")
            string(REPLACE ":" ";" _path_env "$ENV{CPATH}")
            list(APPEND _inc_env "${_path_env}")
            string(REPLACE ":" ";" _path_env "$ENV{INCLUDE_PATH}")
            list(APPEND _inc_env "${_path_env}")
        endif()
    endif()
    list(APPEND _inc_env "${CMAKE_PLATFORM_IMPLICIT_INCLUDE_DIRECTORIES}")
    list(APPEND _inc_env "${CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES}")
    list(REMOVE_DUPLICATES _inc_env)


    # Try to find the hcore header in the given paths
    # -------------------------------------------------
    # call cmake macro to find the header path
    if(HCORE_INCDIR)
        set(HCORE_hcore.h_DIRS "HCORE_hcore.h_DIRS-NOTFOUND")
        find_path(HCORE_hcore.h_DIRS
          NAMES hcore.h
          HINTS ${HCORE_INCDIR})
    else()
        if(HCORE_DIR)
            set(HCORE_hcore.h_DIRS "HCORE_hcore.h_DIRS-NOTFOUND")
            find_path(HCORE_hcore.h_DIRS
              NAMES hcore.h
              HINTS ${HCORE_DIR}
              PATH_SUFFIXES "include" "include/hcore")
        else()
            set(HCORE_hcore.h_DIRS "HCORE_hcore.h_DIRS-NOTFOUND")
            find_path(HCORE_hcore.h_DIRS
              NAMES hcore.h
              HINTS ${_inc_env})
        endif()
    endif()
    mark_as_advanced(HCORE_hcore.h_DIRS)

    # If found, add path to cmake variable
    # ------------------------------------
    if (HCORE_hcore.h_DIRS)
        set(HCORE_INCLUDE_DIRS "${HCORE_hcore.h_DIRS}")
    else ()
        set(HCORE_INCLUDE_DIRS "HCORE_INCLUDE_DIRS-NOTFOUND")
        if(NOT HCORE_FIND_QUIETLY)
            message(STATUS "Looking for hcore -- hcore.h not found")
        endif()
    endif()


    # Looking for lib
    # ---------------

    # Add system library paths to search lib
    # --------------------------------------
    unset(_lib_env)
    set(ENV_HCORE_LIBDIR "$ENV{HCORE_LIBDIR}")
    if(ENV_HCORE_LIBDIR)
        list(APPEND _lib_env "${ENV_HCORE_LIBDIR}")
    elseif(ENV_HCORE_DIR)
        list(APPEND _lib_env "${ENV_HCORE_DIR}")
        list(APPEND _lib_env "${ENV_HCORE_DIR}/lib")
    else()
        if(WIN32)
            string(REPLACE ":" ";" _lib_env "$ENV{LIB}")
        else()
            if(APPLE)
                string(REPLACE ":" ";" _lib_env "$ENV{DYLD_LIBRARY_PATH}")
            else()
                string(REPLACE ":" ";" _lib_env "$ENV{LD_LIBRARY_PATH}")
            endif()
            list(APPEND _lib_env "${CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES}")
            list(APPEND _lib_env "${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}")
        endif()
    endif()
    list(REMOVE_DUPLICATES _lib_env)

    # Try to find the hcore lib in the given paths
    # ----------------------------------------------

    # call cmake macro to find the lib path
    if(HCORE_LIBDIR)
        set(HCORE_hcore_LIBRARY "HCORE_hcore_LIBRARY-NOTFOUND")
        find_library(HCORE_hcore_LIBRARY
            NAMES hcore
            HINTS ${HCORE_LIBDIR})
    else()
        if(HCORE_DIR)
            set(HCORE_hcore_LIBRARY "HCORE_hcore_LIBRARY-NOTFOUND")
            find_library(HCORE_hcore_LIBRARY
                NAMES hcore
                HINTS ${HCORE_DIR}
                PATH_SUFFIXES lib lib32 lib64)
        else()
            set(HCORE_hcore_LIBRARY "HCORE_hcore_LIBRARY-NOTFOUND")
            find_library(HCORE_hcore_LIBRARY
                NAMES hcore
                HINTS ${_lib_env})
        endif()
    endif()
    mark_as_advanced(HCORE_hcore_LIBRARY)

    # If found, add path to cmake variable
    # ------------------------------------
    if (HCORE_hcore_LIBRARY)
        get_filename_component(hcore_lib_path "${HCORE_hcore_LIBRARY}" PATH)
        # set cmake variables
        set(HCORE_LIBRARIES    "${HCORE_hcore_LIBRARY}")
        set(HCORE_LIBRARY_DIRS "${hcore_lib_path}")
    else ()
        set(HCORE_LIBRARIES    "HCORE_LIBRARIES-NOTFOUND")
        set(HCORE_LIBRARY_DIRS "HCORE_LIBRARY_DIRS-NOTFOUND")
        if(NOT HCORE_FIND_QUIETLY)
            message(STATUS "Looking for hcore -- lib hcore not found")
        endif()
    endif ()

    # check a function to validate the find
    if (HCORE_LIBRARIES)

        set(REQUIRED_LDFLAGS)
        set(REQUIRED_INCDIRS)
        set(REQUIRED_LIBDIRS)
        set(REQUIRED_LIBS)

        # HCORE
        if (HCORE_INCLUDE_DIRS)
            set(REQUIRED_INCDIRS "${HCORE_INCLUDE_DIRS}")
        endif()
        if (HCORE_LIBRARY_DIRS)
            set(REQUIRED_LIBDIRS "${HCORE_LIBRARY_DIRS}")
        endif()
        set(REQUIRED_LIBS "${HCORE_LIBRARIES}")
        # CBLAS
        if (CBLAS_INCLUDE_DIRS_DEP)
            list(APPEND REQUIRED_INCDIRS "${CBLAS_INCLUDE_DIRS_DEP}")
        elseif (CBLAS_INCLUDE_DIRS)
            list(APPEND REQUIRED_INCDIRS "${CBLAS_INCLUDE_DIRS}")
        endif()
        if(CBLAS_LIBRARY_DIRS_DEP)
            list(APPEND REQUIRED_LIBDIRS "${CBLAS_LIBRARY_DIRS_DEP}")
        elseif(CBLAS_LIBRARY_DIRS)
            list(APPEND REQUIRED_LIBDIRS "${CBLAS_LIBRARY_DIRS}")
        endif()
        if (CBLAS_LIBRARIES_DEP)
            list(APPEND REQUIRED_LIBS "${CBLAS_LIBRARIES_DEP}")
        elseif(CBLAS_LIBRARIES)
            list(APPEND REQUIRED_LIBS "${CBLAS_LIBRARIES}")
        endif()
        if (BLAS_LINKER_FLAGS)
            list(APPEND REQUIRED_LDFLAGS "${BLAS_LINKER_FLAGS}")
        endif()
        # LAPACK
        if (LAPACK_INCLUDE_DIRS)
            list(APPEND REQUIRED_INCDIRS "${LAPACK_INCLUDE_DIRS}")
        endif()
        if(LAPACK_LIBRARY_DIRS)
            list(APPEND REQUIRED_LIBDIRS "${LAPACK_LIBRARY_DIRS}")
        endif()
        list(APPEND REQUIRED_LIBS "${LAPACK_LIBRARIES}")
        if (LAPACK_LINKER_FLAGS)
            list(APPEND REQUIRED_LDFLAGS "${LAPACK_LINKER_FLAGS}")
        endif()

        # set required libraries for link
        set(CMAKE_REQUIRED_INCLUDES "${REQUIRED_INCDIRS}")
        set(CMAKE_REQUIRED_LIBRARIES)
        list(APPEND CMAKE_REQUIRED_LIBRARIES "${REQUIRED_LDFLAGS}")
        foreach(lib_dir ${REQUIRED_LIBDIRS})
            list(APPEND CMAKE_REQUIRED_LIBRARIES "-L${lib_dir}")
        endforeach()
        list(APPEND CMAKE_REQUIRED_LIBRARIES "${REQUIRED_LIBS}")
        string(REGEX REPLACE "^ -" "-" CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES}")

        # test link
        unset(HCORE_WORKS CACHE)
        include(CheckFunctionExists)
        check_function_exists(hcore_dgetrf HCORE_WORKS)
        mark_as_advanced(HCORE_WORKS)

        if(HCORE_WORKS)
            # save link with dependencies
            set(HCORE_LIBRARIES_DEP    "${REQUIRED_LIBS}")
            set(HCORE_LIBRARY_DIRS_DEP "${REQUIRED_LIBDIRS}")
            set(HCORE_INCLUDE_DIRS_DEP "${REQUIRED_INCDIRS}")
            set(HCORE_LINKER_FLAGS     "${REQUIRED_LDFLAGS}")
            list(REMOVE_DUPLICATES HCORE_LIBRARY_DIRS_DEP)
            list(REMOVE_DUPLICATES HCORE_INCLUDE_DIRS_DEP)
            list(REMOVE_DUPLICATES HCORE_LINKER_FLAGS)
        else()
            if(NOT HCORE_FIND_QUIETLY)
                message(STATUS "Looking for hcore : test of hcore_dgetrf with
                hcore, cblas, and lapack libraries fails")
                message(STATUS "CMAKE_REQUIRED_LIBRARIES: ${CMAKE_REQUIRED_LIBRARIES}")
                message(STATUS "CMAKE_REQUIRED_INCLUDES: ${CMAKE_REQUIRED_INCLUDES}")
                message(STATUS "Check in CMakeFiles/CMakeError.log to figure out why it fails")
            endif()
        endif()
        set(CMAKE_REQUIRED_INCLUDES)
        set(CMAKE_REQUIRED_FLAGS)
        set(CMAKE_REQUIRED_LIBRARIES)
    endif(HCORE_LIBRARIES)

endif( (NOT PKG_CONFIG_EXECUTABLE) OR (PKG_CONFIG_EXECUTABLE AND NOT HCORE_FOUND) OR (HCORE_GIVEN_BY_USER) )

if (HCORE_LIBRARIES)
    if (HCORE_LIBRARY_DIRS)
        set( first_lib_path "" )
        foreach(dir ${HCORE_LIBRARY_DIRS})
            if ("${dir}" MATCHES "hcore")
                set(first_lib_path "${dir}")
            endif()
        endforeach()
        if( NOT first_lib_path )
            list(GET HCORE_LIBRARY_DIRS 0 first_lib_path)
        endif()
    else()
        list(GET HCORE_LIBRARIES 0 first_lib)
        get_filename_component(first_lib_path "${first_lib}" PATH)
    endif()
    if (${first_lib_path} MATCHES "/lib(32|64)?$")
        string(REGEX REPLACE "/lib(32|64)?$" "" not_cached_dir "${first_lib_path}")
        set(HCORE_DIR_FOUND "${not_cached_dir}" CACHE PATH "Installation directory of HCORE library" FORCE)
    else()
        set(HCORE_DIR_FOUND "${first_lib_path}" CACHE PATH "Installation directory of HCORE library" FORCE)
    endif()
endif()

# check that HCORE has been found
# -------------------------------
include(FindPackageHandleStandardArgs)
if (PKG_CONFIG_EXECUTABLE AND HCORE_FOUND)
    find_package_handle_standard_args(HCORE DEFAULT_MSG
                                      HCORE_LIBRARIES)
else()
    find_package_handle_standard_args(HCORE DEFAULT_MSG
                                      HCORE_LIBRARIES
                                      HCORE_WORKS)
endif()
