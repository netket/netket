#.rst:
# FindLAPACKE
# -------------
#
# Find LAPACKE include dirs and libraries
#
# Use this module by invoking find_package with the form::
#
#   find_package(LAPACKE
#     [REQUIRED]             # Fail with error if LAPACKE is not found
#     [COMPONENTS <libs>...] # List of libraries to look for
#     )
#
# Valid names for COMPONENTS libraries are::
#
#   ALL                      - Find all libraries
#   LAPACKE_H                - Find the lapacke.h header file
#   LAPACKE                  - Find a LAPACKE library
#   LAPACK                   - Find a LAPACK library
#   CBLAS                    - Find a CBLAS library
#   BLAS                     - Find a BLAS library
#
#  Not specifying COMPONENTS is identical to choosing ALL
#
# This module defines::
#
#   LAPACKE_FOUND            - True if headers and requested libraries were found
#   LAPACKE_INCLUDE_DIRS     - LAPACKE include directories
#   LAPACKE_LIBRARIES        - LAPACKE component libraries to be linked
#
#
# This module reads hints about search locations from variables
# (either CMake variables or environment variables)::
#
#   LAPACKE_ROOT             - Preferred installation prefix for LAPACKE
#   LAPACKE_DIR              - Preferred installation prefix for LAPACKE
#
#
# The following :prop_tgt:`IMPORTED` targets are also defined::
#
#   LAPACKE::LAPACKE         - Imported target for the LAPACKE library
#   LAPACKE::LAPACK          - Imported target for the LAPACK library
#   LAPACKE::CBLAS           - Imported target for the CBLAS library
#   LAPACKE::BLAS            - Imported target for the BLAS library
#

# ==============================================================================
# Copyright 2018 Damien Nguyen <damien.nguyen@alumni.epfl.ch>
#
# Distributed under the OSI-approved BSD License (the "License")
#
# This software is distributed WITHOUT ANY WARRANTY; without even the
# implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ==============================================================================

set(LAPACKE_SEARCH_PATHS
  ${LAPACKE_ROOT}
  $ENV{LAPACKE_ROOT}
  ${LAPACKE_DIR}
  $ENV{LAPACKE_DIR}
  ${CMAKE_PREFIX_PATH}
  $ENV{CMAKE_PREFIX_PATH}
  /usr
  /usr/local/
  /usr/local/opt # homebrew on mac
  /opt
  /opt/local
  /opt/LAPACKE
  )

set(LIB_PATH_SUFFIXES
  lib64
  lib
  lib/x86_64-linux-gnu
  lib32
  )

set(INC_PATH_SUFFIXES
  include
  include/lapack
  include/lapacke/
  lapack/include
  lapacke/include
  )

if(APPLE)
  list(APPEND LIB_PATH_SUFFIXES lapack/lib openblas/lib)
elseif(WIN32)
  list(APPEND LAPACKE_SEARCH_PATHS "C:/Program Files (x86)/LAPACK")
  list(APPEND LAPACKE_SEARCH_PATHS "C:/Program Files/LAPACK")
endif()

# ==============================================================================
# Prepare some helper variables

set(LAPACKE_INCLUDE_DIRS)
set(LAPACKE_LIBRARIES)
set(LAPACKE_REQUIRED_VARS)
set(LAPACKE_FIND_ALL_COMPONENTS 0)

# ==============================================================================

macro(_find_library_with_header component incname)
  find_library(LAPACKE_${component}_LIB
    NAMES ${ARGN}
    NAMES_PER_DIR
    PATHS ${LAPACKE_SEARCH_PATHS}
    PATH_SUFFIXES ${LIB_PATH_SUFFIXES})
  if(LAPACKE_${component}_LIB)    
    set(LAPACKE_${component}_LIB_FOUND 1)
  endif()
  list(APPEND LAPACKE_REQUIRED_VARS "LAPACKE_${component}_LIB")

  # If necessary, look for the header file as well
  if(NOT "${incname}" STREQUAL "")
    find_path(LAPACKE_${component}_INCLUDE_DIR
      NAMES ${incname}
      PATHS ${LAPACKE_SEARCH_PATHS}
      PATH_SUFFIXES ${INC_PATH_SUFFIXES})
    list(APPEND LAPACKE_REQUIRED_VARS "LAPACKE_${component}_INCLUDE_DIR")
    if(LAPACKE_${component}_LIB)
      set(LAPACKE_${component}_INC_FOUND 1)
    endif()
  else()
    set(LAPACKE_${component}_INC_FOUND 1)
  endif()
  
  if(LAPACKE_${component}_LIB_FOUND AND LAPACKE_${component}_INC_FOUND)
    set(LAPACKE_${component}_FOUND 1)
  else()
    set(LAPACKE_${component}_FOUND 0)
  endif()
endmacro()

# ------------------------------------------------------------------------------

if(NOT LAPACKE_FIND_COMPONENTS OR LAPACKE_FIND_COMPONENTS STREQUAL "ALL")
  set(LAPACKE_FIND_ALL_COMPONENTS 1)
  set(LAPACKE_FIND_COMPONENTS "LAPACKE;LAPACK;CBLAS;BLAS")
endif(NOT LAPACKE_FIND_COMPONENTS OR LAPACKE_FIND_COMPONENTS STREQUAL "ALL")

# Make sure that all components are in capitals
set(_tmp_component_list)
foreach(_comp ${LAPACKE_FIND_COMPONENTS})
  string(TOUPPER ${_comp} _comp)
  list(APPEND _tmp_component_list ${_comp})
endforeach()
set(LAPACKE_FIND_COMPONENTS ${_tmp_component_list})
set(_tmp_component_list)

foreach(_comp ${LAPACKE_FIND_COMPONENTS})
  if(_comp STREQUAL "LAPACKE")
    _find_library_with_header(${_comp} lapacke.h lapacke liblapacke)
  elseif(_comp STREQUAL "LAPACKE_H")
    find_path(LAPACKE_${_comp}_INCLUDE_DIR
      NAMES lapacke.h
      PATHS ${LAPACKE_SEARCH_PATHS}
      PATH_SUFFIXES include lapack/include)
    list(APPEND LAPACKE_REQUIRED_VARS "LAPACKE_${_comp}_INCLUDE_DIR")
    if(LAPACKE_${_comp}_LIB)
      set(LAPACKE_${_comp}_INC_FOUND 1)
    endif()
  elseif(_comp STREQUAL "LAPACK")
    _find_library_with_header(${_comp} "" lapack liblapack)
  elseif(_comp STREQUAL "CBLAS")
    _find_library_with_header(${_comp} cblas.h cblas libcblas)
  elseif(_comp STREQUAL "BLAS")
    _find_library_with_header(${_comp} "" blas blas)
  else()
    message(FATAL_ERROR "Unknown component: ${_comp}")
  endif()
  mark_as_advanced(
    LAPACKE_${_comp}_LIB
    LAPACKE_${_comp}_INCLUDE_DIR)
endforeach()

# ==============================================================================

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACKE
  FOUND_VAR LAPACKE_FOUND
  REQUIRED_VARS ${LAPACKE_REQUIRED_VARS}
  HANDLE_COMPONENTS)

# ==============================================================================

if(LAPACKE_FOUND)
  foreach(_comp ${LAPACKE_FIND_COMPONENTS})
    list(APPEND LAPACKE_INCLUDE_DIRS ${LAPACKE_${_comp}_INCLUDE_DIR})
    list(APPEND LAPACKE_LIBRARIES ${LAPACKE_${_comp}_LIB})
  endforeach()
  
  if("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR
      "${CMAKE_C_COMPILER_ID}" MATCHES ".*GNU.*" OR
      "${CMAKE_C_COMPILER_ID}" MATCHES ".*Intel.*"
      ) #NOT MSVC
    set(MATH_LIB "m")
    list(APPEND LAPACKE_LIBRARIES m)
  endif()
  
  if(NOT "${LAPACKE_INCLUDE_DIRS}" STREQUAL "")
    list(REMOVE_DUPLICATES LAPACKE_INCLUDE_DIRS)
  endif()

  # ----------------------------------------------------------------------------

  # Inspired by FindBoost.cmake
  foreach(_comp ${LAPACKE_FIND_COMPONENTS})
    if(NOT TARGET LAPACKE::${_comp} AND LAPACKE_${_comp}_FOUND)
      get_filename_component(LIB_EXT "${LAPACKE_${_comp}_LIB}" EXT)
      if(LIB_EXT STREQUAL ".a" OR LIB_EXT STREQUAL ".lib")
        set(LIB_TYPE STATIC)
      else()
        set(LIB_TYPE SHARED)
      endif()
      add_library(LAPACKE::${_comp} ${LIB_TYPE} IMPORTED GLOBAL)
      if(LAPACKE_INCLUDE_DIRS)
        set_target_properties(LAPACKE::${_comp} PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${LAPACKE_INCLUDE_DIRS}")
      endif()
      if(EXISTS "${LAPACKE_${_comp}_LIB}")
        set_target_properties(LAPACKE::${_comp} PROPERTIES
          IMPORTED_LOCATION "${LAPACKE_${_comp}_LIB}")
      endif()
      set_target_properties(LAPACKE::${_comp} PROPERTIES
        INTERFACE_LINK_LIBRARIES "${MATH_LIB}")
    endif()
  endforeach()

  # ----------------------------------------------------------------------------

  if(NOT LAPACKE_FIND_QUIETLY)
    message(STATUS "Found LAPACKE and defined the following imported targets:")
    foreach(_comp ${LAPACKE_FIND_COMPONENTS})
      message(STATUS "  - LAPACKE::${_comp}:")
      message(STATUS "      + include:      ${LAPACKE_INCLUDE_DIRS}")
      message(STATUS "      + library:      ${LAPACKE_${_comp}_LIB}")
      message(STATUS "      + dependencies: ${MATH_LIB}")
    endforeach()
  endif()
endif()

# ==============================================================================

mark_as_advanced(
  LAPACKE_FOUND
  LAPACKE_INCLUDE_DIRS
  LAPACKE_LIBRARIES
  )