cmake_minimum_required(VERSION 3.1)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
option(ENABLE_TESTS "Enable unit tests." OFF)
option(NETKET_Sanitizer "Build test suite with Clang sanitizer" OFF)

project(NetKet)

add_library(netket_test INTERFACE)
target_link_libraries(netket_test INTERFACE netket_lib Catch2)

set(NETKET_PYTHON_VERSION "" CACHE STRING "Python version to use for compiling modules")

if(ENABLE_TESTS)
   include(CTest) # adds option BUILD_TESTING (default ON)
endif()

include(ExternalProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/)

#
# Dependencies
#

# json
################################################################################
if (NOT EXISTS "${CMAKE_BINARY_DIR}/External/json/json.hpp")
    file(DOWNLOAD
        "https://github.com/nlohmann/json/releases/download/v3.2.0/json.hpp"
        "External/json/json.hpp")
endif()
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json
    INTERFACE "${CMAKE_BINARY_DIR}/External/json")

# optional-lite
################################################################################
if (NOT EXISTS "${CMAKE_BINARY_DIR}/External/optional-lite/nonstd/optional.hpp")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/martinmoene/optional-lite/e240fdaaa4071a980926dea4ed5b610aa8fbc538/include/nonstd/optional.hpp"
        "External/optional-lite/nonstd/optional.hpp")
endif()
add_library(optional_lite INTERFACE)
target_include_directories(optional_lite
    INTERFACE "${CMAKE_BINARY_DIR}/External/optional-lite")


# span-lite
################################################################################
if (NOT EXISTS "${CMAKE_BINARY_DIR}/External/span-lite/nonstd/span.hpp")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/martinmoene/span-lite/308482109be5df5af750f8ad9d4fb76639a28efb/include/nonstd/span.hpp"
        "External/span-lite/nonstd/span.hpp")
endif()
add_library(span_lite INTERFACE)
target_include_directories(span_lite
    INTERFACE "${CMAKE_BINARY_DIR}/External/span-lite")


# Eigen3
################################################################################
ExternalProject_Add(
    eigen_project
    SOURCE_DIR "${CMAKE_BINARY_DIR}/External/Eigen3"
    URL "https://github.com/eigenteam/eigen-git-mirror/archive/3.3.5.tar.gz"
    URL_HASH MD5=305eed19157335016bde9b4d57ee3d35
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(eigen_project SOURCE_DIR)
add_library(Eigen3 INTERFACE)
target_include_directories(Eigen3 SYSTEM INTERFACE ${SOURCE_DIR})



# IETL
################################################################################
add_library(IETL INTERFACE)
target_include_directories(IETL SYSTEM INTERFACE "${CMAKE_SOURCE_DIR}/External/ietl")

# PYBIND11
###############################################################################
ExternalProject_Add(
    pybind11_project
    SOURCE_DIR "${CMAKE_BINARY_DIR}/External/pybind11"
    URL "https://github.com/pybind/pybind11/archive/v2.2.4.tar.gz"
    URL_HASH MD5=c533a107bc95bd0c5525ff650b11bfa1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
if (NOT NETKET_PYTHON_VERSION)
    set(Python_ADDITIONAL_VERSIONS 3.7 3.6 3.5 2.7)
endif()
message(STATUS "Python version: ${NETKET_PYTHON_VERSION}")
message(STATUS "Python versions: ${Python_ADDITIONAL_VERSIONS}")
find_package(PythonLibsNew ${NETKET_PYTHON_VERSION} REQUIRED)

add_library(pybind11 INTERFACE)
ExternalProject_Get_Property(pybind11_project SOURCE_DIR)
target_include_directories(pybind11 SYSTEM INTERFACE ${SOURCE_DIR}/include)
target_include_directories(pybind11 SYSTEM INTERFACE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(pybind11 INTERFACE ${PYTHON_LIBRARIES})
if(NOT MSVC)
    # Greatly reduces the code bloat
    target_compile_options(pybind11 INTERFACE "-fvisibility=hidden")
endif()

# MPI
################################################################################
find_package(MPI REQUIRED)

# Catch2
################################################################################
if (BUILD_TESTING)
    if (NOT EXISTS "${CMAKE_BINARY_DIR}/External/Catch2/catch2/catch.hpp")
        file(DOWNLOAD
            "https://github.com/catchorg/Catch2/releases/download/v2.4.0/catch.hpp"
            "External/Catch2/catch2/catch.hpp")
    endif()
    add_library(Catch2 INTERFACE)
    target_include_directories(Catch2
        INTERFACE "${CMAKE_BINARY_DIR}/External/Catch2/catch2")
endif()

#
################################################################################

# option(NETKET_ENABLE_BENCHMARKS "Enable building benchmarks." ON)
# option(BENCHMARK_ENABLE_TESTING "" OFF)
# option(BENCHMARK_ENABLE_INSTALL "" OFF)
# option(BENCHMARK_DOWNLOAD_DEPENDENCIES "" ON)
# add_subdirectory(External/benchmark)


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX  OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic -Wextra -Wshadow")
endif()


if("Ninja" STREQUAL ${CMAKE_GENERATOR})
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-fdiagnostics-color" COMPILER_SUPPORTS_-fdiagnostics-color)
    if (COMPILER_SUPPORTS_-fdiagnostics-color)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color")
    endif ()
endif()

#
# NetKet
#

add_library(netket_lib INTERFACE)
target_include_directories(netket_lib INTERFACE NetKet)
target_include_directories(netket_lib SYSTEM INTERFACE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(netket_lib
    INTERFACE
        ${MPI_CXX_LIBRARIES}
        Eigen3
        nlohmann_json
        optional_lite
        span_lite
        IETL
)

# Check if we should use native Lapack with Eigen
option(USE_LAPACK "Enable Lapack Support." OFF)
if(USE_LAPACK)
    find_package(LAPACK REQUIRED)
    add_definitions(-DEIGEN_USE_BLAS)
    target_link_libraries(netket_lib INTERFACE ${LAPACK_LIBRARIES})
    find_package(LAPACKE)
    if(LAPACKE_FOUND)
    add_definitions(-DEIGEN_USE_LAPACKE)
    target_link_libraries(netket_lib INTERFACE ${LAPACKE_LIBRARIES})
    endif()

endif()



#target_link_libraries(netket PUBLIC netket_lib)
# Yes, this is ugly, but until CMake 3.5, we can't add an external project
# dependency to an interface library
#add_dependencies(netket eigen_project)



add_library(netket MODULE NetKet/pynetket.cc)
target_link_libraries(netket  PRIVATE pybind11)
target_link_libraries(netket PUBLIC netket_lib)
set_target_properties(netket  PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")
add_dependencies(netket eigen_project pybind11_project)

if(NETKET_Sanitizer)
    message(STATUS "Building python library with Clang sanitizer")
    if(NOT MSVC)
        target_compile_options(netket
            INTERFACE
                -O2 -g -fno-omit-frame-pointer
                -fsanitize=address -fsanitize=undefined
        )
        target_link_libraries(netket
            INTERFACE
                -fsanitize=address -fsanitize=undefined
        )
    endif()
endif()

#
# Testing
#

if(BUILD_TESTING)
   enable_testing()
   add_subdirectory(Test)
endif()

#
# Installing
#

install (TARGETS netket DESTINATION bin)
