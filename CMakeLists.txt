cmake_minimum_required(VERSION 3.1)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(NetKet LANGUAGES C CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules/)

include(CTest)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NETKET_BUILD_TESTING "Build unit tests." OFF)
option(NETKET_USE_OPENMP "Use OpenMP for multithreading" ON)
option(NETKET_USE_SANITIZER "Build test suite with Clang sanitizer" OFF)
option(NETKET_USE_BLAS "Use system BLAS instead of Eigen's implementation" ON)
option(NETKET_USE_LAPACK "Use system LAPACK instead of Eigen's implementation" ON)
set(NETKET_PYTHON_VERSION "" CACHE STRING "Python version to use for compiling modules")

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "[NetKet] CMAKE_BUILD_TYPE not specified, setting it to "
                 "Release. Use `-DCMAKE_BUILD_TYPE=...` to overwrite.")
  set(CMAKE_BUILD_TYPE Release)
endif()


#
# Dependencies
#

# json
################################################################################
ExternalProject_Add(
    json_project
    SOURCE_DIR "${CMAKE_BINARY_DIR}/External/json/include"
    URL "https://github.com/nlohmann/json/releases/download/v3.6.1/include.zip"
    URL_HASH MD5=0dc903888211db3a0f170304cd9f3a89
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(json_project SOURCE_DIR)
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json SYSTEM INTERFACE ${SOURCE_DIR})
target_include_directories(nlohmann_json SYSTEM INTERFACE ${SOURCE_DIR}/nlohmann)

# Sleef
################################################################################
ExternalProject_Add(
    sleef_project
    SOURCE_DIR "${CMAKE_BINARY_DIR}/External/sleef/"
    URL "https://github.com/shibatch/sleef/archive/3.4.0.tar.gz"
    URL_HASH MD5=fdf620e232a56affaca90c107b70539b
    CMAKE_GENERATOR "${CMAKE_GENERATOR}"
    CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=OFF -DBUILD_DFT=OFF -DBUILD_GNUABI_LIBS=OFF
    -DBUILD_TESTS=OFF -DBUILD_QUAD=OFF -DSLEEF_SHOW_CONFIG=ON
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    INSTALL_COMMAND ""
    # NOTE: This is kind of a hack since we're explicitly specifying
    # "sleef_project-prefix/...", but at least it works.
    BUILD_BYPRODUCTS "sleef_project-prefix/src/sleef_project-build/lib/libsleef.a"
)
add_library(Sleef INTERFACE IMPORTED)
ExternalProject_Get_Property(sleef_project BINARY_DIR)
target_include_directories(Sleef INTERFACE ${BINARY_DIR}/include)
target_link_libraries(Sleef INTERFACE ${BINARY_DIR}/lib/libsleef.a)

# optional-lite
################################################################################
if (NOT EXISTS "${CMAKE_BINARY_DIR}/External/optional-lite/nonstd/optional.hpp")
    file(DOWNLOAD
        "https://github.com/martinmoene/optional-lite/releases/download/v3.2.0/optional.hpp"
        "External/optional-lite/nonstd/optional.hpp")
endif()
add_library(optional_lite INTERFACE)
target_include_directories(optional_lite
    INTERFACE "${CMAKE_BINARY_DIR}/External/optional-lite")


# span-lite
################################################################################
if (NOT EXISTS "${CMAKE_BINARY_DIR}/External/span-lite/nonstd/span.hpp")
    file(DOWNLOAD
        "https://github.com/martinmoene/span-lite/releases/download/v0.5.0/span.hpp"
        "External/span-lite/nonstd/span.hpp")
endif()
add_library(span_lite INTERFACE)
target_include_directories(span_lite
    INTERFACE "${CMAKE_BINARY_DIR}/External/span-lite")

# any-lite
################################################################################
if (NOT EXISTS "${CMAKE_BINARY_DIR}/External/any/any.hpp")
    file(DOWNLOAD
        "https://raw.githubusercontent.com/thelink2012/any/master/any.hpp"
        "External/any/any.hpp")
endif()
add_library(linb_any INTERFACE)
target_include_directories(linb_any
    INTERFACE "${CMAKE_BINARY_DIR}/External/any")


# Eigen3
################################################################################
ExternalProject_Add(
    eigen_project
    SOURCE_DIR "${CMAKE_BINARY_DIR}/External/Eigen3"
    URL "https://github.com/eigenteam/eigen-git-mirror/archive/3.3.7.tar.gz"
    URL_HASH MD5=77a2c934eaf35943c43ee600a83b72df
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(eigen_project SOURCE_DIR)
add_library(Eigen3 INTERFACE)
target_include_directories(Eigen3 SYSTEM INTERFACE ${SOURCE_DIR})


# pybind11
###############################################################################
ExternalProject_Add(
    pybind11_project
    SOURCE_DIR "${CMAKE_BINARY_DIR}/External/pybind11"
    URL "https://github.com/pybind/pybind11/archive/v2.3.0.tar.gz"
    URL_HASH MD5=e2120c5b0e3c20a93f2dfcdc55026ba8
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
if (NOT NETKET_PYTHON_VERSION)
    set(Python_ADDITIONAL_VERSIONS 3.7 3.6 3.5 2.7)
endif()
find_package(PythonLibsNew ${NETKET_PYTHON_VERSION} REQUIRED)

add_library(pybind11 INTERFACE)
ExternalProject_Get_Property(pybind11_project SOURCE_DIR)
target_include_directories(pybind11 SYSTEM INTERFACE ${SOURCE_DIR}/include)
target_include_directories(pybind11 SYSTEM INTERFACE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(pybind11 INTERFACE ${PYTHON_LIBRARIES})
# Greatly reduces the code bloat
target_compile_options(pybind11 INTERFACE "-fvisibility=hidden")


# MPI
################################################################################
find_package(MPI REQUIRED)


# OpenMP
################################################################################
if(NETKET_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    if(NOT TARGET OpenMP::OpenMP_CXX)
        find_package(Threads REQUIRED)
        add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)
        set_property(TARGET OpenMP::OpenMP_CXX
                     PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
        # Only works if the same flag is passed to the linker; use CMake 3.9+ otherwise (Intel, AppleClang)
        set_property(TARGET OpenMP::OpenMP_CXX
                     PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
    endif()
endif()

# Catch2
################################################################################
if (BUILD_TESTING AND NETKET_BUILD_TESTING)
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
# NetKet
#

add_library(netket_lib INTERFACE)
target_include_directories(netket_lib INTERFACE Sources)
target_include_directories(netket_lib SYSTEM INTERFACE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(netket_lib
    INTERFACE
        ${MPI_CXX_LIBRARIES}
        Eigen3
        nlohmann_json
        optional_lite
        span_lite
        linb_any
        Sleef
        ${CMAKE_DL_LIBS}
)
if(NETKET_USE_OPENMP)
    target_link_libraries(netket_lib INTERFACE OpenMP::OpenMP_CXX)
endif()

set(NETKET_WARNING_FLAGS
    -Wall -Wextra -pedantic
    -Wshadow
)
target_compile_options(netket_lib INTERFACE ${NEKET_WARNING_FLAGS})

if(${CMAKE_GENERATOR} STREQUAL "Ninja")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(netket_lib INTERFACE -fcolor-diagnostics)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options(netket_lib INTERFACE -fdiagnostics-color=always)
    endif()
endif()

if(NETKET_USE_BLAS)
    find_package(BLAS REQUIRED)
    target_compile_definitions(netket_lib INTERFACE EIGEN_USE_BLAS=1)
    target_link_libraries(netket_lib INTERFACE ${BLAS_LIBRARIES})
endif()

if(NEKET_USE_LAPACK)
    find_package(LAPACK REQUIRED)
    target_compile_definitions(netket_lib INTERFACE EIGEN_USE_LAPACK=1 EIGEN_USE_LAPACKE=1)
    target_link_libraries(netket_lib INTERFACE ${LAPACK_LIBRARIES})
endif()


set(NETKET_SOURCES
    Sources/pynetket.cc
    Sources/Dynamics/py_dynamics.cpp
    Sources/Graph/hypercube.cc
    Sources/Graph/lattice.cc
    Sources/Graph/custom_graph.cc
    Sources/Graph/abstract_graph.cc
    Sources/Graph/py_graph.cc
    Sources/Hilbert/hilbert_index.cc
    Sources/Hilbert/bosons.cc
    Sources/Hilbert/spins.cc
    Sources/Hilbert/custom_hilbert.cc
    Sources/Hilbert/py_hilbert.cc
    Sources/Machine/abstract_machine.cc
    Sources/Machine/jastrow.cc
    Sources/Machine/jastrow_symm.cc
    Sources/Machine/mps_periodic.cc
    Sources/Machine/rbm_multival.cc
    Sources/Machine/rbm_spin.cc
    Sources/Machine/rbm_spin_phase.cc
    Sources/Machine/rbm_spin_real.cc
    Sources/Machine/rbm_spin_symm.cc
    Sources/Machine/rbm_spin_v2.cc
    Sources/Machine/py_machine.cc
    Sources/Machine/py_abstract_machine.cc
    Sources/Operator/abstract_operator.cc
    Sources/Sampler/vmc_sampling.cc
    Sources/Sampler/metropolis_local_v2.cc
    Sources/Optimizer/stochastic_reconfiguration.cc
    Sources/Optimizer/py_stochastic_reconfiguration.cc
    Sources/Utils/json_utils.cc
    Sources/Utils/mpi_interface.cc
    Sources/Utils/py_utils.cc
)

add_library(netket MODULE ${NETKET_SOURCES})
add_library(log_cosh OBJECT Sources/Utils/log_cosh.cc)
target_compile_options(log_cosh PRIVATE
    -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -mavx -mavx2 -mfma)
target_link_libraries(log_cosh PUBLIC netket_lib)
set_property(TARGET log_cosh PROPERTY POSITION_INDEPENDENT_CODE ON)
add_dependencies(log_cosh eigen_project sleef_project)
target_link_libraries(netket PUBLIC netket_lib log_cosh pybind11)
set_target_properties(netket PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                        SUFFIX "${PYTHON_MODULE_EXTENSION}")
add_dependencies(netket json_project eigen_project pybind11_project sleef_project)
set_target_properties(netket PROPERTIES OUTPUT_NAME "_C_netket")


# if(NETKET_SANITIZER)
#     message(STATUS "[NetKet] Building python library with address and UB sanitizers")
#     target_compile_options(netket_lib
#         INTERFACE
#             -g -fno-omit-frame-pointer
#             -fsanitize=address -fsanitize=undefined
#     )
#     target_link_libraries(netket_lib
#         INTERFACE
#             -fsanitize=address -fsanitize=undefined
#     )
# endif()

#
# Testing
#

if(BUILD_TESTING AND NETKET_BUILD_TESTING)
    add_library(netket_test INTERFACE)
    target_link_libraries(netket_test INTERFACE netket_lib Catch2)
    enable_testing()
    add_subdirectory(Test)
endif()
