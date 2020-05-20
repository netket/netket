cmake_minimum_required(VERSION 3.12)

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
option(NETKET_USE_OPENMP "Use OpenMP for multithreading" OFF)
option(NETKET_USE_SANITIZER "Build test suite with Clang sanitizer" OFF)
option(NETKET_USE_BLAS "Use system BLAS instead of Eigen's implementation" ON)
option(NETKET_USE_LAPACK "Use system LAPACK instead of Eigen's implementation" OFF)

option(NETKET_NATIVE "Use -march=native" OFF)


set(NETKET_PYTHON_VERSION "" CACHE STRING "Python version to use for compiling modules")

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "[NetKet] CMAKE_BUILD_TYPE not specified, setting it to "
                 "Release. Use `-DCMAKE_BUILD_TYPE=...` to overwrite.")
  set(CMAKE_BUILD_TYPE Release)
endif()


#
# Dependencies
#


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

# any
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
    set(Python_ADDITIONAL_VERSIONS 3.7 3.6 3.5)
endif()
find_package(PythonLibsNew ${NETKET_PYTHON_VERSION} REQUIRED)

add_library(pybind11 INTERFACE)
ExternalProject_Get_Property(pybind11_project SOURCE_DIR)
target_include_directories(pybind11 SYSTEM INTERFACE ${SOURCE_DIR}/include)
target_include_directories(pybind11 SYSTEM INTERFACE ${PYTHON_INCLUDE_DIRS})

if(APPLE AND DEFINED ENV{CONDA_PREFIX})
      TARGET_LINK_LIBRARIES(pybind11 INTERFACE "-undefined dynamic_lookup")
      message(STATUS "Building in conda environment on MAC")
else()
      target_link_libraries(pybind11 INTERFACE ${PYTHON_LIBRARIES})
endif()

# Greatly reduces the code bloat
target_compile_options(pybind11 INTERFACE "-fvisibility=hidden")

# MPI
################################################################################
find_package(MPI REQUIRED)
if(MPI_C_VERSION_MAJOR LESS 3)
message( FATAL_ERROR "NetKet requires at least MPI 3." )
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
        optional_lite
        span_lite
        linb_any
        ${CMAKE_DL_LIBS}
)


set(NETKET_WARNING_FLAGS
    -Wall -Wextra -pedantic
    -Wshadow
)
target_compile_options(netket_lib INTERFACE ${NETKET_WARNING_FLAGS})


if(${CMAKE_GENERATOR} STREQUAL "Ninja")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        target_compile_options(netket_lib INTERFACE -fcolor-diagnostics)
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        target_compile_options(netket_lib INTERFACE -fdiagnostics-color=always)
    endif()
endif()



set(NETKET_SOURCES
    Sources/pynetket.cc
    Sources/Graph/lattice.cc
    Sources/Graph/custom_graph.cc
    Sources/Graph/abstract_graph.cc
    Sources/Graph/py_graph.cc
    Sources/Utils/exceptions.cc
    )

add_library(netket MODULE ${NETKET_SOURCES})

# A workaround for missing __cpu_model bug in gcc-5 and Clangs earlier than 6
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0
        AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
    target_link_libraries(netket_lib INTERFACE gcc_s gcc)
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.0)
    string(REGEX REPLACE "^([0-9]+)\..*$" "\\1" NETKET_CLANG_MAJOR "${CMAKE_CXX_COMPILER_VERSION}")
    string(REGEX REPLACE "^[0-9]+\.([0-9])+.*$" "\\1" NETKET_CLANG_MINOR "${CMAKE_CXX_COMPILER_VERSION}")
    find_program(LLVM_CONFIG_EXECUTABLE
        NAMES
            llvm-config-${CMAKE_CXX_COMPILER_VERSION}
            llvm-config-${NETKET_CLANG_MAJOR}.${NETKET_CLANG_MINOR}
            llvm-config-${NETKET_CLANG_MAJOR})

    if (NOT LLVM_CONFIG_EXECUTABLE)
        message(FATAL_ERROR
            "llvm-config: not found. Clang ${CMAKE_CXX_COMPILER_VERSION} has a bug. "
            "A workaround for it involves linking against libclang_rt.builtins-x86_64. "
            "To find that library, we need llvm-config.")
    endif()
    execute_process(
        COMMAND ${LLVM_CONFIG_EXECUTABLE} --libdir
        OUTPUT_VARIABLE NETKET_LLVM_LIBDIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_library(NETKET_CLANG_RT
        NAMES clang_rt.builtins-x86_64
        HINTS ${NETKET_LLVM_LIBDIR}/clang/${CMAKE_CXX_COMPILER_VERSION}/lib/linux)
    if(NOT NETKET_CLANG_RT)
        message(FATAL_ERROR
            "clang_rt: not found. Compilation will fail because "
            "of a known bug in Clang ${CMAKE_CXX_COMPILER_VERSION}")
    endif()
    message(STATUS "Found clang_rt: ${NETKET_CLANG_RT}")
    target_link_libraries(netket_lib INTERFACE ${NETKET_CLANG_RT})
endif()

target_link_libraries(netket PUBLIC netket_lib pybind11)
set_target_properties(netket PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                        SUFFIX "${PYTHON_MODULE_EXTENSION}")
add_dependencies(netket eigen_project pybind11_project)
set_target_properties(netket PROPERTIES OUTPUT_NAME "_C_netket")
