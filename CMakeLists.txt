cmake_minimum_required(VERSION 3.1)

set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(NETKET_ENABLE_TESTS OFF)

project(NetKet)

if(NETKET_ENABLE_TESTS)
   include(CTest) # adds option BUILD_TESTING (default ON)
endif()

include(ExternalProject)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

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
        "https://github.com/martinmoene/optional-lite/releases/download/v3.0.0/optional.hpp"
        "External/optional-lite/nonstd/optional.hpp")
endif()
add_library(optional_lite INTERFACE)
target_include_directories(optional_lite
    INTERFACE "${CMAKE_BINARY_DIR}/External/optional-lite")

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
)

# Check if we can use native Lapack with Eigen
find_package(Lapack)
if(LAPACK_FOUND)
    target_link_libraries(netket_lib INTERFACE ${LAPACK_LIBRARIES})
    target_compile_options(netket_lib INTERFACE -DEIGEN_USE_BLAS -DEIGEN_USE_LAPACKE)
endif()

add_executable(netket NetKet/netket.cc)
target_link_libraries(netket PUBLIC netket_lib)
# Yes, this is ugly, but until CMake 3.5, we can't add an external project
# dependency to an interface library
add_dependencies(netket eigen_project)

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
