cmake_minimum_required(VERSION 3.9)

project(NetKet)

include(CTest) # adds option BUILD_TESTING (default ON)
include(FetchContent)

# set (CMAKE_CXX_STANDARD 11)
# set (NETKET_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/NetKet/)
# set (EXTERNAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/External/)

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
FetchContent_Declare(
    eigen_project
    SOURCE_DIR "External/Eigen3"
    URL "https://github.com/eigenteam/eigen-git-mirror/archive/3.3.5.tar.gz"
    LOG_DOWNLOAD ON
)
FetchContent_GetProperties(eigen_project)
if(NOT eigen_project_POPULATED)
    FetchContent_Populate(eigen_project)
endif()
add_library(Eigen3 INTERFACE)
target_include_directories(Eigen3 SYSTEM INTERFACE ${eigen_project_SOURCE_DIR})

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
target_compile_features(netket_lib INTERFACE cxx_std_11)
target_include_directories(netket_lib INTERFACE NetKet)
target_link_libraries(netket_lib
    INTERFACE
        MPI::MPI_C
        Eigen3
        nlohmann_json
        optional_lite
)

add_executable(netket NetKet/netket.cc)
target_link_libraries(netket PUBLIC netket_lib)

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

