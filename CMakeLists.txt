cmake_minimum_required(VERSION 3.1)


set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(NetKet)

set (CMAKE_CXX_STANDARD 11)
set (NETKET_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set (EXTERNAL_INCLUDE_DIR ${NETKET_SOURCE_DIR}/External/)


set (EIGEN3_INCLUDE_DIR "${EXTERNAL_INCLUDE_DIR}" CACHE STRING "Eigen3 Include Directory" )
set (JSON_INCLUDE_DIR "${EXTERNAL_INCLUDE_DIR}/Json/" CACHE STRING "JSON for Modern C++ Include Directory" )


include(FindMPI)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${NETKET_SOURCE_DIR})

include_directories (${EIGEN3_INCLUDE_DIR})
include_directories (${JSON_INCLUDE_DIR})


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall")



add_executable(netket netket.cc)

target_link_libraries(netket  ${MPI_LIBRARIES})


##testing

include(CTest) #adds option BUILD_TESTING (default ON)

if(BUILD_TESTING)
   enable_testing()
   add_subdirectory(Test)
endif()


#set( CMAKE_VERBOSE_MAKEFILE on )
