cmake_minimum_required(VERSION 3.1)


set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

project(NetKet)

set (CMAKE_CXX_STANDARD 11)
set (NETKET_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/NetKet/)
set (EXTERNAL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/External/)


set (EIGEN3_INCLUDE_DIR "${EXTERNAL_INCLUDE_DIR}" CACHE STRING "Eigen3 Include Directory" )
set (JSON_INCLUDE_DIR "${EXTERNAL_INCLUDE_DIR}/Json/" CACHE STRING "JSON for Modern C++ Include Directory" )

#Find MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
else (MPI_FOUND)
    message(SEND_ERROR "This application cannot compile without MPI")
endif (MPI_FOUND)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${NETKET_SOURCE_DIR})

include_directories (${EIGEN3_INCLUDE_DIR})
include_directories (${JSON_INCLUDE_DIR})


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


set (NETKET_SOURCES ${NETKET_SOURCE_DIR}/netket.cc)
set (NETKET_SOURCES ${NETKET_SOURCES} ${NETKET_SOURCE_DIR}/Graph)
set (NETKET_SOURCES ${NETKET_SOURCES} ${NETKET_SOURCE_DIR}/Hamiltonian)
set (NETKET_SOURCES ${NETKET_SOURCES} ${NETKET_SOURCE_DIR}/Headers)
set (NETKET_SOURCES ${NETKET_SOURCES} ${NETKET_SOURCE_DIR}/Hilbert)
set (NETKET_SOURCES ${NETKET_SOURCES} ${NETKET_SOURCE_DIR}/Learning)
set (NETKET_SOURCES ${NETKET_SOURCES} ${NETKET_SOURCE_DIR}/Lookup)
set (NETKET_SOURCES ${NETKET_SOURCES} ${NETKET_SOURCE_DIR}/Machine)
set (NETKET_SOURCES ${NETKET_SOURCES} ${NETKET_SOURCE_DIR}/Observable)
set (NETKET_SOURCES ${NETKET_SOURCES} ${NETKET_SOURCE_DIR}/Sampler)
set (NETKET_SOURCES ${NETKET_SOURCES} ${NETKET_SOURCE_DIR}/Stats)
set (NETKET_SOURCES ${NETKET_SOURCES} ${NETKET_SOURCE_DIR}/Utils)

add_executable(netket ${NETKET_SOURCES})

target_link_libraries(netket  ${MPI_LIBRARIES})


##testing

include(CTest) #adds option BUILD_TESTING (default ON)

if(BUILD_TESTING)
   enable_testing()
   add_subdirectory(Test)
endif()

##installing
install (TARGETS netket DESTINATION bin)

#set( CMAKE_VERBOSE_MAKEFILE on )
