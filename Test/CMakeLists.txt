option(NETKET_Coverage "Build test suite with coverage information" OFF)
option(NETKET_Sanitizer "Build test suite with Clang sanitizer" OFF)
option(NETKET_Valgrind "Execute test suite with Valgrind" OFF)



if(NETKET_Coverage)
    message(STATUS "Building test suite with coverage information")
    if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        message(FATAL_ERROR "NETKET_Coverage requires GCC.")
    endif()

    # enable profiling
    set(CMAKE_CXX_FLAGS "--coverage -g -O0 -fprofile-arcs -ftest-coverage")

    # from https://github.com/RWTH-HPC/CMake-codecov/blob/master/cmake/FindGcov.cmake
    get_filename_component(COMPILER_PATH "${CMAKE_CXX_COMPILER}" PATH)
    string(REGEX MATCH "^[0-9]+" GCC_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
    find_program(GCOV_BIN NAMES gcov-${GCC_VERSION} gcov HINTS ${COMPILER_PATH})

    # collect all source files from the chosen include dir
    file(GLOB_RECURSE SOURCE_FILES ${NETKET_SOURCE_DIR}*.hpp)

    # add target to collect coverage information and generate HTML file
    # (filter script from https://stackoverflow.com/a/43726240/266378)
    add_custom_target(lcov_html
        COMMAND lcov --directory . --capture --output-file json.info --gcov-tool ${GCOV_BIN} --rc lcov_branch_coverage=1
        COMMAND lcov -e json.info ${SOURCE_FILES} --output-file json.info.filtered --rc lcov_branch_coverage=1
        COMMAND ${CMAKE_SOURCE_DIR}/test/thirdparty/imapdl/filterbr.py json.info.filtered > json.info.filtered.noexcept
        COMMAND genhtml --title "NetKet" --legend --demangle-cpp --output-directory html --show-details --branch-coverage json.info.filtered.noexcept
        COMMENT "Generating HTML report test/html/index.html"
    )
endif()

if(NETKET_Sanitizer)
    message(STATUS "Building test suite with Clang sanitizer")
    if(NOT MSVC)
        set(CMAKE_CXX_FLAGS "-g -O2 -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer")
    endif()
endif()

if(NETKET_Valgrind)
    find_program(CMAKE_MEMORYCHECK_COMMAND valgrind)
    message(STATUS "Executing test suite with Valgrind (${CMAKE_MEMORYCHECK_COMMAND})")
    set(memcheck_command "${CMAKE_MEMORYCHECK_COMMAND} ${CMAKE_MEMORYCHECK_COMMAND_OPTIONS} --error-exitcode=1 --leak-check=full")
    separate_arguments(memcheck_command)
endif()

### Find boost odeint
find_package(Boost)

add_library(netket_test INTERFACE)
target_link_libraries(netket_test INTERFACE netket_lib Catch2)

#
# Catch2 unit tests
#

add_library(catch_main OBJECT "unit-tests.cc")
target_link_libraries(catch_main PUBLIC netket_test)

file(GLOB TEST_SOURCES
    Dynamics/*.cpp
    Graph/*.cc
    Hamiltonian/*.cc
    Hilbert/*.cc
    Machine/*.cc
    Observable/*.cc
    Sampler/*.cc
    Stats/*.cc
    Utils/*.cc)

foreach(_file ${TEST_SOURCES})
   get_filename_component(_file_basename ${_file} NAME_WE)
   string(REGEX REPLACE "unit-(.+)" "test-\\1" testcase ${_file_basename})

   add_executable(${testcase} ${_file})
   target_link_libraries(${testcase} PUBLIC netket_test catch_main)
   if(Boost_FOUND)
       target_link_libraries(${testcase} PUBLIC Boost::boost)
       target_compile_definitions(${testcase} PUBLIC WITH_BOOST_ODEINT)
   endif()

   add_test(NAME "${testcase}_default"
     COMMAND ${testcase} ${CATCH_TEST_FILTER}
     WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
   )
   set_tests_properties("${testcase}_default" PROPERTIES LABELS "default")

   # add_test(NAME "${testcase}_all"
   #   COMMAND ${testcase} ${CATCH_TEST_FILTER} "*"
   #   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
   # )
   # set_tests_properties("${testcase}_all" PROPERTIES LABELS "all")
endforeach()
