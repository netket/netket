option(NETKET_Sanitizer "Build test suite with Clang sanitizer" OFF)
option(NETKET_Valgrind "Execute test suite with Valgrind" OFF)



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
# This is _very_ ugly, but in CMake 3.1 object libraries can't link to anything
# so we can't make it use netket_test
get_property(_catch2_include_dir
    TARGET Catch2
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(catch_main PRIVATE ${_catch2_include_dir} ${MPI_C_INCLUDE_PATH})

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

   add_executable(${testcase} $<TARGET_OBJECTS:catch_main> ${_file})
   target_link_libraries(${testcase} PUBLIC netket_test)
   if(Boost_FOUND)
       target_include_directories(${testcase} SYSTEM PUBLIC ${Boost_INCLUDE_DIRS})
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
