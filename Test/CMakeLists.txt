option(NETKET_Sanitizer "Build test suite with Clang sanitizer" OFF)

add_library(netket_test INTERFACE)
target_link_libraries(netket_test INTERFACE netket_lib Catch2)

if(NETKET_Sanitizer)
    message(STATUS "Building test suite with Clang sanitizer")
    if(NOT MSVC)
        target_compile_options(netket_test
            INTERFACE
                -O2 -g -fno-omit-frame-pointer
                -fsanitize=address -fsanitize=undefined
        )
        target_link_libraries(netket_test
            INTERFACE
                -fsanitize=address -fsanitize=undefined
        )
    endif()
endif()

# Find Boost.Odeint
find_package(Boost)

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
    GroundState/*.cc
    Hamiltonian/*.cc
    Machine/*.cc
    Observable/*.cc
    Sampler/*.cc
    Stats/*.cc
    Utils/*.cc
    Optimizer/*.cc)

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

# Python Tests
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})
# Single call to pytest means that all tests are lumped together -jamesETsmith
#add_test(
#	NAME "Python_Tests"
#	COMMAND py.test -s --verbose ${CMAKE_CURRENT_SOURCE_DIR}
#)

file(GLOB PY_TEST_SOURCES
    Dynamics/*.py
    Graph/*.py
    GroundState/*.py
    Operator/*.py
    Hilbert/*.py
    Machine/*.py
    Observable/*.py
    Sampler/*.py
    Stats/*.py
    Utils/*.py
    Optimizer/*.py
    )


# Make tests for all python files
foreach(_file ${PY_TEST_SOURCES})
  # Strip filename
  get_filename_component(_file_basename ${_file} NAME_WE)
  string(REGEX REPLACE "unit-(.+)" "python-test-\\1" testcase ${_file_basename})

  # Add tests
  add_test(
    NAME ${testcase}
    COMMAND pytest -s --verbose ${_file}
    )

endforeach()
