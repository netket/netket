include_directories(${EXTERNAL_INCLUDE_DIR}/Catch)

add_library(catch_main OBJECT
    "unit-tests.cc"
)


file(GLOB TEST_SOURCES Graph/*.cc Hamiltonian/*.cc Hilbert/*.cc Machine/*.cc Stats/*.cc)

foreach(file ${TEST_SOURCES})
   get_filename_component(file_basename ${file} NAME_WE)

   string(REGEX REPLACE "unit-([^$]+)" "test-\\1" testcase ${file_basename})

   add_executable(${testcase} $<TARGET_OBJECTS:catch_main> ${file})
   target_link_libraries(${testcase}  ${MPI_LIBRARIES})

   add_test(NAME "${testcase}_default"
      COMMAND ${testcase} ${CATCH_TEST_FILTER}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties("${testcase}_default" PROPERTIES LABELS "default")

    add_test(NAME "${testcase}_all"
      COMMAND ${testcase} ${CATCH_TEST_FILTER} "*"
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties("${testcase}_all" PROPERTIES LABELS "all")

endforeach()
