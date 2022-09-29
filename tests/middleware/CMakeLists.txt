set(UNIT_TESTS
    test_fnn
    test_layer_descriptor
)

foreach(TEST ${UNIT_TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    if(COVERAGE)
        target_compile_options(${TEST} PUBLIC ${COVERAGE_COMPILER_FLAGS})
    endif()
    add_test(${TEST} ${TEST})
    target_link_libraries(${TEST} edgelearning)
endforeach()

add_dependencies(tests ${UNIT_TESTS})