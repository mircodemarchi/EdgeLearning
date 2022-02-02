set(UNIT_TESTS
    test_dlmath
    test_layer
    test_loss
    test_cce_loss
    test_recurrent
    test_model
)

foreach(TEST ${UNIT_TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    if(COVERAGE)
        target_compile_options(${TEST} PUBLIC ${COVERAGE_COMPILER_FLAGS})
    endif()
    add_test(${TEST} ${TEST})
    target_link_libraries(${TEST} edgelearning)
endforeach()
