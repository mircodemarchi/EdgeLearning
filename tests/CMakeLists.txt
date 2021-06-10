set(UNIT_TESTS
        # test_estimators
        test_parser
        test_csv
        test_dnn
    )

foreach(TEST ${UNIT_TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    if(COVERAGE)
        target_compile_options(${TEST} PUBLIC ${COVERAGE_COMPILER_FLAGS})
    endif()
    add_test(${TEST} ${TEST})
    target_link_libraries(${TEST} ariadnedl)
endforeach()
