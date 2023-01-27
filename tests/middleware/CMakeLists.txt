set(UNIT_TESTS
    test_fnn
    test_layer_descriptor
)

foreach(TEST ${UNIT_TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    add_test(${TEST} ${TEST})
    target_link_libraries(${TEST} edgelearning)
endforeach()

add_dependencies(tests ${UNIT_TESTS})