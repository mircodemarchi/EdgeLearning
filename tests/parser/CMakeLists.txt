set(UNIT_TESTS
    test_type_checker
    test_parser
    test_csv
    test_mnist
    test_cifar
    test_json
)

foreach(TEST ${UNIT_TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    add_test(${TEST} ${TEST})
    target_link_libraries(${TEST} edgelearning)
endforeach()

add_dependencies(tests ${UNIT_TESTS})