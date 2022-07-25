set(UNIT_TESTS
    test_dlmath
    test_dlgraph
    test_layer
    test_loss
    test_cce_loss
    test_mse_loss
    test_dense
    test_activation
    test_recurrent
    test_convolutional
    test_max_pooling
    test_avg_pooling
    test_dropout
    test_model

    test_optimizer
    test_gd_optimizer
    test_adam_optimizer
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