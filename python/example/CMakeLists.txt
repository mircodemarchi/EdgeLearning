
set(EXAMPLES_PYTHON
    mnist_dense
    simple_classification
    simple_regression
    )

foreach(EXAMPLENAME ${EXAMPLES_PYTHON})
    add_custom_target(example_python_${EXAMPLENAME} COMMAND PYTHONPATH=${PROJECT_BINARY_DIR}/python ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLENAME}.py WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_property(TARGET example_python_${EXAMPLENAME} PROPERTY ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR}/python")
    set_property(TARGET example_python_${EXAMPLENAME} PROPERTY LABELS "python")
endforeach()

