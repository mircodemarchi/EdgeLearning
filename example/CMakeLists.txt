set(EXAMPLES
    simple_regression
    simple_classification
)

foreach(EX ${EXAMPLES})
    add_executable(${EX} ${EX}.cpp util.hpp)
    target_link_libraries(${EX} edgelearning)
endforeach()