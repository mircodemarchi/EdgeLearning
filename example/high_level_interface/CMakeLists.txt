set(EXAMPLES 
    hl_regression
)
foreach(EX ${EXAMPLES})
    add_executable(${EX} ${EX}.cpp)
    target_link_libraries(${EX} edgelearning)
endforeach()