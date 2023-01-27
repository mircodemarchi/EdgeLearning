set(PROFILE_FILES
    profile_fnn_regression
    profile_fnn_classification
    profile_dense
)

foreach(PROFILE ${PROFILE_FILES})
    add_executable(${PROFILE} ${PROFILE}.cpp)
    target_link_libraries(${PROFILE} edgelearning)
endforeach()

add_custom_target(profiles)
add_dependencies(profiles ${PROFILE_FILES})