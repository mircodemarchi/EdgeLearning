include_directories(.)

add_custom_target(tests)
add_subdirectory(dnn)
add_subdirectory(parser)
add_subdirectory(data)
add_subdirectory(middleware)

