set(LIBRARY_NAME edgelearning-parser)

add_library(${LIBRARY_NAME} OBJECT
    parser.cpp
)

if(COVERAGE)
    target_link_libraries(${LIBRARY_NAME} PUBLIC coverage_config)
endif()