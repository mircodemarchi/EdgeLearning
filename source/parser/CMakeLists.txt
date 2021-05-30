set(LIBRARY_NAME ariadnedl-parser)

add_library(${LIBRARY_NAME} OBJECT
    csv.cpp
)

if(COVERAGE)
    target_link_libraries(${LIBRARY_NAME} PUBLIC coverage_config)
endif()