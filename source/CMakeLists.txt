set(LIBRARY_NAME ariadnedl-all)

add_library(${LIBRARY_NAME} OBJECT
    replaceme.cpp
)

if(COVERAGE)
    target_link_libraries(${LIBRARY_NAME} PUBLIC coverage_config)
endif()