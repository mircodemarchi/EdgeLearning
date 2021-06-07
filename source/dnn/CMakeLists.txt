set(LIBRARY_NAME ariadnedl-dnn)

add_library(${LIBRARY_NAME} OBJECT
    layer.cpp
)

if(COVERAGE)
    target_link_libraries(${LIBRARY_NAME} PUBLIC coverage_config)
endif()