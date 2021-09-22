set(LIBRARY_NAME ariadnedl-middleware)

add_library(${LIBRARY_NAME} OBJECT
    feed_forward.cpp
    recurrent.cpp
)

if(COVERAGE)
    target_link_libraries(${LIBRARY_NAME} PUBLIC coverage_config)
endif()