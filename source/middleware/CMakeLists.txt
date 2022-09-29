set(LIBRARY_NAME edgelearning-middleware)

set(SOURCES
    fnn.cpp
    rnn.cpp
    layer_descriptor.cpp)

if (ENABLE_MLPACK)
    list(APPEND SOURCES
    )
endif()

add_library(${LIBRARY_NAME} OBJECT
    ${SOURCES}
)

if(COVERAGE)
    target_link_libraries(${LIBRARY_NAME} PUBLIC coverage_config)
endif()