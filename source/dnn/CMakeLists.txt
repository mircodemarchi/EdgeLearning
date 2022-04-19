set(LIBRARY_NAME edgelearning-dnn)

add_library(${LIBRARY_NAME} OBJECT
    layer.cpp
    model.cpp
    feedforward.cpp
    dense.cpp
    loss.cpp
    cce_loss.cpp
    mse_loss.cpp
    gd_optimizer.cpp
    recurrent.cpp
    convolutional.cpp
    pooling.cpp
    max_pooling.cpp
)

if(COVERAGE)
    target_link_libraries(${LIBRARY_NAME} PUBLIC coverage_config)
endif()