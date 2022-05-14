set(LIBRARY_NAME edgelearning-dnn)

add_library(${LIBRARY_NAME} OBJECT
    layer.cpp
    model.cpp
    feedforward.cpp
    activation.cpp
    dense.cpp
    loss.cpp
    cce_loss.cpp
    mse_loss.cpp
    gd_optimizer.cpp
    recurrent.cpp
    convolutional.cpp
    pooling.cpp
    max_pooling.cpp
    avg_pooling.cpp
    dropout.cpp
)

if(COVERAGE)
    target_link_libraries(${LIBRARY_NAME} PUBLIC coverage_config)
endif()