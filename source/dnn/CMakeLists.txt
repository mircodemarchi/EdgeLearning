set(LIBRARY_NAME edgelearning-dnn)

add_library(${LIBRARY_NAME} OBJECT
    layer.cpp
    feedforward.cpp
    activation.cpp
    dense.cpp
    loss.cpp
    cce_loss.cpp
    mse_loss.cpp
    recurrent.cpp
    convolutional.cpp
    pooling.cpp
    max_pooling.cpp
    avg_pooling.cpp
    dropout.cpp
    # concatenate.cpp

    optimizer.cpp
    gd_optimizer.cpp
    adam_optimizer.cpp

    dlgraph.cpp
    model.cpp
)

if(COVERAGE)
    target_link_libraries(${LIBRARY_NAME} PUBLIC coverage_config)
endif()