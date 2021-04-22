set(LIBRARY_NAME ariadnedl-estimators)

add_library(${LIBRARY_NAME} OBJECT
    time_estimator.cpp
)

if(COVERAGE)
    target_link_libraries(${LIBRARY_NAME} PUBLIC coverage_config)
endif()