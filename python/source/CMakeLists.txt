set(LIBRARY_NAME pyedgelearning-module-obj)

include_directories(${Python3_INCLUDE_DIRS})
include_directories(SYSTEM ${PYBIND11_INCLUDE_DIR})

message(STATUS "Please note that you need to use the Python 3 interpreter to import pyedgelearning correctly.")

add_library(${LIBRARY_NAME} OBJECT
        data_submodule.cpp
        parser_submodule.cpp
        dnn_submodule.cpp
        edgelearning_module.cpp)

if (NOT ENABLE_MLPACK AND NOT WIN32)
    target_compile_options(pyedgelearning-module-obj PUBLIC -Wshadow)
endif()