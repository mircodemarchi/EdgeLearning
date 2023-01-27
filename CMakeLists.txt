cmake_minimum_required(VERSION 3.16)
message(STATUS "CMake version ${CMAKE_VERSION}")

project(EdgeLearning VERSION 0.1)

option(COVERAGE "Enable coverage reporting" OFF)
if(COVERAGE)
    message(STATUS "Enabled coverage reporting")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19.20)
        message(FATAL_ERROR "MSVC version must be at least 19.20!")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    if (NOT EXISTS "/etc/nv_tegra_release") # Nvidia Jetson support.
        if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10.2)
            message(FATAL_ERROR "GCC version must be at least 10.2!")
        endif()
    endif()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11.0)
        message(FATAL_ERROR "Clang version must be at least 11.0!")
    endif()
else()
    message(WARNING "You are using an unsupported compiler! MSVC, GCC and Clang are supported.")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_MACOSX_RPATH 1)
set(BUILD_SHARED_LIBS ON)
if(WIN32)
    set(CMAKE_CXX_STANDARD 20)
else()
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(LIBRARY_KIND STATIC)
elseif(COVERAGE)
    set(LIBRARY_KIND STATIC)
else()
    set(LIBRARY_KIND SHARED)
endif()

# Options
set(ENABLE_MLPACK false CACHE BOOL "Enable MLPACK library. Warning: not supported.")
add_definitions(-DENABLE_MLPACK=${ENABLE_MLPACK})

set(BUILD_PYTHON false CACHE BOOL "Build Python interface library.")
# add_definitions(-DBUILD_PYTHON=${BUILD_PYTHON})

if (NOT ENABLE_MLPACK)
    if(NOT WIN32)
        set(ANY_TARGET_WARN all extra pedantic sign-conversion cast-qual disabled-optimization
                init-self missing-include-dirs sign-promo switch-default undef redundant-decls
                strict-aliasing unused-parameter shadow error)
        foreach(WARN ${ANY_TARGET_WARN})
            add_compile_options(-W${WARN})
        endforeach()
    else()
        add_compile_options(/WX)
    endif()
endif()

find_package(Git)
if(GIT_FOUND)
    if (NOT EXISTS ${PROJECT_SOURCE_DIR}/.git) # Manages the case when an archive is used
        execute_process(COMMAND git init WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_QUIET ERROR_QUIET)
        execute_process(COMMAND rm -Rf ${PROJECT_SOURCE_DIR}/submodules/conclog OUTPUT_QUIET ERROR_QUIET)
        execute_process(COMMAND rm -Rf ${PROJECT_SOURCE_DIR}/submodules/betterthreads OUTPUT_QUIET ERROR_QUIET)
        execute_process(COMMAND git submodule add -f https://github.com/ariadne-cps/conclog submodules/conclog WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_QUIET ERROR_QUIET)
        execute_process(COMMAND git submodule add -f https://github.com/ariadne-cps/betterthreads submodules/betterthreads WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_QUIET ERROR_QUIET)
    endif()
    message(STATUS "Getting all submodules if not downloaded already...")
    execute_process(COMMAND git submodule update --init --recursive WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} OUTPUT_QUIET ERROR_QUIET)
    if(NOT EXISTS ${PROJECT_SOURCE_DIR}/submodules/conclog/CMakeLists.txt OR NOT EXISTS ${PROJECT_SOURCE_DIR}/submodules/betterthreads/CMakeLists.txt)
        message(ERROR "ConcLog and BetterThreads download failure")
    else()
        message(STATUS "ConcLog and BetterThreads submodules found")
    endif()
else()
    message(FATAL_ERROR "Git not found, no submodules could be downloaded")
endif()

if (NOT EXISTS ${PROJECT_SOURCE_DIR}/submodules/.symbolic)
    execute_process(COMMAND mkdir ${PROJECT_SOURCE_DIR}/submodules/.symbolic)
    message(STATUS "Created symbolic links directory for submodules")
endif()

if (NOT EXISTS ${PROJECT_SOURCE_DIR}/submodules/.symbolic/conclog)
    execute_process(COMMAND ln -s ${PROJECT_SOURCE_DIR}/submodules/conclog/include ${PROJECT_SOURCE_DIR}/submodules/.symbolic/conclog)
    message(STATUS "Created symbolic link to ConcLog submodule")
endif()

if (NOT EXISTS ${PROJECT_SOURCE_DIR}/submodules/.symbolic/betterthreads)
    execute_process(COMMAND ln -s ${PROJECT_SOURCE_DIR}/submodules/betterthreads/include ${PROJECT_SOURCE_DIR}/submodules/.symbolic/betterthreads)
    message(STATUS "Created symbolic link to BetterThreads submodule")
endif()

# Find MLPACK and dependencies
message(STATUS "MLPACK enabled: ${ENABLE_MLPACK}")
find_package(Armadillo 8.400.0 QUIET)
find_package(MLPACK QUIET)

if(ENABLE_MLPACK AND NOT MLPACK_FOUND)
    message(FATAL_ERROR "MLPACK enabled but not found")
endif()

if(ENABLE_MLPACK AND MLPACK_FOUND)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND ((${CMAKE_HOST_SYSTEM_NAME} STREQUAL "Darwin") OR APPLE))
        message(FATAL_ERROR "MLPACK is not supported with MacOS and GCC GNU compiler, configure Clang instead.")
    endif()
endif()

include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/source ${PROJECT_SOURCE_DIR}/submodules/.symbolic)
if(ENABLE_MLPACK)
    include_directories(${ARMADILLO_INCLUDE_DIRS})
    include_directories(${MLPACK_INCLUDE_DIRS})
endif()

if(NOT TARGET edgelearning)
    add_subdirectory(source)
    if(BUILD_PYTHON)
        message(STATUS "PYTHON interface enabled")
        add_subdirectory(python)
    endif()

    if(NOT TARGET tests)
        if(COVERAGE)
            include(CodeCoverage)
            append_coverage_compiler_flags()
        endif()

        enable_testing()
        include(CTest)
        add_subdirectory(tests)
        if(BUILD_PYTHON)
            add_subdirectory(python/tests)
        endif()
    endif()

    add_subdirectory(submodules)

    add_subdirectory(profile)

    add_library(edgelearning ${LIBRARY_KIND}
        $<TARGET_OBJECTS:edgelearning-dnn>
        $<TARGET_OBJECTS:edgelearning-middleware>
        $<TARGET_OBJECTS:edgelearning-parser>
        )
    target_link_libraries(edgelearning
        conclog
        betterthreads
        )

    if (EXISTS "/etc/nv_tegra_release")
        target_link_libraries(edgelearning -lstdc++fs)
    endif()

    if(ENABLE_MLPACK)
        target_link_libraries(edgelearning ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})
    endif()

    set(BUILD_EXAMPLE true CACHE BOOL "Build EdgeLearning examples code")
    if(BUILD_EXAMPLE)
        add_subdirectory(example)
    endif()
endif()
