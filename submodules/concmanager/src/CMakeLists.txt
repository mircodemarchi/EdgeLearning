add_library(CONCMANAGER_SRC OBJECT
        thread.cpp
        thread_pool.cpp
        task_manager.cpp
        )

# if(COVERAGE)
#     target_link_libraries(CONCMANAGER_SRC coverage_config)
# endif()