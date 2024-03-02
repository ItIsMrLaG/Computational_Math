file(GLOB_RECURSE C_SOURCES ./*.c)

# target_include_directories(main PUBLIC "${CMAKE_SOURCE_DIR}/include")
find_package(OpenMP)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(main ${C_SOURCES})
target_link_libraries(main m)

if(MSVC)
    target_compile_options(main PRIVATE
        /W4
        /WX
    )
else()
    target_compile_options(main PRIVATE

        # -Wall
        # -Wextra
        # -pedantic
        # -Werror
    )
endif()
