file(GLOB_RECURSE C_SOURCES ./*.c)

# target_include_directories(main PUBLIC "${CMAKE_SOURCE_DIR}/include")
add_executable(main ${C_SOURCES})
# if(MSVC)
#     target_compile_options(main PRIVATE /W4 /WX)
# else()
#     target_compile_options(main PRIVATE -Wall -Wextra -pedantic -Werror)
# endif()
