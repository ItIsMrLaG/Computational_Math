cmake_minimum_required(VERSION 3.7)
project(tests LANGUAGES C)

set(CMAKE_C_STANDARD 99)
file(MAKE_DIRECTORY experiments)

add_subdirectory(src)