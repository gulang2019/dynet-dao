project(dao-simulator)
cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES *.cc)
file(GLOB_RECURSE HEADERS *.h)

add_executable(dao-simulator ${SOURCES} ${HEADERS})