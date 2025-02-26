cmake_minimum_required(VERSION 3.10)
project(TestSolvex)

set(CMAKE_BUILD_TYPE Debug)  # Enables debugging symbols
set(CMAKE_CXX_STANDARD 17)

# Enable testing support
enable_testing()

# Add test executable
add_executable(TestSolvex TestSolvex.cpp)

# ✅ Explicitly link Google Test, pthread, and Solvex
target_link_libraries(TestSolvex PRIVATE Solvex)
