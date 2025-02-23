cmake_minimum_required(VERSION 3.10.0)
project(Solvex VERSION 0.1.0 LANGUAGES C CXX)

# Enable C++17 or later
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the library
add_subdirectory(src)

# Enable testing
enable_testing()
add_subdirectory(tests)

# Examples (Optional)
add_subdirectory(examples)
