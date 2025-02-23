add_library(Solvex STATIC
    Equation.cpp
)

# Include Eigen as an INTERFACE dependency
set(EIGEN_DIR ${CMAKE_SOURCE_DIR}/external/eigen-3.4.0)
target_include_directories(Solvex PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN_DIR}  # Make Eigen available to users of Solvex
)

target_compile_features(Solvex PUBLIC cxx_std_17)
