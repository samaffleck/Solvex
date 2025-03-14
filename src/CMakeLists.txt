add_library(Solvex STATIC
    Equations.cpp
    Solvex.cpp
    BoundaryConditions.cpp
)

# Include Eigen as an INTERFACE dependency
set(EIGEN_DIR ${CMAKE_SOURCE_DIR}/external/eigen-3.4.0)
set(AUTO_DIFF ${CMAKE_SOURCE_DIR}/external)
set(DAE_CPP ${CMAKE_SOURCE_DIR}/external/dae-cpp)
target_include_directories(Solvex PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN_DIR}  # Make Eigen available to users of Solvex
    ${AUTO_DIFF}  
    ${DAE_CPP}  
)

target_compile_features(Solvex PUBLIC cxx_std_17)
