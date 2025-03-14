# System of ode executable
add_executable(SystemOfODEs SystemOfNLEquations.cpp)
target_link_libraries(SystemOfODEs PRIVATE Solvex)
