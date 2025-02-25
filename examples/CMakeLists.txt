# Simple example executable
add_executable(SimpleExample SimpleExample.cpp)
target_link_libraries(SimpleExample PRIVATE Solvex)

# System of ode executable
add_executable(SystemOfODEs SystemOfNLEquations.cpp)
target_link_libraries(SystemOfODEs PRIVATE Solvex)
