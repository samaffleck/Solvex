# Solvex
*A Lightweight C++ Library for Nonlinear Equation Solving Using Eigen*

## Overview
Solvex is a lightweight C++ library designed for solving nonlinear systems of equations efficiently using the **Newton-Raphson method** and other numerical techniques.  
Built with **Eigen**, it provides a simple and intuitive API for high-performance matrix computations.

## Installation
```bash
git clone https://github.com/samaffleck/Solvex.git
cd Solvex
mkdir build && cd build
cmake ..
make
```

## CMake Usage
```bash
add_subdirectory(Solvex)
target_link_libraries(my_project PRIVATE Solvex)
```

## Dependencies
C++17 or newer required.

## License
Solvex is licensed under the MIT License. See LICENSE for details.

⭐ If you find Solvex useful, please consider starring the repo! 🚀✨
