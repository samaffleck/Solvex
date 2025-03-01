#include <iostream>

#include "Solvex/Solvex.h"
#include "Solvex/Equation.h"

#include "autodiff/forward/dual/dual.hpp"
#include <autodiff/forward/real/eigen.hpp>

struct ODE
{
    double A = 1.0;
    double B = 2.0;

    void operator()(const double t,
        const Eigen::VectorXd& x,
        Eigen::VectorXd& dxdt)
    {
        dxdt(0) = x(1);
        dxdt(1) = (A/B) * t * x(0);
    }
};

int main(int, char**)
{
    Eigen::VectorXd x0(2);
    x0 << 0.0, 0.01;

    ODE ode;
    double t = 0;

    Eigen::VectorXd fx = x0;
    Eigen::VectorXd dxdt = x0;
    
    std::cout << "\nInitial conditions: \n" << x0 << "\n";
    Eigen::VectorXd x = Solvex::BDF1Solver(ode, x0, 0, 5);
    Eigen::VectorXd bdf2_x = Solvex::BDF2Solver(ode, x0, 0, 5);
    
    std::cout << "\nBDF1 solution: \n" << x << "\n";
    std::cout << "\nBDF1 err\n x[0]: " << 1.11159 - x[0] << "\tx[1]: " << 1.69654 - x[1] << "\n";

    std::cout << "\nBDF2 solution: \n" << bdf2_x << "\n";
    std::cout << "\nBDF2 err\n x[0]: " << 1.11159 - bdf2_x[0] << "\tx[1]: " << 1.69654 - bdf2_x[1] << "\n";
    
}
