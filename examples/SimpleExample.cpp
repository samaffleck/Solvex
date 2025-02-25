#include <iostream>

#include "Solvex/Solvex.h"
#include "Solvex/Equation.h"


struct ODE
{
    double topBC    = 100;
    double bottomBC = 10;

    void operator()(const Eigen::VectorXd& x,
                    Eigen::VectorXd& dxdt) const
    {
        int N = x.size() - 1;

        dxdt(0) = topBC - x(0);
        for (int n = 1; n < N; ++n)
        {
            dxdt(n) = x(n - 1) - 2 * x(n) * x(n) + x(n + 1);
        }
        dxdt(N) = bottomBC - x(N);
    }
};

int main(int, char**)
{
    Eigen::VectorXd x0(10);
    x0.setConstant(1.0);

    ODE ode;

    std::cout << "\nInitial conditions: \n" << x0 << "\n";
    //Eigen::VectorXd x = Solvex::BFD1Solver(pde, x0, 0, 100);
    Eigen::VectorXd x = Solvex::NLESolver(ode, x0);
    std::cout << "\nFinal solution: \n" << x << "\n";
}
