#include <iostream>

#include "Solvex/Solvex.h"
#include "Solvex/Equation.h"

void pde(const Eigen::VectorXd& x, 
    Eigen::VectorXd& dxdt)
{
    int N = x.size() - 1;

    dxdt(0) = 100 - 2 * x(0) + x(1);
    for (int n = 1; n < N; ++n)
    {
        //dxdt(n) = x(n - 1) - 2 * x(n) * x(n) + x(n + 1);
        dxdt(n) = x(n - 1) - 2 * x(n) + x(n + 1);
    }
    dxdt(N) = 10 - 2 * x(N) + x(N - 1);
}

int main(int, char**)
{
    Eigen::VectorXd x0(10);
    x0.setConstant(1.0);

    std::cout << "\nInitial conditions: \n" << x0 << "\n";
    Eigen::VectorXd x = Solvex::BFD1Solver(pde, x0, 0, 1000);
    std::cout << "\nFinal solution: \n" << x << "\n";

}
