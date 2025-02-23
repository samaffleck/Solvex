#include <iostream>

#include "Solvex/Solvex.h"
#include "Solvex/Equation.h"

void pde(const Eigen::VectorXd& x, 
    Eigen::VectorXd& Fx)
{
    int N = x.size() - 1;

    Fx(0) = 100 - 2 * x(0) + x(1);
    for (int n = 1; n < N; ++n)
    {
        Fx(n) = x(n - 1) - 2 * x(n) * x(n) + x(n + 1);
        //Fx(n) = x(n - 1) - 2 * x(n) + x(n + 1);
    }
    Fx(N) = 20 - 2 * x(N) + x(N - 1);
}

int main(int, char**)
{
    Solvex::Equation my_equation(10); 
    my_equation.x.setOnes();
    my_equation.f = pde;

    std::cout << "\nInitial conditions: \n" << my_equation.x << "\n";
    auto error = Solvex::NewtonSolver(my_equation);
    std::cout << "\nFinal solution: \n" << my_equation.x << "\n";

    std::cout << "Converged? \t" << error.converged;
    std::cout << "\nNumber of itterations: \t" << error.num_of_itterations;
    std::cout << "\nFinal error: \t" << error.error;

}
