#pragma once

#include "eigen3/Eigen/Dense"


namespace Solvex
{
    struct Equation
{
    Equation(int _N) : N(_N)
    {
        J.resize(N, N);
        x.resize(N);
        Fx.resize(N);
    }
    ~Equation() = default;

    int N;
    int num_of_sub_diag = 1;
    int num_of_sup_diag = 1;
    Eigen::MatrixXd J;
    Eigen::VectorXd x;
    Eigen::VectorXd Fx;
    std::function<void(const Eigen::VectorXd& x, Eigen::VectorXd& Fx)> f;

    void updateFx()
    {
        f(x, Fx);
    }

    double getError()
    {
        f(x, Fx);
        return Fx.norm();
    }

    void approximateJ()
    {
        //
        // Function adapted from fsolve : https://people.sc.fsu.edu/~jburkardt/cpp_src/fsolve/fsolve.cpp
        //
        
        double precision = std::numeric_limits<double>::epsilon();
        double eps = sqrt(precision);
        double h = sqrt(precision);
        int msum = num_of_sub_diag + num_of_sup_diag + 1;
        Eigen::VectorXd del_x = x;
        Eigen::VectorXd del_Fx = Fx;

        // Dense approximate of jacobian
        if ( N <= msum )
        {
            for (int j = 0; j < N; ++j)
            {
                h = std::max( eps, eps * fabs( x(j) ) ); // avoids h = 0 leading to divide by 0 error
                del_x(j) = x(j) + h;
                f(del_x, del_Fx);
                del_x(j) = x(j); // reset
                for (int i = 0; i < N; ++i)
                {
                    J(i, j) = (del_Fx(i) - Fx(i)) / h;
                }
            }
        }
        else // Banded computation
        {
            for (int k = 0; k < msum; ++k)
            {
                for (int j = k; j < N; j += msum)
                {
                    h = std::max( eps, eps * fabs( x(j) ) ); // avoids h = 0 leading to divide by 0 error
                    del_x(j) = x(j) + h;
                }
                f(del_x, del_Fx);
                for (int j = k; j < N; j += msum)
                {
                    del_x(j) = x(j); // reset

                    // Loop through all equations
                    for (int n = 0; n < N; ++n)
                    {
                        if (j - num_of_sub_diag <= n && n <= j + num_of_sup_diag)
                        {
                            h = std::max( eps, eps * fabs( x(j) ) ); // avoids h = 0 leading to divide by 0 error            
                            J(n, j) = (del_Fx(n) - Fx(n)) / h;
                        }
                        else
                        {
                            J(n, j) = 0.0;
                        }
                    }
                }
            }
        }
    }
    
};
} // End Solvex namespace
