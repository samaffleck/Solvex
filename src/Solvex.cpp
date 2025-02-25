#include "Solvex/Solvex.h"

#include <iostream>

namespace Solvex
{
    
    void approximateJ(std::function<void(const Eigen::VectorXd& x, Eigen::VectorXd& Fx)> f,
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& Fx,
        Eigen::MatrixXd& J,
        int num_of_sub_diag, 
        int num_of_sup_diag,
        int N)
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

    void NewtonItteration(const Eigen::MatrixXd& J,
        const Eigen::VectorXd& Fx, 
        Eigen::VectorXd& x, 
        double relax_factor)
    {
        Eigen::VectorXd dx = J.colPivHouseholderQr().solve(-Fx);
        x += dx * relax_factor;
    }

    void BDF1Residual(std::function<void(const Eigen::VectorXd& x, Eigen::VectorXd& dx_dt)> f_dxdt,
        Eigen::VectorXd& x,
        const Eigen::VectorXd& x_dt,
        Eigen::VectorXd& Fx,
        double dt)
    {
        //****************************************************************************
        //
        //  Purpose:
        //
        //    backwardsDifferenceResidual() evaluates the residual using the backwards 
        //    difference formula.
        //
        //  Discussion:
        //
        //      Using the backwards difference formula:
        //
        //      dxdt = ( x - x_dt ) / ( dt )
        //
        //    This can be rewritten as
        //
        //      residual = x - x_dt - dt * dxdt
        //
        //    A nonlinear equation solver can be used
        //    to estimate the value x that makes the residual zero.
        //
        //  Input:
        //
        //    std::function<void(double t, const Eigen::VectorXd& x, Eigen::VectorXd& dx_dt)> f_dxdt : 
        //    evaluates the right hand side of the ODE.
        //
        //    double dt = time step
        //
        //    Eigen::VectorXd& x_dt: the old time and solution.
        //
        //    Eigen::VectorXd& x: the current solution.
        //
        //  Output:
        //
        //    Eigen::VectorXd& Fx: the residual vector.
        //

        int N = x.size();
        Eigen::VectorXd dx_dt(N); // Create the dx_dt vector
        f_dxdt(x, dx_dt);

        for (int n = 0; n < N; ++n)
        {
            Fx(n) = x(n) - x_dt(n) - dt * dx_dt(n); 
        }
    }

    void BDF1ApproximateJacobian(std::function<void(const Eigen::VectorXd& x, Eigen::VectorXd& dx_dt)> f_dxdt,
        Eigen::VectorXd& x,
        const Eigen::VectorXd& x_dt,
        Eigen::VectorXd& Fx,
        Eigen::MatrixXd& J,
        double dt,
        int num_of_sup_diag,
        int num_of_sub_diag)
    {
        //****************************************************************************
        //
        //  Purpose:
        //
        //    approximateJacobianBDF1() estimates a Jacobian matrix using forward differences.
        //
        //  Discussion:
        //
        //    This function computes a forward-difference approximation
        //    to the N by N jacobian matrix associated with a specified
        //    problem of N functions in N variables that is solved with
        //    the backwards difference formula
        //
        //  Input:
        //
        //    void f_dxdt:
        //    the name of the user-supplied code which
        //    evaluates the right hand side of the ODE.
        //
        //
        //  Output:
        //

        double precision = std::numeric_limits<double>::epsilon();
        double eps = sqrt(precision);
        double h = sqrt(precision);
        int msum = num_of_sub_diag + num_of_sup_diag + 1;
        int N = x.size();
        Eigen::VectorXd del_x = x;      // Intentional copy
        Eigen::VectorXd del_Fx = Fx;    // Intentional copy
        
        // Dense approximate of jacobian
        if ( N <= msum )
        {
            for (int j = 0; j < N; ++j)
            {
                h = std::max( eps, eps * fabs( x(j) ) ); // avoids h = 0 leading to divide by 0 error
                del_x(j) = x(j) + h;
                BDF1Residual(f_dxdt, del_x, x_dt, del_Fx, dt);
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
                BDF1Residual(f_dxdt, del_x, x_dt, del_Fx, dt);
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

    NewtonSolverMessage BDF1NewtonSolver(std::function<void(const Eigen::VectorXd& x, Eigen::VectorXd& dx_dt)> f_dxdt,
        Eigen::VectorXd& x, 
        const Eigen::VectorXd& x_dt, 
        Eigen::VectorXd& Fx, 
        Eigen::MatrixXd& J,
        double dt,
        int num_of_sup_diag,
        int num_of_sub_diag,
        int max_itterations,
        int jac_update_freq, 
        double abs_tol, 
        double rel_tol, 
        double relax_factor)
    {
        bool isConverged = false;
        int itt = 0;
        double error = abs_tol * 10;

        BDF1Residual(f_dxdt, x, x_dt, Fx, dt);
        BDF1ApproximateJacobian(f_dxdt, x, x_dt, Fx, J, dt, num_of_sup_diag, num_of_sub_diag);

        while (!isConverged && itt <= max_itterations)
        {
            BDF1Residual(f_dxdt, x, x_dt, Fx, dt);
            if (itt % jac_update_freq == 0) // update jacobian evey x itterations. By default, it is updated every itteration
                BDF1ApproximateJacobian(f_dxdt, x, x_dt, Fx, J, dt, num_of_sup_diag, num_of_sub_diag);

            NewtonItteration(J, Fx, x, relax_factor);
            double allowedError = abs_tol + rel_tol * x.norm();
            BDF1Residual(f_dxdt, x, x_dt, Fx, dt);
            error = Fx.norm(); 
            if (error < allowedError)
                isConverged = true;
            
            itt++;
        }

        std::string errMsg{};
        if (itt >= max_itterations)
            errMsg += "Convergence failed. Maximum number of itterations exceeded\n";

        return {isConverged, itt + 1, error, errMsg};
    }

    Eigen::VectorXd BFD1Solver(std::function<void(const Eigen::VectorXd& x, Eigen::VectorXd& dx_dt)> f_dxdt, 
        Eigen::VectorXd& x0,
        double startTime,
        double endTime,
        double absolute_tolerance,
        double relative_tolerance,
        int num_of_sup_diag,
        int num_of_sub_diag,
        int max_num_of_newton_itterations,
        int jacobian_update_frequency,
        double newton_relaxation_factor)
    {
        double dt0 = 1.0;
        double dt = dt0;
        double minimum_dt = 1e-6;
        double time = startTime;

        int N = x0.size();
        Eigen::VectorXd x = x0;         // Copy the initial conditions into our solution vector at the current time step
        Eigen::VectorXd x_dt = x0;      // Copy the initial conditions into our solution vector at the previous time step
        Eigen::VectorXd Fx(N);          // This vector stores the residuals
        Eigen::MatrixXd J(N, N);        // (N x N) Jacobian matrix

        while (time < endTime && dt > minimum_dt)
        {
            auto stepResult = BDF1NewtonSolver(f_dxdt, 
                x, 
                x_dt, 
                Fx, 
                J, 
                dt, 
                num_of_sup_diag, 
                num_of_sub_diag, 
                max_num_of_newton_itterations, 
                jacobian_update_frequency, 
                absolute_tolerance, 
                relative_tolerance, 
                newton_relaxation_factor);

            if (stepResult.converged)
            {
                x0 = x;     // Update the solution vector
                time += dt; // Increment step time

                std::cout << "Time = " << time << "\nx = \n" << x << "\n"; 

                dt *= 2;    // On successful steps, double the step size.
            }
            else
            {
                x = x0;     // reset the solution vector
                dt /= 2;    // On unsuccessful steps, halve the time step.
            }
        }

        return x;
    }

        Eigen::VectorXd NLESolver(std::function<void(const Eigen::VectorXd& x, Eigen::VectorXd& Fx)> f, 
        Eigen::VectorXd& x0,
        double absolute_tolerance,
        double relative_tolerance,
        int num_of_sup_diag,
        int num_of_sub_diag,
        int max_num_of_newton_itterations,
        int jacobian_update_frequency,
        double newton_relaxation_factor)
    {
        int N = x0.size();
        Eigen::VectorXd x = x0;         // Copy the initial conditions into our solution vector
        Eigen::VectorXd Fx(N);          // This vector stores the residuals
        Eigen::MatrixXd J(N, N);        // (N x N) Jacobian matrix

        bool isConverged = false;
        int itt = 0;
        double error = absolute_tolerance * 10;

        while (!isConverged && itt <= max_num_of_newton_itterations)
        {
            f(x, Fx);
            if (itt % jacobian_update_frequency == 0) // update jacobian evey x itterations. By default, it is updated every itteration
                approximateJ(f, x, Fx, J, num_of_sub_diag, num_of_sup_diag, N);

            NewtonItteration(J, Fx, x, newton_relaxation_factor);
            double allowedError = absolute_tolerance + relative_tolerance * x.norm();
            f(x, Fx);
            error = Fx.norm(); 
            if (error < allowedError)
                isConverged = true;
            
            itt++;
        }


        auto stepResult = BDF1NewtonSolver(f_dxdt, 
            x, 
            x_dt, 
            Fx, 
            J, 
            dt, 
            num_of_sup_diag, 
            num_of_sub_diag, 
            max_num_of_newton_itterations, 
            jacobian_update_frequency, 
            absolute_tolerance, 
            relative_tolerance, 
            newton_relaxation_factor);


        return x;
    }
}
