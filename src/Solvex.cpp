#include "Solvex/Solvex.h"

#include "autodiff/forward/dual/dual.hpp"
#include <autodiff/forward/real/eigen.hpp>

#include <iostream>

namespace Solvex
{
    
    void TDMA(const Eigen::MatrixXd& A, 
        const Eigen::VectorXd& y, 
        Eigen::VectorXd& x)
    {
        // Solves Ax = y, where A is tri-diagonal
        int N = x.size();
        Eigen::VectorXd c(N);
        Eigen::VectorXd d(N);

        c(0) = A(0, 1) / A(0, 0);
        d(0) = y(0) / A(0, 0);

        for (int i = 1; i < N - 1; ++i)
        {
            double denom = A(i, i) - A(i, i - 1) * c(i - 1);
            c(i) = A(i, i + 1) / denom;
            d(i) = (y(i) - A(i, i - 1) * d(i - 1)) / denom;
        }

        d(N - 1) = (y(N - 1) - A(N - 1, N - 2) * d(N - 2)) / (A(N - 1, N - 1) - A(N - 1, N - 2) * c(N - 2));

        x(N - 1) = d(N - 1);
        for (int i = N - 2; i >= 0; i--)
        {
            x(i) = d(i) - c(i) * x(i + 1);
        }
    }

    void approximateJ(const Func& f,
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

    void NewtonItterationTDMA(const Eigen::MatrixXd& J,
        const Eigen::VectorXd& Fx,
        Eigen::VectorXd& x,
        double relax_factor)
    {
        Eigen::VectorXd dx(x.size());
        TDMA(J, -Fx, dx);
        x += dx * relax_factor;
    }

    void NewtonItteration(const Eigen::MatrixXd& J,
        const Eigen::VectorXd& Fx, 
        Eigen::VectorXd& x, 
        double relax_factor)
    {
        Eigen::VectorXd dx = J.fullPivLu().solve(-Fx);
        //Eigen::VectorXd dx1 = J.llt().solve(-Fx);
        //Eigen::VectorXd dx2 = J.ldlt().solve(-Fx);
        //Eigen::VectorXd dx3 = J.householderQr().solve(-Fx);

        x += dx * relax_factor;
    }

    void BDF1Residual(const ODEFunc& f_dxdt,
        Eigen::VectorXd& x,
        const Eigen::VectorXd& x_dt,
        Eigen::VectorXd& Fx,
        double time,
        double dt)
    {
        int N = x.size();
        Eigen::VectorXd dx_dt(N); // Create the dx_dt vector
        f_dxdt(time, x, dx_dt);

        for (int n = 0; n < N; ++n)
        {
            Fx(n) = x(n) - x_dt(n) - dt * dx_dt(n); 
        }
    }
    

    void BDF2Residual(const ODEFunc& f_dxdt,
        Eigen::VectorXd& x,
        const Eigen::VectorXd& x_dt,
        const Eigen::VectorXd& x_dt2,
        Eigen::VectorXd& Fx,
        double time,
        double dt,
        double dt2)
    {
        int N = x.size();
        Eigen::VectorXd dx_dt(N); // Create the dx_dt vector
        f_dxdt(time, x, dx_dt);

        double w = ( dt ) / ( dt2 );
        double b = ( 1.0 + w ) * ( 1.0 + w ) / ( 1.0 + 2.0 * w );
        double c = w * w / ( 1.0 + 2.0 * w );
        double d = ( 1.0 + w ) / ( 1.0 + 2.0 * w );

        for (int n = 0; n < N; ++n)
        {
            Fx(n) = x(n) - b * x_dt(n) + c * x_dt2(n) - d * dt * dx_dt(n); 
        }
    }

    void BDF1ApproximateJacobian(const ODEFunc& f_dxdt,
        Eigen::VectorXd& x,
        const Eigen::VectorXd& x_dt,
        Eigen::VectorXd& Fx,
        Eigen::MatrixXd& J,
        double time,
        double dt,
        int num_of_sup_diag,
        int num_of_sub_diag)
    {
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
                h = std::max(1e-8, 1e-6 * fabs(x(j)));
                //h = std::max( eps, eps * fabs( x(j) ) ); // avoids h = 0 leading to divide by 0 error
                del_x(j) = x(j) + h;
                BDF1Residual(f_dxdt, del_x, x_dt, del_Fx, time, dt);
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
                BDF1Residual(f_dxdt, del_x, x_dt, del_Fx, time, dt);
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

    void BDF2ApproximateJacobian(const ODEFunc& f_dxdt,
        Eigen::VectorXd& x,
        const Eigen::VectorXd& x_dt,
        const Eigen::VectorXd& x_dt2,
        Eigen::VectorXd& Fx,
        Eigen::MatrixXd& J,
        double time,
        double dt,
        double dt2,
        int num_of_sup_diag,
        int num_of_sub_diag)
    {
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
                BDF2Residual(f_dxdt, del_x, x_dt, x_dt2, del_Fx, time, dt, dt2);
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
                BDF2Residual(f_dxdt, del_x, x_dt, x_dt2, del_Fx, time, dt, dt2);
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

    NewtonSolverMessage BDF1NewtonSolver(const ODEFunc& f_dxdt,
        Eigen::VectorXd& x, 
        const Eigen::VectorXd& x_dt, 
        Eigen::VectorXd& Fx, 
        Eigen::MatrixXd& J,
        double time,
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
        int rank = 1 + num_of_sub_diag + num_of_sup_diag;

        while (!isConverged && itt <= max_itterations)
        {
            BDF1Residual(f_dxdt, x, x_dt, Fx, time, dt);
            if (itt % jac_update_freq == 0) // update jacobian evey x itterations. By default, it is updated every itteration
            {
                BDF1ApproximateJacobian(f_dxdt, x, x_dt, Fx, J, time, dt, num_of_sup_diag, num_of_sub_diag);
            }
            
            NewtonItteration(J, Fx, x, relax_factor);
            
            double allowedError = abs_tol + rel_tol * x.norm();
            BDF1Residual(f_dxdt, x, x_dt, Fx, time, dt);
            error = Fx.norm(); 
            if (error < allowedError)
                isConverged = true;
            
            itt++;
        }

        std::cout << "\nNumber of itterations = " << itt;
        std::cout << "\nError = " << error;

        std::string errMsg{};
        if (itt >= max_itterations)
            errMsg += "Convergence failed. Maximum number of itterations exceeded\n";

        return {isConverged, itt + 1, error, errMsg};
    }

    NewtonSolverMessage BDF2NewtonSolver(const ODEFunc& f_dxdt,
        Eigen::VectorXd& x, 
        const Eigen::VectorXd& x_dt, 
        const Eigen::VectorXd& x_dt2, 
        Eigen::VectorXd& Fx, 
        Eigen::MatrixXd& J,
        double time,
        double dt,
        double dt2,
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
        int rank = 1 + num_of_sub_diag + num_of_sup_diag;

        while (!isConverged && itt <= max_itterations)
        {
            BDF2Residual(f_dxdt, x, x_dt, x_dt2, Fx, time, dt, dt2);
            if (itt % jac_update_freq == 0) // update jacobian evey x itterations. By default, it is updated every itteration
                BDF2ApproximateJacobian(f_dxdt, x, x_dt, x_dt2, Fx, J, time, dt, dt2, num_of_sup_diag, num_of_sub_diag);

            NewtonItteration(J, Fx, x, relax_factor);
            
            double allowedError = abs_tol + rel_tol * x.norm();
            BDF2Residual(f_dxdt, x, x_dt, x_dt2, Fx, time, dt, dt2);
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

    Eigen::VectorXd BDF1Solver(const ODEFunc& f_dxdt,
        const Eigen::VectorXd& x0,
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
        double dt0 = 0.1;
        double dt = dt0;
        double minimum_dt = 1e-6;
        double time = startTime;

        int N = x0.size();
        Eigen::VectorXd x = x0;         // Copy the initial conditions into our solution vector at the current time step
        Eigen::VectorXd x_dt = x0;      // Copy the initial conditions into our solution vector at the previous time step
        Eigen::VectorXd Fx(N);          // This vector stores the residuals
        Eigen::MatrixXd J(N, N);        // (N x N) Jacobian matrix
        J.setZero();
        Fx.setZero();

        while (time < endTime && dt > minimum_dt)
        {
            if (time + dt > endTime)
            {
                dt = (endTime - time);
            }

            auto stepResult = BDF1NewtonSolver(f_dxdt, 
                x, 
                x_dt, 
                Fx, 
                J,
                time + dt,
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
                x_dt = x;       // Update the solution vector
                time += dt;     // Increment step time

                std::cout << "\nTime = " << time << "\ndt = " << dt << "\n"; 

                //dt *= 2;    // On successful steps, double the step size.
            }
            else
            {
                std::cout << "\n\nFailed.\n";
                return x;
                //x = x_dt;       // reset the solution vector
                //dt /= 2;        // On unsuccessful steps, halve the time step.
            }
        }

        return x;
    }

    Eigen::VectorXd BDF2Solver(const ODEFunc& f_dxdt,
        const Eigen::VectorXd& x0,
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
        double dt0 = 0.1;
        double dt = dt0;
        double dt2 = dt0;
        double minimum_dt = 1e-6;
        double time = startTime;

        int N = x0.size();
        Eigen::VectorXd x = x0;         // Copy the initial conditions into our solution vector at the current time step
        Eigen::VectorXd x_dt = x0;      // Copy the initial conditions into our solution vector at the previous time step
        Eigen::VectorXd x_dt2 = x0;      // Copy the initial conditions into our solution vector at the previous time step
        Eigen::VectorXd Fx(N);          // This vector stores the residuals
        Eigen::MatrixXd J(N, N);        // (N x N) Jacobian matrix
        J.setZero();
        Fx.setZero();

        // For the first step use BDF1
        x = BDF1Solver(f_dxdt, x0, startTime, dt, absolute_tolerance, relative_tolerance, num_of_sup_diag, num_of_sub_diag, max_num_of_newton_itterations, jacobian_update_frequency, newton_relaxation_factor);
        x_dt = x;
        time += dt;

        while (time < endTime && dt > minimum_dt)
        {
            if (time + dt > endTime)
            {
                dt = (endTime - time);
            }

            auto stepResult = BDF2NewtonSolver(f_dxdt, 
                x, 
                x_dt, 
                x_dt2, 
                Fx, 
                J, 
                time + dt,
                dt, 
                dt2, 
                num_of_sup_diag, 
                num_of_sub_diag, 
                max_num_of_newton_itterations, 
                jacobian_update_frequency, 
                absolute_tolerance, 
                relative_tolerance, 
                newton_relaxation_factor);

            if (stepResult.converged)
            {
                // Update the solution vector and time steps
                x_dt2 = x_dt;
                x_dt = x;
                dt2 = dt;

                time += dt;     // Increment step time

                std::cout << "Time = " << time << "\ndt = \n" << dt << "\n"; 

                //dt *= 2;        // On successful steps, double the step size.
            }
            else
            {
                std::cout << "\n\nFailed.\n";
                return x;
                //x = x_dt;       // reset the solution vector
                //dt /= 2;        // On unsuccessful steps, halve the time step.
            }
        }

        return x;
    }

    Eigen::VectorXd NLESolver(const Func& f,
        const Eigen::VectorXd& x0,
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

        return x;
    }

} // End Solvex namespace
