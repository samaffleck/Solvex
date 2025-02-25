#pragma once

#include "eigen3/Eigen/Dense"
#include "Equation.h"


namespace Solvex
{
    struct NewtonSolverMessage
    {
        bool converged = false;
        int num_of_itterations = 0;
        double error = 0.0;
        std::string errorMessage{};
    };

    Eigen::VectorXd BFD1Solver(std::function<void(const Eigen::VectorXd& x, Eigen::VectorXd& dx_dt)> f_dxdt, 
        Eigen::VectorXd& x0,
        double startTime,
        double endTime,
        double absolute_tolerance = 1e-6,
        double relative_tolerance = 1e-6,
        int num_of_sup_diag = 1,
        int num_of_sub_diag = 1,
        int max_num_of_newton_itterations = 500,
        int jacobian_update_frequency = 1,
        double newton_relaxation_factor = 1.0);

    Eigen::VectorXd NLESolver(std::function<void(const Eigen::VectorXd& x, Eigen::VectorXd& Fx)> f, 
        Eigen::VectorXd& x0,
        double absolute_tolerance = 1e-6,
        double relative_tolerance = 1e-6,
        int num_of_sup_diag = 1,
        int num_of_sub_diag = 1,
        int max_num_of_newton_itterations = 500,
        int jacobian_update_frequency = 1,
        double newton_relaxation_factor = 1.0);

} // End ZeroFlux namespace
