#pragma once

#include "Eigen/Dense"
#include "Equation.h"


namespace Solvex
{
    using Func = std::function<void(const Eigen::VectorXd& x, Eigen::VectorXd& Fx)>;

    struct NewtonSolverMessage
    {
        bool converged = false;
        int num_of_itterations = 0;
        double error = 0.0;
        std::string errorMessage{};
    };

    void TDMA(const Eigen::MatrixXd& A, 
        const Eigen::VectorXd& y, 
        Eigen::VectorXd& x);

    Eigen::VectorXd BFD1Solver(const Func& f_dxdt,
        const Eigen::VectorXd& x0,
        double startTime,
        double endTime,
        double absolute_tolerance = 1e-6,
        double relative_tolerance = 1e-6,
        int num_of_sup_diag = 1,
        int num_of_sub_diag = 1,
        int max_num_of_newton_itterations = 500,
        int jacobian_update_frequency = 1,
        double newton_relaxation_factor = 1.0);

    Eigen::VectorXd BFD2Solver(const Func& f_dxdt,
        const Eigen::VectorXd& x0,
        double startTime,
        double endTime,
        double absolute_tolerance = 1e-6,
        double relative_tolerance = 1e-6,
        int num_of_sup_diag = 1,
        int num_of_sub_diag = 1,
        int max_num_of_newton_itterations = 500,
        int jacobian_update_frequency = 1,
        double newton_relaxation_factor = 1.0);


    Eigen::VectorXd NLESolver(const Func& f,
        const Eigen::VectorXd& x0,
        double absolute_tolerance = 1e-6,
        double relative_tolerance = 1e-6,
        int num_of_sup_diag = 1,
        int num_of_sub_diag = 1,
        int max_num_of_newton_itterations = 500,
        int jacobian_update_frequency = 1,
        double newton_relaxation_factor = 1.0);

} // End ZeroFlux namespace
