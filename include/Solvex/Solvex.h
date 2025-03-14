#pragma once

#include "Eigen/Dense"

// DAE-CPP includes
#include "dae-cpp/solver.hpp"

// Solvex includes
#include "Solvex/Equations.h"

namespace Solvex
{
    using namespace daecpp;

    using ODEFunc = std::function<void(double t, const Eigen::VectorXd& x, Eigen::VectorXd& Fx)>;
    
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

    Eigen::VectorXd BDF1Solver(const ODEFunc& f_dxdt,
        const Eigen::VectorXd& x0,
        double startTime,
        double endTime,
        double absolute_tolerance = 1e-10,
        double relative_tolerance = 1e-10,
        int num_of_sup_diag = 1,
        int num_of_sub_diag = 1,
        int max_num_of_newton_itterations = 100,
        int jacobian_update_frequency = 1,
        double newton_relaxation_factor = 1.0);

    Eigen::VectorXd BDF2Solver(const ODEFunc& f_dxdt,
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

    static state_vector getCellCenterVariable(const state_vector& x, SegmentIndex index)
    {
        int N = index.endIndex - index.startIndex;
        if (N <= 1) return state_vector();
    
        state_vector var(N - 1);
        for (int i = 0; i < N - 1; ++i)
        {
            var[i] = x[i + index.startIndex + 1]; // avoids the ghost cells
        }
        return var;
    }
    
    static state_vector getCellFaceVariable(const state_vector& x, SegmentIndex index)
    {
        int N = index.endIndex - index.startIndex + 1;
    
        state_vector var(N);
        for (int i = 0; i < N; ++i)
        {
            var[i] = x[i + index.startIndex]; // avoids the ghost cells
        }
        return var;
    }

} // End ZeroFlux namespace
