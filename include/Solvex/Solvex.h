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

    void NewtonItteration(Equation& equation, 
        double relax_factor = 1.0);

    NewtonSolverMessage NewtonSolver(Equation& equation, 
        int max_itterations = 500,
        int jac_update_freq = 1, 
        double abs_tol = 1e-6, 
        double rel_tol = 1e-6, 
        double relax_factor = 1.0);
    
} // End ZeroFlux namespace
