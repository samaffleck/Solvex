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
        double relax_factor = 1.0)
    {
        Eigen::VectorXd dx = equation.J.colPivHouseholderQr().solve(-equation.Fx);
        equation.x += dx * relax_factor;
    }

    NewtonSolverMessage NewtonSolver(Equation& equation, 
        int max_itterations = 500,
        int jac_update_freq = 1, 
        double abs_tol = 1e-6, 
        double rel_tol = 1e-6, 
        double relax_factor = 1.0)
    {
        bool isConverged = false;
        int itt = 0;
        double error = abs_tol * 10;

        equation.updateFx();
        equation.approximateJ();

        while (!isConverged && itt <= max_itterations)
        {
            equation.updateFx();
            if (itt % jac_update_freq == 0) // update jacobian evey x itterations. By default, it is updated every itteration
                equation.approximateJ();
            NewtonItteration(equation, relax_factor);
            double allowedError = abs_tol + rel_tol * equation.x.norm();
            error = equation.getError(); 
            if (error < allowedError)
                isConverged = true;
            
            itt++;
        }

        std::string errMsg{};
        if (itt >= max_itterations)
            errMsg += "Convergence failed. Maximum number of itterations exceeded\n";

        return {isConverged, itt + 1, error, errMsg};
    }

} // End ZeroFlux namespace
