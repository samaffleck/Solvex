#include "Solvex/Solvex.h"


namespace Solvex
{
    void NewtonItteration(Equation& equation, 
        double relax_factor)
    {
        Eigen::VectorXd dx = equation.J.colPivHouseholderQr().solve(-equation.Fx);
        equation.x += dx * relax_factor;
    }

    NewtonSolverMessage NewtonSolver(Equation& equation, 
        int max_itterations,
        int jac_update_freq, 
        double abs_tol, 
        double rel_tol, 
        double relax_factor)
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

}