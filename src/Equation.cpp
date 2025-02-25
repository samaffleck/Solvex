#include "Solvex/Equation.h"

namespace Solvex
{
    
    Equation::Equation(int _N) : N(_N)
    {
        J.resize(N, N);
        x.resize(N);
        Fx.resize(N);
    }

    Solvex::ODEquation::ODEquation(int _N) : N(_N)
    {
        J.resize(N, N);
        x.resize(N);
        x_dt.resize(N);
        Fx.resize(N);
        dx_dt.resize(N);
    }

} // end Solvex namespace
