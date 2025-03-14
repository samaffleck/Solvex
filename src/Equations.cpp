#include "Solvex/Equations.h"

namespace Solvex
{
    void Equation::log(const state_vector& x, const double t)
    {
        file << t;
        for (int i = 0; i < x.size(); i++)
            file << "," << x[i];
        file << "\n";
    }

    size_t Equation::getSize() const
    {
        return (index.endIndex - index.startIndex + 1);
    }

    void Equation::setInitialCondition(state_vector& x) const
    {
        for (int i = index.startIndex; i <= index.endIndex; ++i)
        {
            x[i] = t0;
        }
    }

    void Equation::setMassMatrix(sparse_matrix& M) const
    {
        for (int i = index.startIndex; i <= index.endIndex; ++i)
        {
            M(i, i, 1.0);
        }
    }

    size_t SystemOfEquation::getSize() const
    {
        size_t size = 0;
        size += T.getSize();
        size += P.getSize();
        size += U.getSize();
        return size;
    }

    void SystemOfEquation::initialise(const std::string& name_id)
    {
        T.file.open(name_id + "_T.csv");
        P.file.open(name_id + "_P.csv");
        U.file.open(name_id + "_U.csv");
    }

    void SystemOfEquation::cleanUp()
    {
        T.file.close();
        P.file.close();
        U.file.close();
    }

} // End Solvex namespace
