// C++ includes
#include <iostream>

#include "dae-cpp/solver.hpp"

// dae-cpp namespace
using namespace daecpp;

namespace FVM
{
    double div(const state_type& x, int i)
    {
        return (x(i + 1) - 2 * x(i) + x(i - 1)).val();
    }
}


struct SegmentIndex
{
    int startIndex;
    int endIndex;
};

struct Equations
{
    SegmentIndex    T{};
    SegmentIndex    P{};
    SegmentIndex    U{};
};

struct MySystem
{
    int_type        Nvar = 3;       // Number of variables to solve for. E.g. temp. and conc.   
    int_type        N = 40;         // Number of cells
    int_type        Nc = 42;        // Number of cells for cell centered variables: N + 2 ghost cells
    int_type        Nf = 41;        // Number of cells for cell face variables: N + 1 
    double          L = 1.0;        // Length of domain [m]
    double          dx = 1.0;       // Cell width [m]
    double          K = 1.0e-9;     // Permeability [m2/s]
    double          vis = 1.0e-5;   // Viscosity [Pa-s]

    double          T0 = 293.0;     // Initial temperature
    double          P0 = 101325.0;  // Initial pressure
    double          U0 = 1.0;       // Initial velocity

    SegmentIndex    T{};
    SegmentIndex    P{};
    SegmentIndex    U{};

    std::vector<state_vector> T_sol{};
    std::vector<state_vector> P_sol{};
    std::vector<state_vector> U_sol{};
    std::vector<state_vector> t_sol{};

    void initialise()
    {
        dx = L / N;

        Nc = N + 2; // add two ghost cells 
        Nf = N + 1; // cell faces

        T.startIndex = 0;
        T.endIndex = T.startIndex + Nc - 1;

        P.startIndex = T.endIndex + 1;
        P.endIndex = P.startIndex + Nc - 1;

        U.startIndex = P.endIndex + 1;
        U.endIndex = U.startIndex + Nf - 1;
    }

    void updateT(state_type& f, const state_type& x) const
    {
        int startIndex = T.startIndex;
        int endIndex = T.endIndex;

        f(startIndex) = 298 - x(startIndex);

        for (int i = startIndex + 1; i < endIndex; ++i)
        {
            f(i) = x(i + 1) - 2 * x(i) + x(i - 1);
        }

        f(endIndex) = 273 - x(endIndex);
    }

    void updateP(state_type& f, const state_type& x) const
    {
        int startIndex = P.startIndex;
        int endIndex = P.endIndex;
        double _dx = 1 / dx;
        double _vis = 1 / vis;
        int u = U.startIndex + 1;

        f(startIndex) = 101425 - x(startIndex);

        for (int i = startIndex + 1; i < endIndex; ++i)
        {
            autodiff::real Pe = 0.5 * (x(i) + x(i + 1));
            autodiff::real Pw = 0.5 * (x(i) + x(i - 1));

            autodiff::real dPdx_e = _dx * (x(i + 1) - x(i));
            autodiff::real dPdx_w = _dx * (x(i) - x(i - 1));

            f(i) = K * _dx * _vis * (dPdx_e * Pe - dPdx_w * Pw);
            //f(i) = _dx * (x[u] * Pe - x[u - 1] * Pw);

            u++;
        }

        f(endIndex) = 101325 - x(endIndex);
    }

    void updateU(state_type& f, const state_type& x) const
    {
        double  _dx = 1 / dx;
        double  _vis = 1 / vis;
        int     p = P.startIndex;
        int     u = U.startIndex;
        int     len = U.endIndex - U.startIndex + 1;

        for (int i = 0; i < len; ++i)
        {
            autodiff::real dPdx = _dx * (x(p + 1) - x(p));

            f(u) = x(u) + K * _vis * dPdx;

            u++;
            p++;
        }
    }

    state_vector getInitialConditions() const
    {
        int Ntot = U.endIndex + 1;
        state_vector x(Ntot);

        for (int i = T.startIndex; i <= T.endIndex; ++i)
        {
            x[i] = T0;
        }

        for (int i = P.startIndex; i <= P.endIndex; ++i)
        {
            x[i] = P0;
        }

        for (int i = U.startIndex; i <= U.endIndex; ++i)
        {
            x[i] = U0;
        }

        return x;
    }

    void setMassMatrix(sparse_matrix& M)
    {
        int Ntot = U.endIndex + 1;
        M.reserve(Ntot);

        for (int i = T.startIndex; i <= T.endIndex; ++i)
        {
            M(i, i, 1.0);
        }

        for (int i = P.startIndex; i <= P.endIndex; ++i)
        {
            M(i, i, 1.0);
        }
    }
};

class MyMassMatrix
{
    MySystem m_system;

public:
    explicit MyMassMatrix(const MySystem& system) : m_system(system) {}

    void operator()(sparse_matrix& M, const double t)
    {
        m_system.setMassMatrix(M);
    }
};

class MyRHS
{
    MySystem m_system;

public:
    explicit MyRHS(const MySystem& system) : m_system(system) {}

    void operator()(state_type& f, const state_type& x, const double t)
    {
        m_system.updateT(f, x);
        m_system.updateP(f, x);
        m_system.updateU(f, x);
    }
};

void printVector(const state_vector& x)
{
    std::cout << "x:\n";
    for (size_t i = 0; i < x.size(); ++i)
    {
        std::cout << "\t" << x[i] << "\n";
    }
    std::cout << "\n";
}

state_vector getCellCenterVariable(const state_vector& x, SegmentIndex index)
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

state_vector getCellFaceVariable(const state_vector& x, SegmentIndex index)
{
    int N = index.endIndex - index.startIndex + 1;

    state_vector var(N);
    for (int i = 0; i < N; ++i)
    {
        var[i] = x[i + index.startIndex]; // avoids the ghost cells
    }
    return var;
}

class MySolutionManager
{
    MySystem& m_system;

public:
    explicit MySolutionManager(MySystem& system) : m_system(system) {}

    int operator()(const state_vector& x, const double t)
    {
        m_system.T_sol.emplace_back(getCellCenterVariable(x, m_system.T));
        m_system.P_sol.emplace_back(getCellCenterVariable(x, m_system.P));
        m_system.U_sol.emplace_back(getCellFaceVariable(x, m_system.U));

        printVector(x);

        return 0;
    }
};

int main()
{
    MySystem system;
    system.N = 10;

    system.initialise();
    state_vector x0 = system.getInitialConditions();

    double t_end = 100.0;
    SolverOptions opt;
    opt.verbosity = verbosity::normal;        // Prints computation time and basic info
    opt.solution_variability_control = false; // Switches off solution variability control for better performance
    opt.BDF_order = 2;
    opt.atol = 1e-10;
    opt.rtol = 1e-10;

    int status = solve(MyMassMatrix(system), MyRHS(system), x0, t_end, MySolutionManager(system), opt);

    return status;
}
