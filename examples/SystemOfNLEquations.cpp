// C++ includes
#include <iostream>
#include <fstream>

// DAE-CPP includes
#include "dae-cpp/solver.hpp"

// dae-cpp namespace
using namespace daecpp;

struct SegmentIndex
{
    int             startIndex{};
    int             endIndex{};
};

struct Equation
{
    SegmentIndex    index{};        // Start and end index
    double          t0{};           // Initial condition
};

struct SystemOfEquation
{
    Equation        T{};            // Temperature
    Equation        P{};            // Pressure
    Equation        U{};            // Velocity
};

struct MySystem
{
    int_type        Nvar = 3;       // Number of variables to solve for. E.g. temp. and conc.   
    int_type        N = 40;         // Number of cells
    double          L = 1.0;        // Length of domain [m]
    double          dx = 1.0;       // Cell width [m]
    double          K = 1.0e-9;     // Permeability [m2/s]
    double          vis = 1.0e-5;   // Viscosity [Pa-s]

    SystemOfEquation eq;

    void initialise()
    {
        dx = L / N;

        int Nc = N + 2; // add two ghost cells 
        int Nf = N + 1; // cell faces

        eq.T.index.startIndex = 0;
        eq.T.index.endIndex = eq.T.index.startIndex + Nc - 1;

        eq.P.index.startIndex = eq.T.index.endIndex + 1;
        eq.P.index.endIndex = eq.P.index.startIndex + Nc - 1;

        eq.U.index.startIndex = eq.P.index.endIndex + 1;
        eq.U.index.endIndex = eq.U.index.startIndex + Nf - 1;
    }

    void updateT(state_type& f, const state_type& x) const
    {
        int startIndex = eq.T.index.startIndex;
        int endIndex = eq.T.index.endIndex;

        f(startIndex) = 298 - x(startIndex);

        for (int i = startIndex + 1; i < endIndex; ++i)
        {
            f(i) = x(i + 1) - 2 * x(i) + x(i - 1);
        }

        f(endIndex) = 273 - x(endIndex);
    }

    void updateP(state_type& f, const state_type& x) const
    {
        int startIndex = eq.P.index.startIndex;
        int endIndex = eq.P.index.endIndex;
        double _dx = 1 / dx;
        double _vis = 1 / vis;
        int u = eq.U.index.startIndex + 1;

        f(startIndex) = 101425 - x(startIndex);

        for (int i = startIndex + 1; i < endIndex; ++i)
        {
            autodiff::real Pe = 0.5 * (x(i) + x(i + 1));
            autodiff::real Pw = 0.5 * (x(i) + x(i - 1));

            autodiff::real dPdx_e = _dx * (x(i + 1) - x(i));
            autodiff::real dPdx_w = _dx * (x(i) - x(i - 1));

            f(i) = K * _dx * _vis * (dPdx_e * Pe - dPdx_w * Pw);

            u++;
        }

        f(endIndex) = 101325 - x(endIndex);
    }

    void updateU(state_type& f, const state_type& x) const
    {
        double  _dx = 1 / dx;
        double  _vis = 1 / vis;
        int     p = eq.P.index.startIndex;
        int     u = eq.U.index.startIndex;
        int     len = eq.U.index.endIndex - eq.U.index.startIndex + 1;

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
        int Ntot = eq.U.index.endIndex + 1;
        state_vector x(Ntot);

        for (int i = eq.T.index.startIndex; i <= eq.T.index.endIndex; ++i)
        {
            x[i] = eq.T.t0;
        }

        for (int i = eq.P.index.startIndex; i <= eq.P.index.endIndex; ++i)
        {
            x[i] = eq.P.t0;
        }

        for (int i = eq.U.index.startIndex; i <= eq.U.index.endIndex; ++i)
        {
            x[i] = eq.U.t0;
        }

        return x;
    }

    void setMassMatrix(sparse_matrix& M) const
    {
        int Ntot = eq.U.index.endIndex + 1;
        M.reserve(Ntot);

        for (int i = eq.T.index.startIndex; i <= eq.T.index.endIndex; ++i)
        {
            M(i, i, 1.0);
        }

        for (int i = eq.P.index.startIndex; i <= eq.P.index.endIndex; ++i)
        {
            M(i, i, 1.0);
        }
    }
};

class MyMassMatrix
{
    MySystem& m_system;

public:
    explicit MyMassMatrix(MySystem& system) : m_system(system) {}

    void operator()(sparse_matrix& M, const double t) const
    {
        m_system.setMassMatrix(M);
    }
};

class MyRHS
{
    MySystem& m_system;

public:
    explicit MyRHS(MySystem& system) : m_system(system) {}

    void operator()(state_type& f, const state_type& x, const double t) const
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

struct VariableFiles
{
    VariableFiles()
    {
        m_fileT.open("T.csv");
        m_fileP.open("P.csv");
        m_fileU.open("U.csv");
    }
    ~VariableFiles()
    {

        m_fileT.close();
        m_fileP.close();
        m_fileU.close();
    }

    std::ofstream m_fileT{};
    std::ofstream m_fileP{};
    std::ofstream m_fileU{};
};

class MySolutionManager
{
    SystemOfEquation& m_eq;
    VariableFiles& m_files;

public:
    MySolutionManager(SystemOfEquation& eq, VariableFiles& files) : m_eq(eq), m_files(files) {}
    
    int operator()(const state_vector& x, const double t)
    {
        auto Tvals = getCellCenterVariable(x, m_eq.T.index);
        auto Pvals = getCellCenterVariable(x, m_eq.P.index);
        auto Uvals = getCellFaceVariable(x, m_eq.U.index);

        m_files.m_fileT << t;           
        for (int i = 0; i < Tvals.size(); i++)
            m_files.m_fileT << "," << Tvals[i];
        m_files.m_fileT << "\n";
        
        m_files.m_fileP << t;
        for (int i = 0; i < Pvals.size(); i++)
            m_files.m_fileP << "," << Pvals[i];
        m_files.m_fileP << "\n";
        
        m_files.m_fileU << t;
        for (int i = 0; i < Uvals.size(); i++)
            m_files.m_fileU << "," << Uvals[i];
        m_files.m_fileU << "\n";

        return 0;
    }
};


struct Study
{
    MySystem        m_system;
    SolverOptions   m_solverOptions;
    double          m_time;
};


void RunStudy(Study& study)
{
    study.m_system.initialise();

    state_vector x0 = study.m_system.getInitialConditions();
    VariableFiles files;

    solve(
        MyMassMatrix(study.m_system), 
        MyRHS(study.m_system), 
        x0, 
        study.m_time, 
        MySolutionManager(study.m_system.eq, files), 
        study.m_solverOptions
    );
}


int main()
{
    Study study;
    study.m_system.N = 10;
    study.m_system.eq.T.t0 = 293.0;
    study.m_system.eq.P.t0 = 101325.0;
    study.m_system.eq.U.t0 = 0.0;

    study.m_time = 100.0;

    study.m_solverOptions.verbosity = verbosity::normal;        
    study.m_solverOptions.solution_variability_control = false; 
    study.m_solverOptions.BDF_order = 4;
    study.m_solverOptions.atol = 1e-10;
    study.m_solverOptions.rtol = 1e-10;

    RunStudy(study);
}
