// C++ includes
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <memory>

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
    std::ofstream   file{};         // csv file for log data
    double          t0{};           // Initial condition
    SegmentIndex    index{};        // Start and end index

    void log(const state_vector& x, const double t)
    {
        file << t;
        for (int i = 0; i < x.size(); i++)
            file << "," << x[i];
        file << "\n";
    }
};

struct SystemOfEquation
{
    Equation        T{};            // Temperature
    Equation        P{};            // Pressure
    Equation        U{};            // Velocity
    int_type        Nvar = 3;       // Number of variables to solve for. E.g. temp. and conc.   

    void initialise(int_type N, int_type startIndex)
    {
        int Nc = N + 2; // add two ghost cells 
        int Nf = N + 1; // cell faces

        T.index.startIndex = startIndex;
        T.index.endIndex = T.index.startIndex + Nc - 1;

        P.index.startIndex = T.index.endIndex + 1;
        P.index.endIndex = P.index.startIndex + Nc - 1;

        U.index.startIndex = P.index.endIndex + 1;
        U.index.endIndex = U.index.startIndex + Nf - 1;

        T.file.open("T.csv");
        P.file.open("P.csv");
        U.file.open("U.csv");
    }

    void cleanUp()
    {
        T.file.close();
        P.file.close();
        U.file.close();
    }
};

struct ISystemBlock
{
    SystemOfEquation&   eq;
    double              L = 1.0;        // Length of domain [m]
    double              dx = 1.0;       // Cell width [m]
    int_type            N = 40;         // Number of cells

    ISystemBlock(SystemOfEquation& eq) : eq(eq) {}
    virtual ~ISystemBlock() = default;

    void initialise(int_type startIndex)
    {
        dx = L / N;
        eq.initialise(N, startIndex);
    }

    void cleanUp()
    {
        eq.cleanUp();
    }

    void setInitialConditions(int startIndex, state_vector& x) const
    {
        int T_start = startIndex + eq.T.index.startIndex;
        int T_end = startIndex + eq.T.index.endIndex;

        int P_start = startIndex + eq.P.index.startIndex;
        int P_end = startIndex + eq.P.index.endIndex;

        int U_start = startIndex + eq.U.index.startIndex;
        int U_end = startIndex + eq.U.index.endIndex;

        for (int i = T_start; i <= T_end; ++i)
        {
            x[i] = eq.T.t0;
        }

        for (int i = P_start; i <= P_end; ++i)
        {
            x[i] = eq.P.t0;
        }

        for (int i = U_start; i <= U_end; ++i)
        {
            x[i] = eq.U.t0;
        }
    }

    virtual void setMassMatrix(int startIndex, sparse_matrix& M) const = 0;
    virtual void updateT(state_type& f, const state_type& x) const = 0;
    virtual void updateP(state_type& f, const state_type& x) const = 0;
    virtual void updateU(state_type& f, const state_type& x) const = 0;
};

struct PorousMedia : ISystemBlock
{
    double              K = 1.0e-9;     // Permeability [m2/s]
    double              vis = 1.0e-5;   // Viscosity [Pa-s]

    PorousMedia(SystemOfEquation& eq) : ISystemBlock(eq) {}
    ~PorousMedia() = default;

    void setMassMatrix(int startIndex, sparse_matrix& M) const override
    {
        int T_start = startIndex + eq.T.index.startIndex;
        int T_end = startIndex + eq.T.index.endIndex;

        int P_start = startIndex + eq.P.index.startIndex;
        int P_end = startIndex + eq.P.index.endIndex;

        for (int i = T_start; i <= T_end; ++i)
        {
            M(i, i, 1.0);
        }

        for (int i = P_start; i <= P_end; ++i)
        {
            M(i, i, 1.0);
        }
    }

    void updateT(state_type& f, const state_type& x) const override
    {
        int startIndex = eq.T.index.startIndex;
        int endIndex = eq.T.index.endIndex;

        f(startIndex) = autodiff::real(298) - x(startIndex);

        for (int i = startIndex + 1; i < endIndex; ++i)
        {
            f(i) = x(i + 1) - autodiff::real(2) * x(i) + x(i - 1);
        }

        f(endIndex) = autodiff::real(273) - x(endIndex);
    }

    void updateP(state_type& f, const state_type& x) const override
    {
        int startIndex = eq.P.index.startIndex;
        int endIndex = eq.P.index.endIndex;
        autodiff::real _dx = 1 / dx;
        int u = eq.U.index.startIndex + 1;
        autodiff::real K1 = K / (dx * vis);
        
        f(startIndex) = autodiff::real(101425) - x(startIndex);

        for (int i = startIndex + 1; i < endIndex; ++i)
        {
            autodiff::real Pe = autodiff::real(0.5) * (x(i) + x(i + 1));
            autodiff::real Pw = autodiff::real(0.5) * (x(i) + x(i - 1));

            autodiff::real dPdx_e = _dx * (x(i + 1) - x(i));
            autodiff::real dPdx_w = _dx * (x(i) - x(i - 1));

            f(i) = K1 * (dPdx_e * Pe - dPdx_w * Pw);

            u++;
        }

        f(endIndex) = autodiff::real(101325) - x(endIndex);
    }

    void updateU(state_type& f, const state_type& x) const override
    {
        autodiff::real  _dx = 1 / dx;
        autodiff::real  _vis = 1 / vis;
        autodiff::real  K1 = K / vis;
        int     p = eq.P.index.startIndex;
        int     u = eq.U.index.startIndex;
        int     len = eq.U.index.endIndex - eq.U.index.startIndex + 1;

        for (int i = 0; i < len; ++i)
        {
            autodiff::real dPdx = _dx * (x(p + 1) - x(p));

            f(u) = x(u) + K1 * dPdx;

            u++;
            p++;
        }
    }   
};


struct MySystem
{
    std::vector<std::unique_ptr<ISystemBlock>>  m_sys;
    std::unordered_map<std::string, int>        m_map;
    SystemOfEquation                            eq;

    void addBlock(const std::string& name, std::unique_ptr<ISystemBlock> block)
    {
        m_sys.emplace_back(std::move(block));
        m_map[name] = m_sys.size() - 1;
    }

    ISystemBlock& getBlock(const std::string& name)
    {
        return *m_sys[m_map[name]];
    }

    void removeBlock(const std::string& name)
    {
        m_sys.erase(m_sys.begin() + m_map[name]);
        m_map.erase(name);
    }

    void initialise()
    {
        int startIndex = 0;

        for (auto& block : m_sys)
        {
            std::cout << "Initialising block with start index " << startIndex << "\n";
            block->initialise(startIndex);
            startIndex = block->eq.U.index.endIndex + 1;
        }
    }

    void cleanUp()
    {
        for (auto& block : m_sys)
        {
            block->cleanUp();
        }
    }

    void updateT(state_type& f, const state_type& x) const
    {
        for (auto& block : m_sys)
        {
            block->updateT(f, x);
        }
    }

    void updateP(state_type& f, const state_type& x) const
    {
        for (auto& block : m_sys)
        {
            block->updateP(f, x);
        }
    }

    void updateU(state_type& f, const state_type& x) const
    {
        for (auto& block : m_sys)
        {
            block->updateU(f, x);
        }
    }

    state_vector getInitialConditions() const
    {
        size_t x_size = 0;
        for (auto& block : m_sys)
        {
            std::cout << "Getting initial conditions for block with start index " << x_size << "\n";
            x_size += block->eq.U.index.endIndex + 1; // todo: create a function to get the end index from the system block
        }

        std::cout << "Initialising state vector of size " << x_size << "\n";
        state_vector x(x_size);
        int startIndex = 0;

        for (auto& block : m_sys)
        {
            std::cout << "Setting initial conditions for block with start index " << startIndex << "\n";
            block->setInitialConditions(startIndex, x);
            startIndex = block->eq.U.index.endIndex + 1;
        }

        return x;
    }

    void setMassMatrix(sparse_matrix& M) const
    {
        size_t x_size = 0;
        for (auto& block : m_sys)
        {
            x_size += block->eq.U.index.endIndex + 1; // todo: create a function to get the end index from the system block
        }
    
        M.reserve(x_size);
        
        int startIndex = 0;
        for (auto& block : m_sys)
        {
            block->setMassMatrix(startIndex, M);
            startIndex += block->eq.U.index.endIndex + 1; // todo: create a function to get the end index from the system block 
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

class MySolutionManager
{
    SystemOfEquation& m_eq;

public:
    explicit MySolutionManager(SystemOfEquation& eq) : m_eq(eq) {}
    
    int operator()(const state_vector& x, const double t)
    {
        auto Tvals = getCellCenterVariable(x, m_eq.T.index);
        auto Pvals = getCellCenterVariable(x, m_eq.P.index);
        auto Uvals = getCellFaceVariable(x, m_eq.U.index);

        m_eq.T.log(Tvals, t);
        m_eq.P.log(Pvals, t);
        m_eq.U.log(Uvals, t);

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
    // Initialise
    study.m_system.initialise();
    
    // Generate the initial condition vector
    state_vector x0 = study.m_system.getInitialConditions();
    printVector(x0);

    // Solve the system of ODEs
    solve(
        MyMassMatrix(study.m_system), 
        MyRHS(study.m_system), 
        x0, 
        study.m_time, 
        MySolutionManager(study.m_system.eq), 
        study.m_solverOptions
    );

    // Clean up
    study.m_system.cleanUp();
}


int main()
{
    Study study;
    PorousMedia porousMedia1(study.m_system.eq);
    porousMedia1.N = 4;
    porousMedia1.eq.T.t0 = 293.0;
    porousMedia1.eq.P.t0 = 101325.0;
    porousMedia1.eq.U.t0 = 0.0;
    study.m_system.addBlock("PorousMedia1", std::make_unique<PorousMedia>(porousMedia1));

    PorousMedia porousMedia2(study.m_system.eq);
    porousMedia2.N = 8;
    porousMedia2.eq.T.t0 = 293.0;
    porousMedia2.eq.P.t0 = 101325.0;
    porousMedia2.eq.U.t0 = 0.0;
    study.m_system.addBlock("PorousMedia2", std::make_unique<PorousMedia>(porousMedia2));

    study.m_time = 100.0;

    study.m_solverOptions.verbosity = verbosity::normal;        
    study.m_solverOptions.solution_variability_control = false; 
    study.m_solverOptions.BDF_order = 4;
    study.m_solverOptions.atol = 1e-10;
    study.m_solverOptions.rtol = 1e-10;

    RunStudy(study);
}
