// C++ includes
#include <unordered_map>
#include <memory>

// DAE-CPP includes
#include "dae-cpp/solver.hpp"

// Solvex includes
#include "Solvex/BoundaryConditions.h"
#include "Solvex/Equations.h"
#include "Solvex/Variables.h"
#include "Solvex/Solvex.h"

// namespace
using namespace daecpp;
using namespace Solvex;


struct DiffusionCoefficient : VariableInTime
{
    SystemOfEquation& m_eq;

    explicit DiffusionCoefficient(SystemOfEquation& equations) : m_eq(equations) {}
    ~DiffusionCoefficient() final = default;

    void update(double t) override
    {
        m_var = 10 * (1 / (t + 1));
    }
};

struct DiffusionCoefficient1D : VariableInTime_1D
{
    SystemOfEquation& m_eq;

    explicit DiffusionCoefficient1D(SystemOfEquation& equations) : m_eq(equations) {}
    ~DiffusionCoefficient1D() final = default;

    void update(const state_type& x, double t) override
    {
        int T = m_eq.T.startIndex;
        double T_ref = 298;

        for (int i = 0; i < m_var.size(); ++i)
        {
            m_var[i] = 1e-3 * (x[T + i] / T_ref);
        }
    }
};


struct ISystemBlock
{
    SystemOfEquation        eq;
    std::string         m_name{};
    double              L = 1.0;        // Length of domain [m]
    double              dx = 1.0;       // Cell width [m]
    int_type            N = 40;         // Number of cells

    size_t getSize() const
    {
        return eq.getSize();
    }

    void log(const state_vector& x, double t)
    {
        eq.log(x, t);
    }

    explicit ISystemBlock(const std::string& name) : m_name(name) {}
    virtual ~ISystemBlock() = default;
    virtual void initialise() = 0;
    virtual void cleanUp() = 0;
    virtual void setInitialConditions(state_vector& x) const = 0;
    virtual void setMassMatrix(sparse_matrix& M) const = 0;
    virtual void updateT(state_type& f, const state_type& x) const = 0;
    virtual void updateP(state_type& f, const state_type& x) const = 0;
    virtual void updateU(state_type& f, const state_type& x) const = 0;
    virtual void updateVariables(const state_type& x, double t) = 0;
};


struct PorousMedia : ISystemBlock
{
    DiffusionCoefficient    D;
    DiffusionCoefficient1D  D1;
    double                  K = 1.0e-9;     // Permeability [m2/s]
    double                  vis = 1.0e-5;   // Viscosity [Pa-s]

    explicit PorousMedia(const std::string& name) : ISystemBlock(name), D(eq), D1(eq) {}
    
    ~PorousMedia() final = default;

    void initialise() override
    {
        dx = L / N;
        eq.initialise(m_name);
        D1.initialise(N + 2);
    }

    void cleanUp() override
    {
        eq.cleanUp();
    }

    void setInitialConditions(state_vector& x) const override
    {
        eq.T.setInitialCondition(x);
        eq.P.setInitialCondition(x);
        eq.U.setInitialCondition(x);
    }
    
    void setMassMatrix(sparse_matrix& M) const override
    {
        eq.T.setMassMatrix(M);
        eq.P.setMassMatrix(M);
    }

    void updateT(state_type& f, const state_type& x) const override
    {
        int startIndex = eq.T.startIndex;
        int endIndex = eq.T.endIndex;

        eq.T.bot_bc->apply(f, x, startIndex, dx);

        for (int i = startIndex + 1; i < endIndex; ++i)
        {
            f(i) = x(i + 1) - autodiff::real(2) * x(i) + x(i - 1);
        }

        eq.T.top_bc->apply(f, x, endIndex, dx);
    }

    void updateP(state_type& f, const state_type& x) const override
    {
        int startIndex = eq.P.startIndex;
        int endIndex = eq.P.endIndex;
        autodiff::real _dx = 1 / dx;
        int u = eq.U.startIndex + 1;
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
        autodiff::real  K1 = K / vis;
        int     p = eq.P.startIndex;
        int     u = eq.U.startIndex;
        int     len = eq.U.endIndex - eq.U.startIndex + 1;

        for (int i = 0; i < len; ++i)
        {
            autodiff::real dPdx = _dx * (x(p + 1) - x(p));

            f(u) = x(u) + K1 * dPdx;

            u++;
            p++;
        }
    }

    void updateVariables(const state_type& x, double t) override
    {
        D.update(t);
        D1.update(x, t);
    }
};


struct MySystem
{
    std::vector<std::unique_ptr<ISystemBlock>>  m_sys;
    std::unordered_map<std::string, int>        m_map;

    void addBlock(std::unique_ptr<ISystemBlock> block)
    {
        m_map[block->m_name] = m_sys.size() - 1;
        m_sys.emplace_back(std::move(block));
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

    void initialise() const
    {
        int start_index = 0;

        for (const auto& block : m_sys)
        {
            block->eq.T.startIndex = start_index;
            block->eq.T.endIndex = start_index + block->N + 1;
            start_index = block->eq.T.endIndex + 1;
        }
        
        for (const auto& block : m_sys)
        {
            block->eq.P.startIndex = start_index;
            block->eq.P.endIndex = start_index + block->N + 1;
            start_index = block->eq.P.endIndex + 1;
        }

        for (const auto& block : m_sys)
        {
            block->eq.U.startIndex = start_index;
            block->eq.U.endIndex = start_index + block->N;
            start_index = block->eq.U.endIndex + 1;
        }

        for (const auto& block : m_sys)
        {
            block->initialise();
        }
    }

    void cleanUp() const
    {
        for (const auto& block : m_sys)
        {
            block->cleanUp();
        }
    }

    void update(state_type& f, const state_type& x, double t) const
    {
        for (auto& block : m_sys)
        {
            block->updateT(f, x);
        }

        for (auto& block : m_sys)
        {
            block->updateP(f, x);
        }

        for (auto& block : m_sys)
        {
            block->updateU(f, x);
        }

        for (auto& block : m_sys)
        {
            block->updateVariables(x, t);
        }
    }

    size_t getSystemSize() const
    {
        size_t x_size = 0;

        for (auto& block : m_sys)
        {
            x_size += block->getSize();
        }

        return x_size;
    }

    state_vector getInitialConditions() const
    {
        state_vector x(getSystemSize());
        
        for (auto& block : m_sys)
        {
            block->setInitialConditions(x);
        }

        return x;
    }

    void setMassMatrix(sparse_matrix& M) const
    {
        M.reserve(getSystemSize());
        
        for (auto& block : m_sys)
        {
            block->setMassMatrix(M);
        }
    }

    void log(const state_vector& x, double t)
    {
        for (const auto& block : m_sys)
        {
            block->log(x, t);
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
        m_system.update(f, x, t);
    }
};

class MySolutionManager
{
    MySystem& m_sys;

public:
    explicit MySolutionManager(MySystem& sys) : m_sys(sys) {}
    
    int operator()(const state_vector& x, const double t)
    {
        m_sys.log(x, t);
        
        return 0;
    }
};


struct Study
{
    MySystem        m_system;
    SolverOptions   m_solverOptions;
    double          m_time = 100;
};


static void RunStudy(Study& study)
{
    // Initialise
    study.m_system.initialise();
    
    // Generate the initial condition vector
    state_vector x0 = study.m_system.getInitialConditions();

    // Solve the system of ODEs
    solve(
        MyMassMatrix(study.m_system), 
        MyRHS(study.m_system), 
        x0, 
        study.m_time, 
        MySolutionManager(study.m_system), 
        study.m_solverOptions
    );

    // Clean up
    study.m_system.cleanUp();
}


int main()
{
    Study study;

    // CREATE LAYERS
    auto pm1 = std::make_unique<PorousMedia>("PorousMedia1");
    auto pm2 = std::make_unique<PorousMedia>("PorousMedia2");
    pm1->N = 40;
    pm2->N = 20;

    // BOUNDARY CONDITIONS
    pm1->eq.T.bot_bc = std::make_unique<DirichletBC>(298, Location::Bottom);
    pm1->eq.T.top_bc = std::make_unique<DirichletBC>(273, Location::Top);
    pm2->eq.T.bot_bc = std::make_unique<DirichletBC>(298, Location::Bottom);
    pm2->eq.T.top_bc = std::make_unique<DirichletBC>(273, Location::Top);

    //connectBoundaryConditions(pm1->eq.T, pm2->eq.T);

    // INITIAL CONDITIONS
    pm1->eq.T.t0 = 293.0;
    pm1->eq.P.t0 = 101325.0;
    pm1->eq.U.t0 = 0.0;
    pm2->eq.T.t0 = 298.0;
    pm2->eq.P.t0 = 111325.0;
    pm2->eq.U.t0 = 0.4;

    // ADD BLOCKS TO SYSTEM
    study.m_system.addBlock(std::move(pm1));
    study.m_system.addBlock(std::move(pm2));

    study.m_time = 100.0;

    study.m_solverOptions.verbosity = verbosity::normal;        
    study.m_solverOptions.solution_variability_control = false; 
    study.m_solverOptions.BDF_order = 1;
    study.m_solverOptions.atol = 1e-10;
    study.m_solverOptions.rtol = 1e-10;

    RunStudy(study);
}
