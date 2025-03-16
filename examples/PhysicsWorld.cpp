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


struct VariableInTimeAndSpace
{
    std::ofstream       file{};         // csv file for log data
    state_type          m_var{};        // Variable that changes with time and space in 1 dimention
};

static void updateSolidThermalConductivity(VariableInTimeAndSpace& thermal_conductivity, const Equation& T, const state_type& x, double t)
{
    int T_index = T.startIndex;
    double T_ref = 298;

    for (int i = 0; i < thermal_conductivity.m_var.size(); ++i)
    {
        thermal_conductivity.m_var[i] = 1e-3 * (x[T_index + i] / T_ref);
    }
}


struct ICouplingTerm 
{
    virtual ~ICouplingTerm() = default;
    virtual void apply(const state_type& x) = 0;
};

struct ConductionCoupling : public ICouplingTerm
{
    Equation& eqA;   
    Equation& eqB;   
    double    h;     // conduction or convective coefficient
    double    d;     // characteristic length, etc.

    ConductionCoupling(Equation& eA, Equation& eB, double h_, double d_)
        : eqA(eA), eqB(eB), h(h_), d(d_) {}

    void apply(const state_type& x) override
    {
        int nA = eqA.endIndex - eqA.startIndex + 1;
        int nB = eqB.endIndex - eqB.startIndex + 1;
        assert(nA == nB); 

        for (int iCell = 0; iCell < nA; ++iCell)
        {
            int iA = eqA.startIndex + iCell;
            int iB = eqB.startIndex + iCell;

            double TA = x(iA).val();  // autodiff::real or double
            double TB = x(iB).val();

            double flux = 4.0 * h / d * (TA - TB);

            eqA.source(iCell) += flux;
            eqB.source(iCell) -= flux;
        }
    }
};



struct Domain
{
    std::string name{};
    double  L = 1.0;        // Length of domain [m]
    double  dx = 1.0;       // Cell width [m]
    int     N = 40;         // Number of cells
};


struct Solid
{
    //VariableInTimeAndSpace K;     // Thermal Conductivity
    double K;                       // Thermal Conductivity
    double rho;                     // Density
    double Cp;                      // Heat cap
};


struct SolidPhysicsBlock
{
    Domain      m_dom;
    Solid       m_prop;
    Equation    T;
};


struct Gas 
{
    double Cp;                  // Heat cap
    double Vis;                 // Viscosity
    double K;                   // Thermal conductivity
    double gamma;               // Ratio of specific heats
    double Mr;                  // Molar mass [kg/mol]
};


struct GasPhysicsBlock
{
    Domain      m_dom;
    Gas         m_prop;
    Equation    Rho;    // Density [mg/m3]
    Equation    M;      // Mass flux [kg/(m2-s)]
    Equation    H;      // Enthalpy = rho*E
};

struct PhysicsWorld
{
    std::vector<std::unique_ptr<ICouplingTerm>> m_couplings;

    std::vector<SolidPhysicsBlock>  m_solids;
    std::vector<GasPhysicsBlock>    m_gas;

    // Add more vectors of Physics blocks as needed
};


static void initialiseEquation(Equation& eq, int& index, int N)
{
    eq.startIndex = index;
    if (eq.cellType == CellType::CENTER)
        index += N + 1;
    else
        index += N;

    eq.endIndex = index;
    index++;

    eq.source.resize(eq.endIndex - eq.startIndex + 1);
}

static void initialiseSolidPyhsicsBlock(SolidPhysicsBlock& solid, int& index)
{
    solid.m_dom.dx = solid.m_dom.L / solid.m_dom.N;
    initialiseEquation(solid.T, index, solid.m_dom.N);
    solid.T.file.open(solid.m_dom.name + "_T.csv");
}

static void initialiseGasPhysicsBlock(GasPhysicsBlock& gas, int& index)
{
    gas.m_dom.dx = gas.m_dom.L / gas.m_dom.N;

    initialiseEquation(gas.Rho, index, gas.m_dom.N);
    initialiseEquation(gas.M, index, gas.m_dom.N);
    initialiseEquation(gas.H, index, gas.m_dom.N);
    gas.Rho.file.open(gas.m_dom.name + "_Rho.csv");
    gas.M.file.open(gas.m_dom.name + "_M.csv");
    gas.H.file.open(gas.m_dom.name + "_H.csv");
}

static void initialisePhysicsWorld(PhysicsWorld& world)
{
    int index = 0;

    for (SolidPhysicsBlock& solid : world.m_solids)
    {
        initialiseSolidPyhsicsBlock(solid, index);
    }

    for (GasPhysicsBlock& gas : world.m_gas)
    {
        initialiseGasPhysicsBlock(gas, index);
    }

    // More physics blocks ...
}

static void updateSolidTemperature(const SolidPhysicsBlock& solid, state_type& f, const state_type& x)
{
    const autodiff::real Mt = solid.m_prop.Cp * solid.m_prop.rho;
    const autodiff::real a = - solid.m_prop.K / (Mt * solid.m_dom.dx * solid.m_dom.dx);

    solid.T.bot_bc->apply(f, x, solid.T.startIndex, solid.m_dom.dx);
    solid.T.top_bc->apply(f, x, solid.T.endIndex, solid.m_dom.dx);

    for (int i = solid.T.startIndex + 1; i < solid.T.endIndex; ++i)
    {
        f(i) = a * (x(i + 1) - 2 * x(i) + x(i - 1)) + solid.T.source[i - solid.T.startIndex] / Mt;
    }
}

static void updateSolidPyhsicsBlock(SolidPhysicsBlock& solid, state_type& f, const state_type& x, double t)
{
    //updateSolidThermalConductivity(solid.m_prop.K, solid.T, x, t);
    updateSolidTemperature(solid, f, x);
}

static void updateGasDensity(const GasPhysicsBlock& gas, state_type& f, const state_type& x)
{
    autodiff::real _dx = 1 / gas.m_dom.dx;

    gas.Rho.bot_bc->apply(f, x, gas.Rho.startIndex, gas.m_dom.dx);
    gas.Rho.top_bc->apply(f, x, gas.Rho.endIndex, gas.m_dom.dx);

    for (int i = gas.Rho.startIndex + 1; i < gas.Rho.endIndex; ++i)
    {
        const autodiff::real M_P = x[gas.M.startIndex + i];
        const autodiff::real M_E = x[gas.M.startIndex + i + 1];
        const autodiff::real M_W = x[gas.M.startIndex + i - 1];
        const autodiff::real M_e = 0.5 * (M_P + M_E);
        const autodiff::real M_w = 0.5 * (M_P + M_W);
        
        f[gas.Rho.startIndex + i] = -_dx * (M_e - M_w);
    }
}

static void updateGasMassFlux(const GasPhysicsBlock& gas, state_type& f, const state_type& x)
{
    autodiff::real _dx = 1 / gas.m_dom.dx;

    gas.M.bot_bc->apply(f, x, gas.M.startIndex, gas.m_dom.dx);
    gas.M.top_bc->apply(f, x, gas.M.endIndex, gas.m_dom.dx);

    for (int i = gas.M.startIndex + 1; i < gas.M.endIndex; ++i)
    {
        const autodiff::real rho_P = x[gas.Rho.startIndex + i];
        const autodiff::real rho_E = x[gas.Rho.startIndex + i + 1];
        const autodiff::real rho_W = x[gas.Rho.startIndex + i - 1];
        const autodiff::real rho_e = 0.5 * (rho_P + rho_E);
        const autodiff::real rho_w = 0.5 * (rho_P + rho_W);

        const autodiff::real M_P = x[gas.M.startIndex + i];
        const autodiff::real M_E = x[gas.M.startIndex + i + 1];
        const autodiff::real M_W = x[gas.M.startIndex + i - 1];
        const autodiff::real M_e = 0.5 * (M_P + M_E);
        const autodiff::real M_w = 0.5 * (M_P + M_W);

        const autodiff::real H_P = x[gas.H.startIndex + i];
        const autodiff::real H_E = x[gas.H.startIndex + i + 1];
        const autodiff::real H_W = x[gas.H.startIndex + i - 1];

        // p_i = (gamma - 1)*( H_i - 0.5*(M_i^2 / rho_i) )
        const autodiff::real p_P = (gas.m_prop.gamma - 1.0) * (H_P - 0.5 * (M_P * M_P / rho_P));
        const autodiff::real p_E = (gas.m_prop.gamma - 1.0) * (H_E - 0.5 * (M_E * M_E / rho_E));
        const autodiff::real p_W = (gas.m_prop.gamma - 1.0) * (H_W - 0.5 * (M_W * M_W / rho_W));
        const autodiff::real p_e = 0.5 * (p_P + p_E);
        const autodiff::real p_w = 0.5 * (p_P + p_W);

        // Flux function F(M) = M^2/rho + p
        const autodiff::real Flux_e = (M_e * M_e) / rho_e + p_e;
        const autodiff::real Flux_w = (M_w * M_w) / rho_w + p_w;

        f[gas.M.startIndex + i] = -_dx * (Flux_e - Flux_w);
    }
}

static void updateGasEnthalpy(const GasPhysicsBlock& gas, state_type& f, const state_type& x)
{
    autodiff::real _dx = 1.0 / gas.m_dom.dx;
    autodiff::real gamma = gas.m_prop.gamma;

    gas.H.bot_bc->apply(f, x, gas.H.startIndex, gas.m_dom.dx);
    gas.H.top_bc->apply(f, x, gas.H.endIndex, gas.m_dom.dx);

    for (int i = gas.H.startIndex + 1; i < gas.H.endIndex; ++i)
    {
        // Cell-centered values at i, i+1 (E), i-1 (W)
        const autodiff::real rho_P = x[gas.Rho.startIndex + i];
        const autodiff::real rho_E = x[gas.Rho.startIndex + i + 1];
        const autodiff::real rho_W = x[gas.Rho.startIndex + i - 1];

        const autodiff::real M_P = x[gas.M.startIndex + i];
        const autodiff::real M_E = x[gas.M.startIndex + i + 1];
        const autodiff::real M_W = x[gas.M.startIndex + i - 1];

        const autodiff::real H_P = x[gas.H.startIndex + i];
        const autodiff::real H_E = x[gas.H.startIndex + i + 1];
        const autodiff::real H_W = x[gas.H.startIndex + i - 1];

        // Face-averaged ("east" face) = average of center i and i+1
        const autodiff::real rho_e = 0.5 * (rho_P + rho_E);
        const autodiff::real M_e = 0.5 * (M_P + M_E);
        const autodiff::real H_e = 0.5 * (H_P + H_E);

        // Face-averaged ("west" face) = average of center i and i-1
        const autodiff::real rho_w = 0.5 * (rho_P + rho_W);
        const autodiff::real M_w = 0.5 * (M_P + M_W);
        const autodiff::real H_w = 0.5 * (H_P + H_W);

        // Compute pressures at cell centers
        // p = (gamma - 1) * ( H - 0.5*(M^2 / rho ) )
        const autodiff::real p_P = (gamma - 1.0) * (H_P - 0.5 * (M_P * M_P / rho_P));
        const autodiff::real p_E = (gamma - 1.0) * (H_E - 0.5 * (M_E * M_E / rho_E));
        const autodiff::real p_W = (gamma - 1.0) * (H_W - 0.5 * (M_W * M_W / rho_W));

        // Face-averaged pressures
        const autodiff::real p_e = 0.5 * (p_P + p_E);
        const autodiff::real p_w = 0.5 * (p_P + p_W);

        // Flux for Enthalpy: F(H) = (H + p) * (M / rho)
        const autodiff::real Flux_e = (H_e + p_e) * (M_e / rho_e);
        const autodiff::real Flux_w = (H_w + p_w) * (M_w / rho_w);

        // Update the time derivative of H in cell i
        // dH/dt = -(Flux_e - Flux_w)/dx
        f[gas.H.startIndex + i] = -_dx * (Flux_e - Flux_w);
    }
}


static void updateGasPhysicsBlock(const GasPhysicsBlock& gas, state_type& f, const state_type& x, double t)
{
    updateGasDensity(gas, f, x);
    updateGasMassFlux(gas, f, x);
    updateGasEnthalpy(gas, f, x);
}

void resetSourceTerms(PhysicsWorld& world)
{
    for (auto& solid : world.m_solids)
    {
        std::fill(solid.T.source.begin(), solid.T.source.end(), 0.0);
    }

    for (auto& gas : world.m_gas)
    {
        std::fill(gas.Rho.source.begin(), gas.Rho.source.end(), 0.0);
        std::fill(gas.M.source.begin(), gas.M.source.end(), 0.0);
        std::fill(gas.H.source.begin(), gas.H.source.end(), 0.0);
    }

}

static void updatePhysicsWorld(PhysicsWorld& world, state_type& f, const state_type& x, double t)
{
    resetSourceTerms(world);

    for (const auto& cptr : world.m_couplings)
    {
        cptr->apply(x);
    }

    for (SolidPhysicsBlock& solid : world.m_solids)
    {
        updateSolidPyhsicsBlock(solid, f, x, t);
    }

    for (GasPhysicsBlock& gas : world.m_gas)
    {
        updateGasPhysicsBlock(gas, f, x, t);
    }
}

static int getEquationSize(const Equation& equation, int N)
{
    if (equation.cellType == CellType::CENTER)
        return N + 2;
    else
        return N + 1;
}

static int getSolidSize(const SolidPhysicsBlock& solid)
{
    int size = 0;
    size += getEquationSize(solid.T, solid.m_dom.N);
    return size;
}

static int getGasSize(const GasPhysicsBlock& gas)
{
    int size = 0;
    size += getEquationSize(gas.Rho, gas.m_dom.N);
    size += getEquationSize(gas.M, gas.m_dom.N);
    size += getEquationSize(gas.H, gas.m_dom.N);
    return size;
}

static int getPhysicsWorldSize(const PhysicsWorld& world)
{
    int size = 0;
    
    for (const SolidPhysicsBlock& solid : world.m_solids)
    {
        size += getSolidSize(solid);
    }

    for (const GasPhysicsBlock& gas : world.m_gas)
    {
        size += getGasSize(gas);
    }

    return size;
}

static void setEquationInitialConditions(const Equation& equation, state_vector& x)
{
    for (int i = equation.startIndex; i <= equation.endIndex; ++i)
    {
        x[i] = equation.t0;
    }
}

static void setSolidInitilConditions(const SolidPhysicsBlock& solid, state_vector& x)
{
    setEquationInitialConditions(solid.T, x);
}

static void setGasInitilConditions(const GasPhysicsBlock& gas, state_vector& x)
{
    setEquationInitialConditions(gas.Rho, x);
    setEquationInitialConditions(gas.M, x);
    setEquationInitialConditions(gas.H, x);
}

static state_vector getPhysicsWorldInitialConditions(const PhysicsWorld& world)
{
    state_vector x(getPhysicsWorldSize(world));

    for (const SolidPhysicsBlock& solid : world.m_solids)
    {
        setSolidInitilConditions(solid, x);
    }

    for (const GasPhysicsBlock& gas : world.m_gas)
    {
        setGasInitilConditions(gas, x);
    }

    return x;
}

static void setEquationMassMatrix(const Equation& equation, sparse_matrix& M)
{
    for (int i = equation.startIndex + 1; i < equation.endIndex; ++i) // Ignore the first and last nodes as they are ghost cells
    {
        M(i, i, 1.0);
    }
}

static void setSolidMassMatrix(const SolidPhysicsBlock& solid, sparse_matrix& M)
{
    setEquationMassMatrix(solid.T, M);
}

static void setGasMassMatrix(const GasPhysicsBlock& gas, sparse_matrix& M)
{
    setEquationMassMatrix(gas.Rho, M);
    setEquationMassMatrix(gas.M, M);
    setEquationMassMatrix(gas.H, M);
}

static void setPhysicsWorldMassMatrix(const PhysicsWorld& world, sparse_matrix& M)
{
    M.reserve(getPhysicsWorldSize(world));

    for (const SolidPhysicsBlock& solid : world.m_solids)
    {
        setSolidMassMatrix(solid, M);
    }

    for (const GasPhysicsBlock& gas : world.m_gas)
    {
        setGasMassMatrix(gas, M);
    }
}


static void logSolidPhysicsBlock(SolidPhysicsBlock& solid, const state_vector& x, const double t)
{
    solid.T.log(x, t);
}

static void logGasPhysicsBlock(GasPhysicsBlock& gas, const state_vector& x, const double t)
{
    gas.Rho.log(x, t);
    gas.M.log(x, t);
    gas.H.log(x, t);
}

static void logPhysicsWorld(PhysicsWorld& world, const state_vector& x, const double t)
{
    for (SolidPhysicsBlock& solid : world.m_solids)
    {
        logSolidPhysicsBlock(solid, x, t);
    }

    for (GasPhysicsBlock& gas : world.m_gas)
    {
        logGasPhysicsBlock(gas, x, t);
    }
}

class PhysicsWorldMassMatrix
{
    PhysicsWorld& m_world;

public:
    explicit PhysicsWorldMassMatrix(PhysicsWorld& world) : m_world(world) {}

    void operator()(sparse_matrix& M, const double t) const
    {
        setPhysicsWorldMassMatrix(m_world, M);
    }
};

class PhysicsWorldRHS
{
    PhysicsWorld& m_world;

public:
    explicit PhysicsWorldRHS(PhysicsWorld& world) : m_world(world) {}

    void operator()(state_type& f, const state_type& x, const double t) const
    {
        updatePhysicsWorld(m_world, f, x, t);
    }
};

class PhysicsWorldSolutionManager
{
    PhysicsWorld& m_world;

public:
    explicit PhysicsWorldSolutionManager(PhysicsWorld& world) : m_world(world) {}
    
    int operator()(const state_vector& x, const double t)
    {
        logPhysicsWorld(m_world, x, t);
        
        return 0;
    }
};


struct Study
{
    PhysicsWorld    m_world;
    SolverOptions   m_solverOptions;
    double          m_time = 100;
};


static void RunStudy(Study& study)
{
    // Initialise
    initialisePhysicsWorld(study.m_world);
    
    // Generate the initial condition vector
    state_vector x0 = getPhysicsWorldInitialConditions(study.m_world);

    // Solve the system of ODEs
    solve(
        PhysicsWorldMassMatrix(study.m_world), 
        PhysicsWorldRHS(study.m_world),
        x0, 
        study.m_time, 
        PhysicsWorldSolutionManager(study.m_world),
        study.m_solverOptions
    );

    // Clean up
    //study.m_system.cleanUp();
    //cleanUpPhysicsWorld(study.m_world);
}


int main()
{
    Study study;

    study.m_world.m_solids.reserve(10);

    // ADD BLOCKS TO SYSTEM
    SolidPhysicsBlock& sol = study.m_world.m_solids.emplace_back();
    SolidPhysicsBlock& air = study.m_world.m_solids.emplace_back();
    SolidPhysicsBlock& liq = study.m_world.m_solids.emplace_back();

    sol.m_dom.name = "Solid";
    air.m_dom.name = "Air";
    liq.m_dom.name = "Liquid";

    sol.m_dom.L = 1.0;
    air.m_dom.L = 1.0;
    liq.m_dom.L = 1.0;

    sol.m_dom.N = 40;
    liq.m_dom.N = 40;
    air.m_dom.N = 40;
    
    sol.m_prop.Cp = 7500;
    air.m_prop.Cp = 500;
    liq.m_prop.Cp = 1000;
    
    sol.m_prop.rho = 7800;
    air.m_prop.rho = 1;
    liq.m_prop.rho = 1000;
    
    // BOUNDARY CONDITIONS
    sol.T.bot_bc = std::make_unique<NeumannBC>(0.0, Location::Bottom); 
    sol.T.top_bc = std::make_unique<NeumannBC>(0.0, Location::Top);
    air.T.bot_bc = std::make_unique<NeumannBC>(0.0, Location::Bottom);
    air.T.top_bc = std::make_unique<NeumannBC>(0.0, Location::Top);
    liq.T.bot_bc = std::make_unique<NeumannBC>(0.0, Location::Bottom);
    liq.T.top_bc = std::make_unique<NeumannBC>(0.0, Location::Top);


    // INITIAL CONDITIONS
    sol.T.t0 = 333.0;
    air.T.t0 = 393.0;
    liq.T.t0 = 293.0;

    // COUPLINGS
    auto conduction_sol_air = std::make_unique<ConductionCoupling>(air.T, sol.T, 1.0e-4, 0.01);
    auto conduction_sol_liq = std::make_unique<ConductionCoupling>(sol.T, liq.T, 1.0e-4, 0.02);
    study.m_world.m_couplings.push_back(std::move(conduction_sol_air));
    study.m_world.m_couplings.push_back(std::move(conduction_sol_liq));

    study.m_time = 100.0;

    study.m_solverOptions.verbosity = verbosity::normal;        
    study.m_solverOptions.solution_variability_control = false; 
    study.m_solverOptions.BDF_order = 1;
    study.m_solverOptions.atol = 1e-10;
    study.m_solverOptions.rtol = 1e-10;

    RunStudy(study);
}
