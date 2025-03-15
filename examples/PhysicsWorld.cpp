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


struct Domain
{
    double  L = 1.0;        // Length of domain [m]
    double  dx = 1.0;       // Cell width [m]
    int     N = 40;         // Number of cells
};


struct Solid
{
    VariableInTimeAndSpace K;   // Thermal Conductivity
    double rho;                 // Density
    double Cp;                  // Heat cap
};


struct SolidPhysicsBlock
{
    Domain      m_dom;
    Solid       m_prop;
    Equation    T;
};


struct GasPhysicsBlock
{
    Domain      m_dom;
    Solid       m_prop;
    Equation    T;
    Equation    P;
    Equation    U;
};

struct PhysicsWorld
{
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
}

static void initialiseSolidPyhsicsBlock(SolidPhysicsBlock& solid, int& index)
{
    initialiseEquation(solid.T, index, solid.m_dom.N);
}

static void initialiseGasPhysicsBlock(GasPhysicsBlock& gas, int& index)
{
    initialiseEquation(gas.T, index, gas.m_dom.N);
    initialiseEquation(gas.P, index, gas.m_dom.N);
    initialiseEquation(gas.U, index, gas.m_dom.N);
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
    solid.T.bot_bc->apply(f, x, solid.T.startIndex, solid.m_dom.dx);

    for (int i = solid.T.startIndex + 1; i < solid.T.endIndex; ++i)
    {
        f(i) = x(i + 1) - autodiff::real(2) * x(i) + x(i - 1);
    }

    solid.T.top_bc->apply(f, x, solid.T.endIndex, solid.m_dom.dx);
}

static void updateSolidPyhsicsBlock(SolidPhysicsBlock& solid, state_type& f, const state_type& x, double t)
{
    updateSolidThermalConductivity(solid.m_prop.K, solid.T, x, t);
    updateSolidTemperature(solid, f, x);
}

static void updateGasPressure(const GasPhysicsBlock& gas, state_type& f, const state_type& x)
{
    autodiff::real _dx = 1 / gas.m_dom.dx;

    gas.P.bot_bc->apply(f, x, gas.P.startIndex, gas.m_dom.dx);

    for (int i = gas.P.startIndex + 1; i < gas.P.endIndex; ++i)
    {
        autodiff::real Pe = autodiff::real(0.5) * (x(i) + x(i + 1));
        autodiff::real Pw = autodiff::real(0.5) * (x(i) + x(i - 1));

        autodiff::real dPdx_e = _dx * (x(i + 1) - x(i));
        autodiff::real dPdx_w = _dx * (x(i) - x(i - 1));

        f(i) = (dPdx_e * Pe - dPdx_w * Pw);
    }

    gas.P.top_bc->apply(f, x, gas.P.endIndex, gas.m_dom.dx);
}

static void updateGasTemperature(const GasPhysicsBlock& gas, state_type& f, const state_type& x)
{
    // TODO
}

static void updateGasVelocity(const GasPhysicsBlock& gas, state_type& f, const state_type& x)
{
    // TODO
}

static void updateGasPhysicsBlock(const GasPhysicsBlock& gas, state_type& f, const state_type& x, double t)
{
    updateGasPressure(gas, f, x);
    updateGasTemperature(gas, f, x);
    updateGasVelocity(gas, f, x);
}

static void updatePhysicsWorld(PhysicsWorld& world, state_type& f, const state_type& x, double t)
{
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
    size += getEquationSize(gas.T, gas.m_dom.N);
    size += getEquationSize(gas.P, gas.m_dom.N);
    size += getEquationSize(gas.U, gas.m_dom.N);
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
    setEquationInitialConditions(gas.T, x);
    setEquationInitialConditions(gas.P, x);
    setEquationInitialConditions(gas.U, x);
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
    setEquationMassMatrix(gas.T, M);
    setEquationMassMatrix(gas.P, M);
    // Velocity is not an ODE
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

class MyMassMatrix
{
    PhysicsWorld& m_world;

public:
    explicit MyMassMatrix(PhysicsWorld& world) : m_world(world) {}

    void operator()(sparse_matrix& M, const double t) const
    {
        setPhysicsWorldMassMatrix(m_world, M);
    }
};

class MyRHS
{
    PhysicsWorld& m_world;

public:
    explicit MyRHS(PhysicsWorld& world) : m_world(world) {}

    void operator()(state_type& f, const state_type& x, const double t) const
    {
        updatePhysicsWorld(m_world, f, x, t);
    }
};

class MySolutionManager
{
    PhysicsWorld& m_world;

public:
    explicit MySolutionManager(PhysicsWorld& world) : m_world(world) {}
    
    int operator()(const state_vector& x, const double t)
    {
        //logPhysicsWorld(m_world, x, t);
        
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
        MyMassMatrix(study.m_world), 
        MyRHS(study.m_world),
        x0, 
        study.m_time, 
        MySolutionManager(study.m_world),
        study.m_solverOptions
    );

    // Clean up
    //study.m_system.cleanUp();
    //cleanUpPhysicsWorld(study.m_world);
}


int main()
{
    Study study;

    SolidPhysicsBlock solid;
    solid.m_dom.L = 1.0;
    solid.m_dom.N = 40;
    solid.m_prop.Cp = 500;
    solid.m_prop.rho = 7800;
    
    // BOUNDARY CONDITIONS
    solid.T.bot_bc = std::make_unique<DirichletBC>(298, Location::Bottom); 
    solid.T.top_bc = std::make_unique<DirichletBC>(273, Location::Top);
    
    // INITIAL CONDITIONS
    solid.T.t0 = 293.0;

    // ADD BLOCKS TO SYSTEM
    study.m_world.m_solids.emplace_back(std::move(solid));
    
    study.m_time = 100.0;

    study.m_solverOptions.verbosity = verbosity::normal;        
    study.m_solverOptions.solution_variability_control = false; 
    study.m_solverOptions.BDF_order = 1;
    study.m_solverOptions.atol = 1e-10;
    study.m_solverOptions.rtol = 1e-10;

    RunStudy(study);
}
