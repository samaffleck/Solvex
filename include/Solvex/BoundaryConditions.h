#pragma once

// DAE-CPP includes
#include "dae-cpp/solver.hpp"


namespace Solvex
{
    struct Equation;

    using namespace daecpp;

    // Enums
    enum class BoundaryType 
    {
        Dirichlet,  // Fixed value
        Neumann,    // Fixed gradient
        Robin,      // Mixed (convective)
        Periodic,   // Periodic boundaries
        Interface
    };

    enum class Location
    {
        Top,
        Bottom
    };

    // Boundary condition interface
    struct BoundaryCondition 
    {
        BoundaryType    m_type;
        Location        m_loc;

        BoundaryCondition(BoundaryType type, Location loc);
        virtual ~BoundaryCondition();

        virtual void apply(state_type& f, const state_type& x, int index, double dx) const = 0;
    };

    struct DirichletBC : BoundaryCondition 
    {
        autodiff::real m_value;

        DirichletBC(autodiff::real fixedValue, Location location);
        void apply(state_type& f, const state_type& x, int index, double dx) const override;
    };

    struct NeumannBC : BoundaryCondition 
    {
        autodiff::real m_gradient;

        NeumannBC(autodiff::real fixedGradient, Location location);
        void apply(state_type& f, const state_type& x, int index, double dx) const override;

    private:
        int dir = -1;
    };

    struct InterfaceBoundary : BoundaryCondition 
    {
        int m_connecting_index;

        InterfaceBoundary(int connecting_index, Location location);
        void apply(state_type& f, const state_type& x, int index, double dx) const override;
    };

    void connectBoundaryConditions(Equation& top_equation, Equation& bottom_equation);

} // end Solvex namespace
