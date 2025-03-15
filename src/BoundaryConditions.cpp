#include "Solvex/BoundaryConditions.h"
#include "Solvex/Equations.h"

namespace Solvex
{
    BoundaryCondition::BoundaryCondition(BoundaryType type, Location loc)
        : m_type(type), m_loc(loc) {}

    BoundaryCondition::~BoundaryCondition() = default;

    DirichletBC::DirichletBC(autodiff::real fixedValue, Location location)
        : BoundaryCondition(BoundaryType::Dirichlet, location), m_value(fixedValue) {}

    void DirichletBC::apply(state_type& f, const state_type& x, int index, double dx) const
    {
        f(index) = m_value - x(index);
    }

    NeumannBC::NeumannBC(autodiff::real fixedGradient, Location location)
        : BoundaryCondition(BoundaryType::Neumann, location), m_gradient(fixedGradient)
    {
        if (location == Location::Top)
        {
            dir = 1;
        }
    }

    void NeumannBC::apply(state_type& f, const state_type& x, int index, double dx) const
    {
        f(index) = m_gradient - autodiff::real(dir) * (x(index) - x(index - dir)) / autodiff::real(dx);
    }

    InterfaceBoundary::InterfaceBoundary(int connecting_index, Location location)
        : BoundaryCondition(BoundaryType::Interface, location), 
          m_connecting_index(connecting_index) {}

    void InterfaceBoundary::apply(state_type& f, const state_type& x, int index, double dx) const
    {
        f(index) = x(index) - x(m_connecting_index);
    }

    void connectBoundaryConditions(Equation& top_equation, Equation& bottom_equation)
    {
        int top_index = top_equation.endIndex - 1;
        int bottom_index = bottom_equation.startIndex + 1;

        top_equation.top_bc = std::make_unique<InterfaceBoundary>(bottom_index, Location::Top);
        bottom_equation.bot_bc = std::make_unique<InterfaceBoundary>(top_index, Location::Bottom);
    }

} // end Solvex namespace

