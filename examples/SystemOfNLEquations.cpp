// C++ includes
#include <iostream>

#include "Solvex/Solvex.h"

// autodiff include
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using namespace autodiff;


// Your ODE functor, unchanged
struct ODE
{
    double A = 1.0;
    double B = 2.0;
    void operator()(autodiff::real t, const VectorXreal& x, VectorXreal& dxdt) const
    {
        dxdt(0) = 0.2 * x(1);
        dxdt(1) = (A / B) * x(0);
    }
};

VectorXreal BDF(const VectorXreal& x, const VectorXreal& x_dt, VectorXreal& Fx, ODE& ode, autodiff::real time, autodiff::real dt)
{
    int N = x.size();

    VectorXreal dx_dt(N); // not Eigen::VectorXd
    ode(time, x, dx_dt);

    for (int n = 0; n < N; ++n)
        Fx(n) = x(n) - x_dt(n) + dt * dx_dt(n);

    return Fx;
}

int main()
{
    ODE ode;

    // Now use ode_wrap with autodiff
    real t = 0.0;                  // if you want to treat t as constant, keep it as a plain double
    real dt = 1.0;                  // if you want to treat t as constant, keep it as a plain double
    VectorXreal x(2);             // autodiff variable for the state
    x << 1.0, 2.0;

    VectorXreal x_dt = x;             

    VectorXreal fx(2);

    Eigen::MatrixXd J = jacobian(BDF, autodiff::wrt(x), at(x, x_dt, fx, ode, t, dt), fx);

    // Print
    std::cout << "fx = \n" << fx << '\n';
    std::cout << "Jacobian = \n" << J << '\n';
}


/*
#include <iostream>

#include "Solvex/Solvex.h"
#include "Solvex/Equation.h"


struct ODE1
{
    double topBC    = 100;
    double bottomBC = 10;

    void operator()(const Eigen::Ref<const Eigen::VectorXd>& x1,
                    const Eigen::Ref<const Eigen::VectorXd>& x2,
                    Eigen::Ref<Eigen::VectorXd> dxdt1) const
    {
        int N = x1.size() - 1;

        dxdt1(0) = topBC - x1(0);
        for (int n = 1; n < N; ++n)
        {
            dxdt1(n) = x1(n - 1) - 2 * x1(n)*x1(n) + x1(n + 1) 
                       + 0.1 * x2(n);
        }
        dxdt1(N) = bottomBC - x1(N);
    }
};

struct ODE2
{
    double topBC    = 50;
    double bottomBC = 30;

    void operator()(const Eigen::Ref<const Eigen::VectorXd>& x2,
                    const Eigen::Ref<const Eigen::VectorXd>& x1,
                    Eigen::Ref<Eigen::VectorXd> dxdt2) const
    {
        int N = x2.size() - 1;

        dxdt2(0) = topBC - x2(0);
        for (int n = 1; n < N; ++n)
        {
            dxdt2(n) = x2(n - 1) - 2 * x2(n)*x2(n) + x2(n + 1)
                       + 0.05 * x1(n);
        }
        dxdt2(N) = bottomBC - x2(N);
    }
};

struct CoupledSystem
{
    ODE1 ode1;
    ODE2 ode2;

    int N;

    void operator()(const Eigen::VectorXd& x, Eigen::VectorXd& dxdt) const
    {
        assert(x.size() == 2*N);
        assert(dxdt.size() == 2*N);

        // Make "mapped" subvectors for x1 and x2:
        Eigen::Map<const Eigen::VectorXd> x1(&x[0], N);
        Eigen::Map<const Eigen::VectorXd> x2(&x[N], N);

        // Map subvectors for dxdt1 and dxdt2 (places we will write results)
        Eigen::Map<Eigen::VectorXd> dxdt1(&dxdt[0], N);
        Eigen::Map<Eigen::VectorXd> dxdt2(&dxdt[N], N);

        // Now call each subsystem’s operator on the appropriate sub-block:
        ode1(x1, x2, dxdt1);
        ode2(x2, x1, dxdt2);
    }
};

int main()
{
    // Suppose each subsystem is of dimension 10, so total dimension is 20:
    int N = 10;

    ODE1 ode1;
    ode1.topBC    = 100;
    ode1.bottomBC = 10;

    ODE2 ode2;
    ode2.topBC    = 50;
    ode2.bottomBC = 30;

    CoupledSystem system{ode1, ode2, N};

    Eigen::VectorXd x0(2*N);
    x0.setConstant(1.0);

    Eigen::VectorXd solution = Solvex::NLESolver(system, x0);

    std::cout << "Solution:\n" << solution << std::endl;
    return 0;
}
*/