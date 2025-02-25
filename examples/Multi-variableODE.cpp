#include <iostream>
#include <unordered_map>

#include "Solvex/Solvex.h"
#include "Solvex/Equation.h"


struct updateLayerTemperature
{
    double topBC = 100;
    double bottomBC = 10;

    void operator()(const Eigen::Ref<const Eigen::VectorXd>& T,
        const Eigen::Ref<const Eigen::VectorXd>& P,
        Eigen::Ref<Eigen::VectorXd> dT_dt) const
    {
        int N = T.size() - 1;

        dT_dt(0) = topBC - T(0);
        for (int n = 1; n < N; ++n)
        {
            dT_dt(n) = T(n - 1) - 2 * T(n) + T(n + 1) 
                       + 0.1 * P(n);
        }
        dT_dt(N) = bottomBC - T(N);
    }
};

struct updateLayerPressure
{
    double topBC = 50;
    double bottomBC = 30;

    void operator()(const Eigen::Ref<const Eigen::VectorXd>& P,
        const Eigen::Ref<const Eigen::VectorXd>& T,
        Eigen::Ref<Eigen::VectorXd> dP_dt) const
    {
        int N = P.size() - 1;

        dP_dt(0) = topBC - P(0);
        for (int n = 1; n < N; ++n)
        {
            dP_dt(n) = P(n - 1) - 2 * P(n)*P(n) + P(n + 1) 
                       + 0.1 * T(n);
        }
        dP_dt(N) = bottomBC - P(N);
    }
};

struct Layer
{
    size_t N = 10;

    Eigen::VectorXd T0{};
    Eigen::VectorXd P0{};

    mutable Eigen::VectorXd T{};
    mutable Eigen::VectorXd P{};

    updateLayerTemperature updateTemperature;
    updateLayerPressure updatePressure;
};

struct System
{
    Layer layer;
    
    void operator()(const Eigen::VectorXd& x, Eigen::VectorXd& dxdt) const
    {
        Eigen::Map<const Eigen::VectorXd> T(&x[0], layer.N);
        Eigen::Map<const Eigen::VectorXd> P(&x[layer.N], layer.N);

        Eigen::Map<Eigen::VectorXd> dT_dt(&dxdt[0], layer.N);
        Eigen::Map<Eigen::VectorXd> dP_dt(&dxdt[layer.N], layer.N);

        layer.T = T;
        layer.P = P;

        layer.updateTemperature(T, P, dT_dt);
        layer.updatePressure(P, T, dP_dt);
    }

    Eigen::VectorXd getInitialConditions() const
    {
        Eigen::VectorXd x0(2 * layer.N);
        x0 << layer.T0, layer.P0; // uses Eigen's "comma initializer" for stack assignment
        return x0;
    }
};

int main()
{
    int N = 10;

    System system;
    system.layer.N = N;
    system.layer.updatePressure.topBC = 50;
    system.layer.updatePressure.bottomBC = 30;

    system.layer.T0.setConstant(N, 50.0);
    system.layer.P0.setConstant(N, 30.0);

    Eigen::VectorXd x0 = system.getInitialConditions();

    std::cout << "x0: " << x0 << "\n";

    //Eigen::VectorXd solution = Solvex::NLESolver(system, x0);
    Eigen::VectorXd solution = Solvex::BFD1Solver(system, x0, 0, 100);

    std::cout << "Solution:\n" << solution << std::endl;
    return 0;
}
