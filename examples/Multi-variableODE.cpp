#include <iostream>
#include <unordered_map>
#include <map>

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

template <typename T>
class Collection
{
public:

    void add(const std::string& key, const T& item)
    {
        items[key] = item;
        insertionOrder.push_back(key);
    }

    T* get(const std::string& key)
    {
        auto it = items.find(key);
        return (it != items.end()) ? &it->second : nullptr;
    }

    bool contains(const std::string& key) const
    {
        return items.find(key) != items.end();
    }

    bool remove(const std::string& key)
    {
        auto it = items.find(key);
        if (it != items.end())
        {
            items.erase(it);
            insertionOrder.erase(
                std::remove(insertionOrder.begin(), insertionOrder.end(), key),
                insertionOrder.end()
            );
            return true;
        }
        return false;
    }

    std::vector<T*> getOrderedItems()
    {
        std::vector<T*> orderedItems;
        for (const auto& key : insertionOrder)
        {
            orderedItems.push_back(&items[key]);
        }
        return orderedItems;
    }

private:
    std::unordered_map<std::string, T> items;
    std::vector<std::string> insertionOrder;

};

struct System
{
    Collection<Layer> layers;

    int getNumberOfCells()
    {
        int total = 0;
        for (const Layer* layer : layers.getOrderedItems())
        {
            total += layer->N;
        }
        return total;
    }

    void operator()(double time, const Eigen::VectorXd& x, Eigen::VectorXd& dxdt)
    {
       int Nt = 0, Np = 0;

        for (const Layer* layer : layers.getOrderedItems())
        {
            Eigen::Map<const Eigen::VectorXd> T(&x[Nt], layer->N);
            Eigen::Map<Eigen::VectorXd> dT_dt(&dxdt[Nt], layer->N);

            int P_offset = getNumberOfCells() + Np;
            Eigen::Map<const Eigen::VectorXd> P(&x[P_offset], layer->N);
            Eigen::Map<Eigen::VectorXd> dP_dt(&dxdt[P_offset], layer->N);

            layer->T = T;
            layer->updateTemperature(T, P, dT_dt);

            layer->P = P;
            layer->updatePressure(P, T, dP_dt);

            Nt += layer->N;
            Np += layer->N;
        }
    }

    Eigen::VectorXd getInitialConditions()
    {
        int Nc = getNumberOfCells();

        Eigen::VectorXd x0(Nc * 2); // Allocate space for T and P

        int offset_T = 0;
        int offset_P = Nc;

        for (const Layer* layer : layers.getOrderedItems())
        {
            x0.segment(offset_T, layer->N) = layer->T0;
            x0.segment(offset_P, layer->N) = layer->P0;
            offset_P += layer->N;
            offset_T += layer->N;
        }

        return x0;
    }

};

int main()
{
    int N = 10;

    System system;

    Layer layer1;
    layer1.N = 10;
    layer1.updatePressure.topBC = 50;
    layer1.updatePressure.bottomBC = 30;
    layer1.updateTemperature.topBC = 100;
    layer1.updateTemperature.bottomBC = 0;
    layer1.T0.setConstant(layer1.N, 50.0);
    layer1.P0.setConstant(layer1.N, 30.0);

    Layer layer2;
    layer2.N = 20;
    layer2.updatePressure.topBC = 50;
    layer2.updatePressure.bottomBC = 30;
    layer2.updateTemperature.topBC = 100;
    layer2.updateTemperature.bottomBC = 0;
    layer2.T0.setConstant(layer2.N, 50.0);
    layer2.P0.setConstant(layer2.N, 30.0);

    system.layers.add("layer 1", layer1);
    system.layers.add("layer 2", layer2);

    Eigen::VectorXd x0 = system.getInitialConditions();

    std::cout << "x0: " << x0 << "\n";

    Eigen::VectorXd BDF1_solution = Solvex::BDF1Solver(system, x0, 0, 1000);
    Eigen::VectorXd BDF2_solution = Solvex::BDF2Solver(system, x0, 0, 1000);

    double errorBDF1 = (BDF2_solution - BDF1_solution).norm();
    
    std::cout << "\nError BDF1:\n" << errorBDF1 << std::endl;
    return 0;
}
