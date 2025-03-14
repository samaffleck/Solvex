#pragma once

// STL includes
#include <iostream>
#include <fstream>
#include <vector>

// DAE-CPP includes
#include "dae-cpp/solver.hpp"

namespace Solvex
{
    using namespace daecpp;

    struct VariableInTime
    {
        std::ofstream   file{};         // csv file for log data
        double          m_var{};        // Variable that changes with time, t

        VariableInTime() = default;
        virtual ~VariableInTime() = default;

        virtual void update(double t) = 0;

        void log(double t)
        {
            file << t << "," << m_var << "\n";
        }
    };

    struct VariableInTime_1D
    {
        std::ofstream       file{};         // csv file for log data
        std::vector<double> m_var{};        // Variable that changes with time and space in 1 dimention

        VariableInTime_1D() = default;
        virtual ~VariableInTime_1D() = default;

        virtual void update(const state_vector& x, double t) = 0;

        void initialise(int N)
        {
            m_var.resize(N);
        }

        void log(double t)
        {
            file << t;
            for (int i = 0; i < m_var.size(); i++)
                file << "," << m_var[i];
            file << "\n";
        }
    };

} // End Solvex namespace
