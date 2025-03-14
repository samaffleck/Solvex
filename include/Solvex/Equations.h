#pragma once

// Solvex includes
#include "Solvex/BoundaryConditions.h"

// STL includes
#include <iostream>
#include <fstream>
#include <memory>

namespace Solvex
{
    struct SegmentIndex
    {
        int startIndex{};
        int endIndex{};
    };

    struct Equation
    {
        std::ofstream                       file;           // CSV file for log data
        std::unique_ptr<BoundaryCondition>  top_bc;        // Top boundary condition
        std::unique_ptr<BoundaryCondition>  bot_bc;        // Bottom boundary condition
        double                              t0{};          // Initial condition
        SegmentIndex                        index;         // Start and end index

        void log(const state_vector& x, const double t);
        size_t getSize() const;
        void setInitialCondition(state_vector& x) const;
        void setMassMatrix(sparse_matrix& M) const;
    };

    struct SystemOfEquation
    {
        Equation T;     // Temperature
        Equation P;     // Pressure
        Equation U;     // Velocity
        int_type Nvar = 3;  // Number of variables to solve for

        size_t getSize() const;
        void initialise(const std::string& name_id);
        void cleanUp();
    };

} // End Solvex namespace
