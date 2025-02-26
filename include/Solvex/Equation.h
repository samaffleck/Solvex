#pragma once

#include "Eigen/Dense"

#include <vector>

namespace Solvex
{
    struct Subsystem
    {
        std::size_t dimension = 0;
        std::function<void(
            const Eigen::Ref<const Eigen::VectorXd>& myX,
            const std::vector<Eigen::Ref<const Eigen::VectorXd>>& otherX,
            Eigen::Ref<Eigen::VectorXd> dmyX
        )> compute;
    };

    struct Equation
    {
        Equation(int _N);
        ~Equation() = default;

        int N;
        int num_of_sub_diag = 1;
        int num_of_sup_diag = 1;
        Eigen::MatrixXd J;
        Eigen::VectorXd x;
        Eigen::VectorXd Fx;
    };
  
    struct ODEquation
    {
        ODEquation(int _N);
        ~ODEquation() = default;

        int N;                  // Number of unknowns. i.e. size of x
        int num_of_sub_diag = 1;// Number of non-zero sub diagonals
        int num_of_sup_diag = 1;// Number of non-zero super diagonals
        Eigen::MatrixXd J;      // Jacobian matrix
        Eigen::VectorXd x;      // Vector of unknowns
        Eigen::VectorXd x_dt;   // previous time step
        Eigen::VectorXd Fx;     // Resudual vector
        Eigen::VectorXd dx_dt;  // Time derivative
    };

} // End Solvex namespace
