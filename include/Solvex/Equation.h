#pragma once

#include "eigen3/Eigen/Dense"


namespace Solvex
{
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
        std::function<void(const Eigen::VectorXd& x, Eigen::VectorXd& Fx)> f;

        void updateFx();
        double getError();
        void approximateJ();
    };
  
} // End Solvex namespace
