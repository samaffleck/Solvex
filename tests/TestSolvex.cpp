#include <iostream>

#include "Solvex/Solvex.h"

void TDMA_SolvesCorrectly()
{
    Eigen::MatrixXd A(4, 4);
    A.setZero();
    A(0, 0) = 1;
    A(0, 1) = 2;
    
    A(1, 0) = 3;
    A(1, 1) = 2;
    A(1, 2) = 1;

    A(2, 1) = 2;
    A(2, 2) = 3;
    A(2, 3) = 1;

    A(3, 2) = 1;
    A(3, 3) = 2;

    Eigen::VectorXd x(4);
    x.setZero();
    x(0) = 0;   // 1
    x(1) = 0;   // -2
    x(2) = 0;   // 3
    x(3) = 0;   // 2.5

    Eigen::VectorXd x_solution(4);
    x.setZero();
    x(0) = 1;  
    x(1) = -2; 
    x(2) = 3;  
    x(3) = 2.5;

    Eigen::VectorXd y(4);
    y(0) = -3;
    y(1) = 2;
    y(2) = 7.5;
    y(3) = 8;

    Solvex::TDMA(A, y, x);

    assert(x.isApprox(x_solution, 1e-6));
}


int main(int argc, char **argv)
{
    TDMA_SolvesCorrectly();
}
