#pragma once
#include <Eigen/Dense>
#include <vector>

using mx = Eigen::MatrixXd;
using vx = Eigen::VectorXd;

Eigen::MatrixXd modifyMatrixReshape(const mx& matrix, const mx& kernel);
Eigen::MatrixXd modifyReshapedToMatrix(const mx& input, const mx& kernel);
Eigen::MatrixXd modifyMatrixToVector(const mx& kernel);
Eigen::MatrixXd modifyVectorToMatrix(const vx& vec);
Eigen::MatrixXd convolve(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel);
Eigen::MatrixXd convolve_multiply(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel);
Eigen::MatrixXd gradientTransfer(const mx& kernel, const mx& gradient);
Eigen::MatrixXd deltaKernel(mx& input, mx& kernel, mx& grad);

Eigen::MatrixXd learnKernel(mx& input, mx& kernel, mx& target, int iterations = 500, double eta=0.01);