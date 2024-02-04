#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "Convolutions.h"




int main() {
	mx input(4, 4);
	mx kernel(3, 3);
	mx target(2, 2);

	input <<
		1, 0, 1, 0,
		1, 1, 1, 0,
		0, 0, 0, 0,
		1, 1, 1, 1;

	kernel <<
		1, 0, 1,
		0, -1, 0,
		1, 0, 1;

	target <<
		1.0, 0.5,
		0.0, 0.0;
	
	// learning process
	double eta = 0.01;
	for (int i = 0; i < 100; ++i) {
		// convolve
		mx out = convolve(input, kernel);
		// calculate gradient
		mx grad = out - target;
		// show MSE
		std::cout << "Iteration " << i << " MSE= " << grad.array().square().sum() << '\n';
		// calculate delta
		mx delta = deltaKernel(input, kernel, grad);
		// update weights
		kernel = kernel - delta * eta;
	}

	return 0;
}