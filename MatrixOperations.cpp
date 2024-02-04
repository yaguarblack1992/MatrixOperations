#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "Convolutions.h"




int main() {
	srand(time(NULL));
	mx input(5, 5);
	mx kernel1(3, 3), kernel2(3, 3);
	mx target(1,1);

	input <<
		1, 0, 1, 0, 1,
		1, 1, 1, 0, 1,
		0, 0, 0, 0, 1,
		1, 1, 1, 1, 1,
		0, 0, 0, 0, 1;

	kernel1 <<
		1, 0, -1,
		0, -1, 0,
		0, 0, 0;

	target <<
		0.0;
	
	// learning process
	double eta = 0.01;
	for (int i = 0; i < 100; ++i) {
		// convolve
		mx out1 = convolve(input, kernel1);
		mx out2 = convolve(out1, kernel1);
		// calculate gradient
		mx grad = out2 - target;
		// show MSE
		std::cout << "Iteration " << i << " MSE= " << grad.array().square().sum() << '\n';
		// calculate delta in each steps

		mx delta2 = deltaKernel(out1, kernel1, grad);
		// calculate new grad
		auto grad1 = gradientTransfer(kernel1, grad);
		mx delta1 = deltaKernel(input, kernel1, grad1);
		// update weights
		kernel1 = kernel1 - delta1 * eta - delta2 * eta;
	}

	std::cout << kernel1 << "\n\n" << "\n\n";

	std::cout << convolve(input, kernel1) << "\n\n";
	std::cout << convolve(convolve(input, kernel1), kernel1);

	return 0;
}