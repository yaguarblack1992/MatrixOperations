#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "Convolutions.h"




int main() {
	srand(time(NULL));
	mx input(5, 5);
	mx kernel(3, 3);
	mx target(1,1);
	// some input pattern
	input <<
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0,
		0, 1, 1, 1, 0,
		0, 1, 0, 1, 0,
		0, 1, 0, 1, 0;
	// random kernel
	kernel.setRandom();

	target <<
		1.0;
	
	// learning process
	double eta = 0.01;
	for (int i = 0; i < 100; ++i) {
		// convolve
		mx out1 = convolve(input, kernel);
		mx out2 = convolve(out1, kernel);
		// calculate gradient
		mx grad = out2 - target;
		// show MSE
		std::cout << "Iteration " << i << " MSE= " << grad.array().square().sum() << '\n';
		// calculate delta in each steps

		mx delta2 = deltaKernel(out1, kernel, grad);
		// calculate new grad
		auto grad1 = gradientTransfer(kernel, grad);
		mx delta1 = deltaKernel(input, kernel, grad1);
		// update weights
		kernel = kernel - delta1 * eta - delta2 * eta;
	}
	std::cout << "Learned kernel\n";
	std::cout << kernel << "\n\n" << "\n\n";
	std::cout << "Convolution 1\n";
	std::cout << convolve(input, kernel) << "\n\n";
	std::cout << "Convolution 2\n";
	std::cout << convolve(convolve(input, kernel), kernel) << "\n\n";

	mx kernel5(5, 5);
	kernel5.setRandom();
	std::cout << "kernel 5x5 random\n" << kernel5 << "\n\n";
	std::cout << "convolution with random kernel\n" << convolve(input, kernel5) << "\n\n";
	std::cout << "Kernel 5x5 learned\n";
	mx kernel5_learned = learnKernel(input, kernel5, target, 200, 0.01);
	std::cout << kernel5_learned << "\n\n";
	std::cout << "convolved:\n" << convolve(input, kernel5_learned);
	return 0;
}