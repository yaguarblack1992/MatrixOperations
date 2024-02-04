#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "Convolutions.h"

std::vector <mx> patterns;


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
	// vector with patterns
	mx input1(5, 5), input2(5, 5), input3(5, 5);
	input1 <<
		0, 0, 0, 0, 0,
		1, 1, 1, 0, 0,
		0, 0, 1, 1, 1,
		1, 1, 1, 0, 0,
		0, 0, 0, 0, 0;
	input2 <<
		0, 0, 0, 0, 0,
		0, 0, 1, 1, 1,
		1, 1, 1, 0, 0,
		0, 0, 1, 1, 1,
		0, 0, 0, 0, 0;
	input3 <<
		0, 1, 0, 1, 0,
		0, 1, 0, 1, 0,
		0, 1, 1, 1, 0,
		0, 0, 1, 0, 0,
		0, 0, 1, 0, 0;

	patterns = { input,input1,input2,input3 };

	// gonna try to recognize pattern no matter of its orientation
	// some parameters
	int iterations = 300;
	double eta = 0.01;

	std::cout << "Set parameters\n" << "Iterations ";
	std::cin >> iterations;
	std::cout << "Learning rate ";
	std::cin >> eta;
	// learning process
	for (int i = 0; i < iterations; ++i) {
		// convolve
		mx out1 = convolve(patterns[i%4], kernel);
		mx out2 = convolve(out1, kernel);
		// calculate gradient
		mx grad = out2 - target;
		// show MSE
		std::cout << "Iteration " << i << " MSE= " << grad.array().square().sum() << '\n';
		// calculate delta in each steps

		mx delta2 = deltaKernel(out1, kernel, grad);
		// calculate new grad
		auto grad1 = gradientTransfer(kernel, grad);
		mx delta1 = deltaKernel(patterns[i%4], kernel, grad1);
		// update weights
		kernel = kernel - delta1 * eta - delta2 * eta;
	}
	std::cout << "Learned kernel\n";
	std::cout << kernel << "\n\n" << "\n\n";
	std::cout << "Convolution 1\n";
	std::cout << convolve(input, kernel) << "\n\n";
	std::cout << "Convolution 2\n";
	std::cout << convolve(convolve(input, kernel), kernel) << "\n\n";

	
	// now we add some random matrices to learning process
	std::cout << "With random noisy matrices\n\n";
	
	for (int i = 0; i < iterations; ++i) {
		// generate random matrix each second iteration and use as input, set target accordingly
		mx input_matrix(patterns[0]);
		if (i % 2 == 0)
			input_matrix = patterns[rand() % 4], target << 1.0;
		else
			input_matrix.setRandom(), target << 0.0;
		// convolve
		mx out1 = convolve(input_matrix, kernel);
		mx out2 = convolve(out1, kernel);
		// calculate gradient
		mx grad = out2 - target;
		// show MSE
		std::cout << "Iteration " << i << " MSE= " << grad.array().square().sum() << '\n';
		// calculate delta in each steps

		mx delta2 = deltaKernel(out1, kernel, grad);
		// calculate new grad
		auto grad1 = gradientTransfer(kernel, grad);
		mx delta1 = deltaKernel(input_matrix, kernel, grad1);
		// update weights
		kernel = kernel - delta1 * eta - delta2 * eta;
	}

	std::cout << "Learned kernel\n";
	std::cout << kernel << "\n\n" << "\n\n";
	std::cout << "Convolution 1\n";
	std::cout << convolve(input, kernel) << "\n\n";
	std::cout << "Convolution 2\n";
	std::cout << convolve(convolve(input, kernel), kernel) << "\n\n";
	mx random(5, 5);
	random.setRandom();
	std::cout << "Convolution with random matrix\n";
	std::cout << convolve(random, kernel) << "\n\n";
	std::cout << "Convolution 2 with random matrix\n";
	std::cout << convolve(convolve(random, kernel), kernel) << "\n\n";

	return 0;
}