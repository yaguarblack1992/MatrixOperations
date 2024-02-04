#include "Convolutions.h"

Eigen::MatrixXd modifyMatrixReshape(const mx& matrix, const mx& kernel) {
	// size of modified matrix
	int out_rows = matrix.rows() - kernel.rows() + 1;
	int out_cols = matrix.cols() - kernel.cols() + 1;
	int mod_rows = out_rows * out_cols;
	int mod_cols = kernel.size();

	mx modified(mod_rows, mod_cols);

	int rownumber = 0;
	for (int r = 0; r < out_rows; ++r)
		for (int c = 0; c < out_cols; ++c) {
			mx blok = matrix.block(r, c, kernel.rows(), kernel.cols());
			// transpose blok and convert it to vector
			blok.transposeInPlace();
			modified.row(rownumber) = Eigen::Map<Eigen::RowVectorXd>(blok.data(), mod_cols);
			rownumber++;
		}
	return modified;
}

Eigen::MatrixXd modifyReshapedToMatrix(const mx& input, const mx& kernel) {
	// change rows into blocks and place it into matrix
	int kernel_size = kernel.rows();
	int out_size = std::sqrt(input.rows());
	int input_size = out_size + kernel_size - 1;

	mx result(input_size, input_size);
	int rownumber = 0;
	for (int r = 0; r < out_size; r++)
		for (int c = 0; c < out_size; c++)
		{
			//read row from matrix, and convert it to block
			mx row = input.row(rownumber);
			mx blok = Eigen::Map<mx>(row.data(), kernel_size, kernel_size);
			blok.transposeInPlace();
			//save it into matrix
			result.block(r, c, kernel_size, kernel_size) = blok;
			rownumber++;
		}
	return result;
}

Eigen::MatrixXd modifyMatrixToVector(const mx& kernel) {
	// transpose kernel
	mx trans = kernel.transpose();
	// map
	return Eigen::Map<Eigen::VectorXd>(trans.data(), kernel.size());
}

Eigen::MatrixXd modifyVectorToMatrix(const vx& vec) {
	// map into matrix
	int size = vec.size();
	size = std::sqrt(size);

	vx input = vec;
	auto mod = Eigen::Map<Eigen::MatrixXd>(input.data(), size, size);
	// return transposed matrix
	mx mod_transposed = mod.transpose();
	return mod_transposed;
}

Eigen::MatrixXd convolve(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel) {
	int size = input.rows() - kernel.rows() + 1;
	mx result(size, size);
	for (int r = 0; r < size; r++)
		for (int c = 0; c < size; c++) {
			mx blok = input.block(r, c, kernel.rows(), kernel.rows());
			result(r, c) = (blok.array() * kernel.array()).sum();
		}
	return result;
}

Eigen::MatrixXd convolve_multiply(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel) {
	auto mod = modifyMatrixReshape(input, kernel);
	auto new_kernel = modifyMatrixToVector(kernel);
	auto result = mod * new_kernel;
	return modifyVectorToMatrix(result);
}
Eigen::MatrixXd gradientTransfer(const mx& kernel, const mx& gradient) {
	// multiply 
	auto mod_grad = modifyMatrixToVector(gradient);
	auto mod_kernel = modifyMatrixToVector(kernel);
	auto modified = mod_grad * mod_kernel.transpose();

	return modifyReshapedToMatrix(modified, kernel);
}

Eigen::MatrixXd deltaKernel(mx& input, mx& kernel, mx& grad) {
	// calculate delta in kernel weights as in ffnn
	auto mod_input = modifyMatrixReshape(input, kernel);
	auto mod_gradient = modifyMatrixToVector(grad);
	// multiply gradient with input matrix transposed
	mx delta = mod_input.transpose() * mod_gradient;
	// transpose result
	mx delta_transposed = delta.transpose();
	// reshape it
	auto delta_reshaped = modifyReshapedToMatrix(delta_transposed, kernel);
	return delta_reshaped;
}