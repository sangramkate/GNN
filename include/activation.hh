#pragma once

#include "nn_layers.hh"

class ReLUActivation : public NNLayer {
private:
	Matrix A;

	Matrix Z;
	Matrix dZ;
        Matrix stored_Z;

public:
	ReLUActivation(std::string name);
	~ReLUActivation();

	Matrix& forward(Matrix& Z, bool training, bool freeMatrix);
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};
