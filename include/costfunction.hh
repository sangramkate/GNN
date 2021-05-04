#pragma once
#include "matrix.hh"

class CostFunction {
public:
	float cost(Matrix& predictions, Matrix& target, int *node_array_device, int num_test_nodes);
	Matrix& dCost(Matrix& predictions, Matrix& target, Matrix& dY, int *node_array_device, int num_test_nodes);
};
