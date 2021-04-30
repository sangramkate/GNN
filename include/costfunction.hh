#pragma once
#include "matrix.hh"

class CostFunction {
public:
	float cost(Matrix& predictions, Matrix& target);
	Matrix& dCost(Matrix& predictions, Matrix& target, Matrix& dY);
};
