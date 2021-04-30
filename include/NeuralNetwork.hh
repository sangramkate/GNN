#pragma once

#include <vector>
#include "nn_layers.hh"
#include "costfunction.hh"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	CostFunction bce_cost;

	Matrix Y;
	Matrix dY;
	float learning_rate;

public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();

	Matrix forward(Matrix X, bool training);
	void backprop(Matrix predictions, Matrix target);

	void addLayer(NNLayer *layer);
	std::vector<NNLayer*> getLayers() const;

};
