#include "NeuralNetwork.hh"
#include "nn_exception.hh"

NeuralNetwork::NeuralNetwork(float learning_rate) :
	learning_rate(learning_rate)
{ }

NeuralNetwork::~NeuralNetwork() {
	for (auto layer : layers) {
		delete layer;
	}
}

void NeuralNetwork::addLayer(NNLayer* layer) {
	this->layers.push_back(layer);
}

Matrix NeuralNetwork::forward(Matrix X, bool training) {
	Matrix Z = X;
        int cnt = 0;
        bool freeMatrix;
	for (auto layer : layers) {
                if(cnt == 0)
                    freeMatrix = false;
                else
                    freeMatrix = true;
		Z = layer->forward(Z,training,freeMatrix);
                cudaDeviceSynchronize();
	        cnt++;
        }

	Y = Z;
	return Y;
}

void NeuralNetwork::backprop(Matrix predictions, Matrix target, int* node_array_device, int num_test_nodes) {
//	std::cout << "dY allocated device:" << dY.device_allocated << "\n";
        dY.allocateMemoryIfNotAllocated(predictions.shape);
	Matrix& error = bce_cost.dCost(predictions, target, dY, node_array_device, num_test_nodes);
        //std::cout << "Error.x = " << error.shape.x << "\n";
        //std::cout << "Error.y = " << error.shape.y << "\n";

	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		error = (*it)->backprop(error, learning_rate);
                cudaDeviceSynchronize();
	}
        //error.freeMem();
        dY.freeMem();
	cudaDeviceSynchronize();
}

std::vector<NNLayer*> NeuralNetwork::getLayers() const {
	return layers;
}
