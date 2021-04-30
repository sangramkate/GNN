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
	        cnt++;
        }

	Y = Z;
	return Y;
}

void NeuralNetwork::backprop(Matrix predictions, Matrix target) {
//	std::cout << "dY allocated device:" << dY.device_allocated << "\n";
	//std::cout << "pred shape: " << predictions.shape.y << "\n";
	//std::cout << "tar shape: " << target.shape.y << "\n";
        dY.allocateMemoryIfNotAllocated(predictions.shape);
	Matrix& error = bce_cost.dCost(predictions, target, dY);
        //std::cout << "Error.x" = error.shape.x << "\n";
        //std::cout << "Error.y" = error.shape.y << "\n";

	for (auto it = this->layers.rbegin(); it != this->layers.rend(); it++) {
		error = (*it)->backprop(error, learning_rate);
	}
        //error.freeMem();
        dY.freeMem();
	cudaDeviceSynchronize();
}

void NeuralNetwork::NodeAggSetData(int* row, int* col) {

	for(auto layer : layers) {
	    if((layer->getName() == "nodeagg1") || (layer->getName() == "nodeagg3"))
		layer->setData(row,col);
	}
} 
std::vector<NNLayer*> NeuralNetwork::getLayers() const {
	printf("yo");
//	for(auto layer: layers) {
//	std::cout << layer.name() << "\n";
//	}
	return layers;
}
