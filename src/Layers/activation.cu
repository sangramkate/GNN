#include "activation.hh"
#include "nn_exception.hh"
#include <iostream>
__global__ void ReluActivationForward(float* Z, float* A,float* Stored_Z, int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		A[index] = fmaxf(Z[index], 0);
                Stored_Z[index] = A[index];
	}
}

__global__ void ReluActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < Z_x_dim * Z_y_dim) {
		if (Z[index] > 0) {
			dZ[index] = dA[index];
		}
		else {
			dZ[index] = 0;
		}
	}
}

ReLUActivation::ReLUActivation(std::string name) {
	this->name = name;
}

ReLUActivation::~ReLUActivation() { }

Matrix& ReLUActivation::forward(Matrix& P, bool training, bool freeMatrix) {
        //std::cout << "Relu Layer forward\n";
	this->Z = P;
	A.allocateCuda(Z.shape); 
	stored_Z.allocateCuda(Z.shape);

	dim3 block_size(64);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	ReluActivationForward<<<num_of_blocks, block_size>>>(Z.data_device, A.data_device,stored_Z.data_device, Z.shape.x, Z.shape.y);
      //  std::cout << "Relu forward\n";
	NNException::throwIfDeviceErrorOccurred("Cannot perform ReLU forward propagation.");
        P.freeMem();
        if(training == false){
           stored_Z.freeMem();
        }
	return A;
}

Matrix& ReLUActivation::backprop(Matrix& dA, float learning_rate) {
        //std::cout << "Relu Layer backward\n";
	dZ.allocateCuda(stored_Z.shape);
	dim3 block_size(64);
	dim3 num_of_blocks((stored_Z.shape.y * stored_Z.shape.x + block_size.x - 1) / block_size.x);
	ReluActivationBackprop<<<num_of_blocks, block_size>>>(stored_Z.data_device, dA.data_device,dZ.data_device, stored_Z.shape.x, stored_Z.shape.y);
        //std::cout << "Relu Backward\n"; 	
        NNException::throwIfDeviceErrorOccurred("Cannot perform ReLU back propagation");

        dA.freeMem();
        stored_Z.freeMem();
	return dZ;
}
