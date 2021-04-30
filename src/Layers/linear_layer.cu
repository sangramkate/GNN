#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>
#include <cublas_v2.h>

#include "linear_layer.hh"
#include "nn_exception.hh"

__global__ void linearLayerForward( float* W, float* A, float* Z, float* b,
                                                                           int W_x_dim, int W_y_dim,
                                                                           int A_x_dim, int A_y_dim){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
  
    int Z_x_dim = A_x_dim;
    int Z_y_dim = W_x_dim;
  
    float Z_value = 0;
  
    if( row < Z_x_dim && col < Z_y_dim){
       for(int i=0; i< W_y_dim; i=i+1){
           Z_value += W[i + W_y_dim * col] * A[i + A_y_dim * row]; 
       }
       Z[row * Z_y_dim + col] = Z_value + b[col];
      // if(Z[row * Z_y_dim + col] > 0)
      //    printf("Z[%d]: %f\n", row * Z_y_dim + col, Z[row * Z_y_dim + col]);
    }
}

__global__ void linearLayerBackprop( float* W, float* dZ, float*dA,
                                                                    int W_x_dim, int W_y_dim,
                                                                    int dZ_x_dim, int dZ_y_dim){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
  
    int dA_x_dim = dZ_x_dim;
    int dA_y_dim = W_y_dim;
  
    float dA_value = 0.0f;
  	if (row < dA_x_dim && col < dA_y_dim) {
		    for (int i = 0; i < W_x_dim; i++) {
			      dA_value += W[i * W_y_dim + col] * dZ[ i + dZ_y_dim * row];
		    }
		    dA[row * dA_y_dim + col] = dA_value;
	  }
}

__global__ void linearLayerUpdateWeights(  float* dZ, float* A, float* W,
										   int dZ_x_dim, int dZ_y_dim,
										   int A_x_dim, int A_y_dim,
										   float learning_rate) {

	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	// A is treated as transposed
	int W_x_dim = dZ_y_dim;
	int W_y_dim = A_y_dim;

	float dW_value = 0.0f;

	if (row < W_x_dim && col < W_y_dim) {
		for (int i = 0; i < dZ_x_dim; i++) {
		    dW_value += dZ[i * dZ_y_dim + row ] * A[col + A_y_dim * i];
		}
		W[row * W_y_dim + col] = W[row * W_y_dim + col] - learning_rate * (dW_value / A_x_dim);
	}
}

__global__ void linearLayerUpdateBias(  float* dZ, float* b,
										int dZ_x_dim, int dZ_y_dim,
										int b_x_dim,
										float learning_rate) {
	int index = blockIdx.y * blockDim.y + threadIdx.y;

	if (index < dZ_x_dim * dZ_y_dim) {
		int dZ_x = index % dZ_y_dim;
		int dZ_y = index / dZ_y_dim;
		atomicAdd(&b[dZ_y], - learning_rate * (dZ[dZ_y * dZ_y_dim + dZ_x] / dZ_y_dim));
	}
}

LinearLayer::LinearLayer(std::string name, Shape W_shape):
    W(W_shape),b(W_shape.y,1)
{
    this->name = name;
//    std::cout << "updated layer name\n";
    b.allocateCudaMemory();
//    std::cout << "b allocated\n";
    W.allocateMemory();
//    std::cout << "w allocated\n";
    initializeBiasWithZeros();
//   std::cout << "bias initialized\n";
    initializeWeightsRandomly();
//    std::cout << "weights initialized\n";
}

LinearLayer::~LinearLayer()
{ };

void LinearLayer::initializeWeightsRandomly(){
    std::default_random_engine generator;
    std::normal_distribution<float> normal_distribution(0.0, 1.0);
//    std::cout << "W.shape.x:" << W.shape.x <<"\n";	
//    std::cout << "W.shape.y:" << W.shape.y <<"\n";	
    for(int x = 0; x < W.shape.x; x++){
	for(int y = 0 ; y < W.shape.y; y++){
	     W[x * W.shape.y + y] = normal_distribution(generator) * weights_init_threshold;	
	}
    }
//    std::cout << "copying data from host to device\n";
    W.copyHostToDevice();
    free(W.data_host);
}

void LinearLayer::initializeBiasWithZeros() {
	//for (int x = 0; x < b.shape.x; x++) {
	//	b[x] = 0;
	//}
	//b.copyHostToDevice();
        cudaMemset(b.data_device, 0, b.shape.x * b.shape.y* sizeof(float));
}

Matrix& LinearLayer::forward(Matrix& A, bool training, bool freeMatrix){
 //   std::cout << " Linear forward A.x:" << A.shape.x << "\n";
 //  std::cout << " Linear forward A.y:" << A.shape.y << "\n";
 //  std::cout << " Linear forward W.x:" << W.shape.x << "\n";
 //  std::cout << " Linear forward W.y:" << W.shape.y << "\n";
 //   std::cout << " Linear forward A address:" << A.data_device << "\n";
//	std::cout << "input ptr to ll " << A.data_device << "\n";
    assert(W.shape.y = A.shape.y);
  //  std::cout << "Linear layer forward\n";
    //std::cout<< "Linear Layer ptr:" << A.data_device << "\n";
    this->A = A;
    //std::cout<< "Linear Layer ptr:" << A.data_device << "\n";
    Shape Z_shape(A.shape.x,W.shape.x);
    Z.allocateCuda(Z_shape);
//    printf("A.shape.x %d A.shape.y %d\n", A.shape.x, A.shape.y);
//    printf("W.shape.x %d W.shape.y %d\n", W.shape.x, W.shape.y);
//    printf("Z.shape.x %d Z.shape.y %d\n", Z.shape.x, Z.shape.y);
    computeAndStoreLayerOutput(A);
//    std::cout << "Linear Layer forward\n";
    NNException::throwIfDeviceErrorOccurred("Cannot perform Linear Layer forward propagation");
    
//    std::cout << " Linear forward shape.x:" << Z.shape.x << "\n";
//    std::cout << " Linear forward shape.y:" << Z.shape.y << "\n";
//    std::cout << " Linear forward A shape.x:" << A.shape.x << "\n";
//    std::cout << " Linear forward A shape.y:" << A.shape.y << "\n";
//    std::cout << " Linear forward A address:" << A.data_device << "\n";
    //if(training == false)
    //    A.freeMem();
    return Z;
	
}
void LinearLayer::computeAndStoreLayerOutput(Matrix& A) {
dim3 block_size(32,32);
dim3 num_of_blocks(((Z.shape.x + block_size.x - 1) / block_size.x),((Z.shape.y + block_size.y - 1) / block_size.y) );

//cublasStatus_t result1;
//cublasHandle_t handle;
//cublasCreate(&handle);
//const float alp = 1;
//const float bet = 1;
//const float* alpha = &alp;
//const float* beta = &bet;
//result1 = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 1433, 7, 1000, alpha, A.data_device, 1433, W.data_device, 7, beta, Z.data_device,  1000);
//if(result1 != cudaSuccess)
//    printf("SGEMM failed\n");

linearLayerForward<<<num_of_blocks, block_size>>>( W.data_device,
				                   A.data_device,
						   Z.data_device,
						   b.data_device,
						   W.shape.x, W.shape.y,
						   A.shape.x, A.shape.y);
}

Matrix& LinearLayer::backprop(Matrix& dZ, float learning_rate) {
  //      std::cout << "Linear layer backword\n";
//	dA.allocateCuda(A.shape);

      //  std::cout << "Linear Layer backward\n";
//	printf("ll back in dZ shape %d shape %d\n", dZ.shape.x ,dZ.shape.y);
//	computeAndStoreBackpropError(dZ);
	NNException::throwIfDeviceErrorOccurred("Cannot perform back propagation.");

	updateBias(dZ, learning_rate);
	NNException::throwIfDeviceErrorOccurred("Cannot perform bias update.");
        
        //std::cout << " A ptr: " << A.data_device << "\n";
        //std::cout << " A last :" << A.data_device + (A.shape.x *  A.shape.y * 4) << "\n";  
        //std::cout << " dZ ptr: " << dZ.data_device << "\n";
        //std::cout << " dZ last :" << dZ.data_device + (dZ.shape.x *  dZ.shape.y * 4) << "\n";  
        //std::cout << " Linear backward shape dZ.x:" << dZ.shape.x << "\n";
        //std::cout << " Linear backward shape dZ.y:" << dZ.shape.y << "\n";
        //std::cout << " Linear backward shape A.x:" << A.shape.x << "\n";
        //std::cout << " Linear backward shape A.y:" << A.shape.y << "\n";
	updateWeights(dZ, learning_rate);
	NNException::throwIfDeviceErrorOccurred("Cannot perform weights update.");

        //std::cout << " Linear backward shape.x:" << dA.shape.x << "\n";
        //std::cout << " Linear backward shape.y:" << dA.shape.y << "\n";
  //      dZ.freeMem();
	//printf("ll back out dZ shape %d shape %d\n", dA.shape.x, dA.shape.y);
        //if(A.device_allocated == true) A.freeMem();
	return dZ;
}

void LinearLayer::computeAndStoreBackpropError(Matrix& dZ) {
	dim3 block_size(32, 32);
	dim3 num_of_blocks((A.shape.x + block_size.x - 1) / block_size.x,(A.shape.y + block_size.y - 1) / block_size.y);
	linearLayerBackprop<<<num_of_blocks, block_size >>> ( W.data_device,
							     dZ.data_device,
							     dA.data_device,
							     W.shape.x, W.shape.y,
							     dZ.shape.x, dZ.shape.y);
}

void LinearLayer::updateWeights(Matrix& dZ, float learning_rate) {
	dim3 block_size(32, 32);
	dim3 num_of_blocks((W.shape.x + block_size.x - 1) / block_size.x,(W.shape.y + block_size.y - 1) / block_size.y);
	linearLayerUpdateWeights<<<num_of_blocks, block_size>>>(dZ.data_device,
								A.data_device,
								W.data_device,
								dZ.shape.x, dZ.shape.y,
								A.shape.x, A.shape.y,
								learning_rate);
}

void LinearLayer::updateBias(Matrix& dZ, float learning_rate) {
	dim3 block_size(512);
	dim3 num_of_blocks( (dZ.shape.y * dZ.shape.x + block_size.x - 1) / block_size.x);
	linearLayerUpdateBias<<<num_of_blocks, block_size>>>(dZ.data_device,
							     b.data_device,
							     dZ.shape.x, dZ.shape.y,
							     b.shape.x, learning_rate);
}

int LinearLayer::getXdim() const {
	return W.shape.x;
}

int LinearLayer::getYdim() const {
	return W.shape.y;
}

Matrix LinearLayer::getWeightsMatrix() const {
	return W;
}

Matrix LinearLayer::getBiasVector() const {
	return b;
}

void LinearLayer::setData(int* row, int* col) {
}
	    
