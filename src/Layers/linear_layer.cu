#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>
#include <cublas_v2.h>

#include "linear_layer.hh"
#include "nn_exception.hh"

#define LEARNING_RATE 0.01

__global__ void ReluActivationForward(float* Z, float* A,float* Stored_Z, int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		A[index] = fmaxf(Z[index], 0);
                Stored_Z[index] = A[index];
	}
}


__global__ void ReluActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {
	
	//int nnodes = 2708;
	//int num_test_nodes = nnodes - (0.6*nnodes);
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < Z_x_dim * Z_y_dim) {
		if (Z[index] > 0) {
			dZ[index] = dA[index];
		}
		else {
			dZ[index] = 0;
		}
		//Adding it to quickly see if I can set output of node agg 0 for test nodes
		/*
		if(index < num_test_nodes) {
			dZ[index] = 0;
		}*/
	}
	/*
	if((row > 2700)) {
		printf("ReLU x = %d, y = %d, dZ = %f, dA = %f\n", row, i, dZ[i + dA_y_dim * row], dA[i + dA_y_dim * row]); 
	}*/
}
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

__global__ void linearLayerForwardAddBias( float* Z, float* bias, int numFeatures) {

    // APARNA TODO: fuse bias addition and reLU application
    // APARNA TODO: if this takes a lot of time -- can merge computations for some features like fuseGNN
    //Add Z: #nodes * #labels , b: labels * 1 (or 1 * labels) doesn't matter
  
    //APARNA TODO: maybe doing an inner loop where we process > 1 node per CTA will help  -- will reduce launch overhead

    /*
    for(int feature = threadIdx.x ; feature < numFeatures; feature += blockDim.x) {
	Z[blockIdx.x * numFeatures + feature] = Z[blockIdx.x * numFeatures + feature] + bias[feature];
    }*/
	
    Z[blockIdx.x * numFeatures + threadIdx.x] = Z[blockIdx.x * numFeatures + threadIdx.x] + bias[threadIdx.x];
    
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
			      dA_value += -1 * W[i * W_y_dim + col] * dZ[ i + dZ_y_dim * row];
		    }
		    dA[row * dA_y_dim + col] = dA_value;
	  }
}

__global__ void linearLayerUpdateWeights(  float* W, float* dW,
						int W_x_dim, int W_y_dim,
						float learning_rate) {

	//W = W - (n) * dW

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if( x < W_x_dim && y < W_y_dim) {
	    W[x * W_y_dim + y] += (-1) * (learning_rate) * dW[x * W_y_dim + y];
	}
}

/*
//Reduces mxn array into 1xm array
__global__ void reduce_array(volatile scalar_t* sdata, unsigned int tid, unsigned int reduce_len, unsigned int f_dim){

    __shared__ scalar_t s_feature[blockSize];


    while (reduce_len > 1){
        __syncthreads();
        // add the remainer
        if ((tid < f_dim) && (reduce_len % 2 == 1)){
            sdata[tid] += sdata[tid + f_dim * (reduce_len - 1)];
        }
        reduce_len /= 2;
        if (tid < f_dim * reduce_len){
            sdata[tid] += sdata[tid + f_dim * reduce_len];
        }
    }
}
*/

__global__ void linearLayerUpdateBias(  float* dZ, float* b,
										int dZ_x_dim, int dZ_y_dim,
										int b_x_dim,
										float learning_rate) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < dZ_x_dim * dZ_y_dim) {
		int dZ_x = index % dZ_y_dim;
		int dZ_y = index / dZ_y_dim;
		atomicAdd(&b[dZ_y], - learning_rate * (dZ[dZ_y * dZ_y_dim + dZ_x] / dZ_y_dim));
	}
}


//__global__ void linearLayerUpdateBias(  float* dZ, float* b,
//										int dZ_x_dim, int dZ_y_dim,
//										int b_x_dim,
//										float learning_rate) {
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//	int tid = threadIdx.x;
//
//	//Setting to a high value
//	extern __shared__ float buffer[];
//	
//	//Assuming #of output features > # of threads
//	if(tid < dZ_y_dim) {
//		buffer[tid] = 0;
//	}
//	__syncthreads();
//
//	if (index < dZ_x_dim * dZ_y_dim) {
//		int dZ_x = index / dZ_y_dim;
//		int dZ_y = index % dZ_y_dim;
//		atomicAdd(&buffer[dZ_y], dZ[dZ_x * dZ_y_dim + dZ_y]);
//	}
//
//	__syncthreads();
//
//	if(tid < dZ_y_dim) {
//		atomicAdd(&b[tid], -learning_rate*buffer[tid]/dZ_y_dim);
//	}
//}


void LinearLayer::runGEMM(Matrix& A, Matrix& B, Matrix& C, bool transposeA, bool transposeB) {
	//The take transpose function is for back propagation --> we multiply A.B' instead of A.B if this is turned on

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	// Do the actual multiplication
	//alpha * op(A) * op(B) + beta * OP(C)
	// C(m,n) = A(m,k) * B(k,n)

	int m = C.shape.x;
	int n = C.shape.y;
	int k = transposeA ? B.shape.x : A.shape.y;

	//int lda=k,ldb=n,ldc=n;
	int lda=transposeA ? m : k;  //mxk
	//int ldb= n; //transposeB ? n : k;  //  kxn
	int ldb= transposeB ? k : n;  //  kxn
	int ldc=n; //mxn

	const float alf = 1;
	const float bet = 0;

	const float *alpha = &alf;
	const float *beta = &bet;

	//Note: This function can't support the case when both transposeA and B are set to 1	
	cublasSgemm(handle, 
		    transposeB ? CUBLAS_OP_T : CUBLAS_OP_N, 
		    transposeA ? CUBLAS_OP_T : CUBLAS_OP_N, 
		    n, m, k, alpha, B.data_device, ldb, A.data_device, lda, beta, C.data_device, ldc);
	
	//print_kernel<<<1,1>>>(Z.data_device);

	// Destroy the handle
	cublasDestroy(handle);
}

LinearLayer::LinearLayer(std::string name,int layer_num, Shape W_shape):
    W(W_shape),b(W_shape.y,1),dW(W_shape)
{
    this->name = name;
    this->layer_num = layer_num;
//    std::cout << "updated layer name\n";
    b.allocateCudaMemory();
//    std::cout << "b allocated\n";
    W.allocateMemory();
    dW.allocateMemory();
//    std::cout << "w allocated\n";
    initializeBiasWithZeros();
//   std::cout << "bias initialized\n";
    initializeWeightsRandomly();
//    std::cout << "weights initialized\n";
}

void LinearLayer::free_matrix(){
  dW.freeMem();
}

LinearLayer::~LinearLayer()
{ };

void LinearLayer::initializeWeightsRandomly(){
    std::default_random_engine generator;
    std::normal_distribution<float> normal_distribution(0.0, 0.1);
//    std::cout << "W.shape.x:" << W.shape.x <<"\n";	
//    std::cout << "W.shape.y:" << W.shape.y <<"\n";	
    for(int x = 0; x < W.shape.x; x++){
	for(int y = 0 ; y < W.shape.y; y++){
	     W[x * W.shape.y + y] = normal_distribution(generator)*0.1;	
	     //printf("W[%d] = %f\n", (x * W.shape.y + y), W[x * W.shape.y + y]);
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

__global__ void print_kernel_lin(float *A, int size, std::string str) {
	for(int i=0; i<size; i++) {
		if(A[i] != 0.0) {
		    printf("The value of %s[%d] = %f\n", str, i, A[i]);
		}
	}
}


Matrix& LinearLayer::forward(Matrix& A, bool training, bool freeMatrix){
//   std::cout << " Linear forward A.x:" << A.shape.x << "\n";
//  std::cout << " Linear forward A.y:" << A.shape.y << "\n";
//  std::cout << " Linear forward W.x:" << W.shape.x << "\n";
//  std::cout << " Linear forward W.y:" << W.shape.y << "\n";
//   std::cout << " Linear forward A address:" << A.data_device << "\n";
    assert(W.shape.x = A.shape.y);
   // std::cout << "Linear layer forward\n";
    //std::cout<< "Linear Layer ptr:" << A.data_device << "\n";
    this->A = A;

    //std::cout<< "Linear Layer ptr:" << A.data_device << "\n";
    Shape Z_shape(A.shape.x,W.shape.y);
    Z.allocateCuda(Z_shape);
    computeAndStoreLayerOutput(A);
    if(layer_num != 2){
	stored_Z.allocateCuda(Z.shape);
	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	ReluActivationForward<<<num_of_blocks, block_size>>>(Z.data_device, Z.data_device,stored_Z.data_device, Z.shape.x, Z.shape.y);
        
    }
//    std::cout << "Linear Layer forward\n";
    NNException::throwIfDeviceErrorOccurred("Cannot perform Linear Layer forward propagation");
    
//    std::cout << " Linear forward shape.x:" << Z.shape.x << "\n";
//    std::cout << " Linear forward shape.y:" << Z.shape.y << "\n";
//    std::cout << " Linear forward A shape.x:" << A.shape.x << "\n";
//    std::cout << " Linear forward A shape.y:" << A.shape.y << "\n";
//    std::cout << " Linear forward A address:" << A.data_device << "\n";
    if(training == false) {
	if(freeMatrix) {
            A.freeMem();
	}
     }

    return Z;
	
}


__global__ void print_weight_sum(float *W, float *dW, int size) {
	float w_sum = 0;
	float dw_sum = 0;

	float w_sum_mod = 0;
	float dw_sum_mod = 0;
	for(int i=0; i<size; i++) {
		w_sum += W[i];
		dw_sum += dW[i];
		w_sum_mod +=   (W[i] > 0) ?  W[i] :  -W[i]; 
		dw_sum_mod += (dW[i] > 0) ? dW[i] : -dW[i]; 
	}

	printf("The value of Weight Sum = %f, dW sum = %f\n", w_sum, dw_sum);
	printf("The value of MOD Weight Sum = %f, dW sum = %f\n", w_sum_mod, dw_sum_mod);
}

void LinearLayer::computeAndStoreLayerOutput(Matrix& A) {

	runGEMM(A, W, Z, false, false);	
	//Num CTAs = #nodes, #threads = min(256, numFeatures)
	int threadsPerBlock = std::min(256, (int) W.shape.y);
	linearLayerForwardAddBias<<<(Z.shape.x + threadsPerBlock - 1)/threadsPerBlock, threadsPerBlock>>>(Z.data_device, b.data_device, Z.shape.y);
	
}

Matrix& LinearLayer::backprop(Matrix& dZ, float learning_rate, bool freeMatrix) {
      //  std::cout << "Linear layer backword\n";
        if(layer_num != 2){
         	dim3 block_size(256);
         	dim3 num_of_blocks((stored_Z.shape.y * stored_Z.shape.x + block_size.x - 1) / block_size.x);
         	ReluActivationBackprop<<<num_of_blocks, block_size>>>(stored_Z.data_device, dZ.data_device,dZ.data_device, stored_Z.shape.x, stored_Z.shape.y);
                NNException::throwIfDeviceErrorOccurred("Cannot perform ReLU back propagation");

        } 

	dA.allocateCuda(A.shape);
	//dW.allocateCuda(W.shape); //A'.dZ

      //  std::cout << "Linear Layer backward\n";
	//print_kernel_lin<<<1,1>>>(dZ.data_device, dZ.shape.x*dZ.shape.y, "dZ - pre backprop ");
	computeAndStoreBackpropError(dZ);
	NNException::throwIfDeviceErrorOccurred("Cannot perform back propagation.");

	/*
	if(dZ.shape.y == 32) {
	    printf("Printing dZ of lin 1 layer\n");
	    print_kernel_lin<<<1,1>>>(dZ.data_device, dZ.shape.x*dZ.shape.y, "dZ - in backprop ");
	}*/
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
        stored_Z.freeMem();
        //dZ.freeMem();
	//dW.freeMem();
        if(A.device_allocated == true){
            if(freeMatrix){
               A.freeMem();
            }
        }
	return dA;
}

void LinearLayer::computeAndStoreBackpropError(Matrix& dZ) {

	//std::cout << "dZ.x = " << dZ.shape.x << ", dZ.y = " << dZ.shape.y << std::endl;
	//std::cout << "dA.x = " << dA.shape.x << ", dA.y = " << dA.shape.y << std::endl;

	//W: 10x7, dz: 2708x7, dA: 2708x10 
	// So dA = dz.W'
	runGEMM(dZ, W, dA, false, true);	
	//TODO: need to multiply dA with -1. <<< Are we sure??? -- why not do that in dZ calculation?>>>

	/*
	if(dZ.shape.y == 7) {
		printf("Printing dA\n");
		print_kernel_lin<<<1,1>>>(dA.data_device, dA.shape.x*dA.shape.y, "dA ");
		cudaDeviceSynchronize();
	}*/

}

void LinearLayer::updateWeights(Matrix& dZ, float learning_rate) {

	//dW = A'.dZ
	//dw: 10x7, A: 2708x10, dZ: 2708x7	
	runGEMM(A, dZ, dW, true, false);	

	//print_weight_sum<<<1,1>>>(W.data_device, dW.data_device, W.shape.x*W.shape.y);
	
	//Weight size is 1433x16 and 16x7	

	//W = W - (n) * dW
	dim3 block_size(16, 16);
	dim3 num_of_blocks((W.shape.x + block_size.x - 1) / block_size.x,(W.shape.y + block_size.y - 1) / block_size.y);
	linearLayerUpdateWeights<<<num_of_blocks, block_size>>>(W.data_device,
								dW.data_device,
								W.shape.x, W.shape.y,
								learning_rate);
}

void LinearLayer::updateBias(Matrix& dZ, float learning_rate) {

	//db: 1x7
	//The operation is dB = dZ.(reduce in Xdim) so 2708x7 --> 1x7 
	//Then b = b - (n) * dB
	
	//Need to write a reduction kernel for the first line
	//print_kernel_lin<<<1,1>>>(dZ.data_device, dZ.shape.x*dZ.shape.y, "dZ - pre bias ");


	dim3 block_size(512);
	dim3 num_of_blocks( (dZ.shape.y * dZ.shape.x + block_size.x - 1) / block_size.x);
	linearLayerUpdateBias<<<num_of_blocks, block_size, dZ.shape.y>>>(dZ.data_device,
							     b.data_device,
							     dZ.shape.x, dZ.shape.y,
							     b.shape.x, learning_rate);

	//printf("Bias X: %d, Y: %d\n", b.shape.x, b.shape.y);
	//print_kernel_lin<<<1,1>>>(b.data_device, b.shape.x*b.shape.y, "bias");
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
	    
