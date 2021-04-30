#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "softmax.hh"
#include "nn_exception.hh"

__global__ void SoftMaxForward( float* A, float* Z,int A_x_dim, int A_y_dim){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
  
    int Z_x_dim = A_x_dim;
    int Z_y_dim = A_y_dim;
  
    float sum = 0.0f;
    if(row < Z_x_dim){
       for(int i=0; i< Z_y_dim; i=i+1){
           float tmp = exp(A[i +  Z_y_dim * row]);
           Z[i + Z_y_dim * row] = tmp;
           sum += tmp;  
       }
       for(int i= 0; i < Z_y_dim; i=i+1){
           Z[i + Z_y_dim * row] /= sum;
       }
    }
}
__global__ void SoftMaxBackprop( float* dZ, float*dA, float* A, int dZ_x_dim, int dZ_y_dim){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
  
    int dA_x_dim = dZ_x_dim;
    int dA_y_dim = dZ_y_dim;
  
    float sum = 0.0f;
  	if (row < dA_x_dim) {
            for(int i=0; i< dA_y_dim; i=i+1){
                dA[i + dA_y_dim * row] = A[i + dA_y_dim * row] - dZ[i + dA_y_dim * row];
            }
        }
}

//__global__ void SoftMaxBackprop( float* dZ, float*dA, float* A, int dZ_x_dim, int dZ_y_dim){
//    int row = blockIdx.x * blockDim.x + threadIdx.x;
//  
//    int dA_x_dim = dZ_x_dim;
//    int dA_y_dim = dZ_y_dim;
//  
//    float sum = 0.0f;
//  	if (row < dA_x_dim) {
//            for(int i=0; i< dA_y_dim; i=i+1){
//                float tmp = exp(A[i + dA_y_dim * row]);
//                dA[i + dA_y_dim * col] = tmp;
//                sum += tmp;  
//            }
//            for(int j=0; j< dA_y_dim; j=j+1){
//                for(int i=0; i< dA_y_dim; i=i+1){
//                    if(i==j){
//                        dA[j + dA_y_dim * row] += dZ[row * dA_y_dim + i] * (sum - exp(dZ[row * dA_y_dim + i]))/ (sum * sum) * exp(dZ[j * dA_x_dim + i]);
//                    }
//                    else{
//                        dA[j + dA_y_dim * row] -= dZ[row * dA_x_dim + i] *  exp(dZ[j * dA_x_dim + i])/ (sum * sum) * exp(dZ[j * dA_x_dim + i]);
//                    }
//                }
//            }
//	}
//}

SoftMax::SoftMax(std::string name)
{
    this->name = name;
}

SoftMax::~SoftMax()
{ }

Matrix& SoftMax::forward(Matrix& A,bool  training, bool freeMatrix){
    //std::cout << "SoftMax forward\n";
    this->A = A;
    //printf("soft in A.shape %d A,shape %d\n",A.shape.x, A.shape.y);
    Shape Z_shape(A.shape.x,A.shape.y);
    Z.allocateCuda(Z_shape);
   // std::cout<<"softmax forward\n";
    LayerOutput(A);
 //   printf("soft out Z.shape %d Z,shape %d\n",Z.shape.x, Z.shape.y);
    NNException::throwIfDeviceErrorOccurred("Cannot perform Linear Layer forward propagation");
    A.freeMem();
    return Z;
}
void SoftMax::LayerOutput(Matrix& A) {
    int block_size(256);
    int num_of_blocks((Z.shape.x + block_size - 1) / block_size);
    SoftMaxForward<<<num_of_blocks, block_size>>>( A.data_device,Z.data_device,A.shape.x, A.shape.y);
}

Matrix& SoftMax::backprop(Matrix& dZ, float learning_rate) {
    //std::cout << "SoftMax backward\n";
    dA.allocateCuda(A.shape);
    //printf("soft out back dA.shape %d dz,shape %d\n",dA.shape.x, dA.shape.y);
    //std::cout<<"softmax backward\n";
    //std::cout<<"softmax backward\n";
    //std::cout << " softmax backward shape.x:" << dZ.shape.x << "\n";
    //std::cout << " softmax backward shape.y:" << dZ.shape.y << "\n";
    BackpropError(dZ);
    NNException::throwIfDeviceErrorOccurred("Cannot perform back propagation.");
    dZ.freeMem();
    return dA;
}

void SoftMax::BackpropError(Matrix& dZ) {
    int block_size(256);
    int num_of_blocks ((dZ.shape.x + block_size - 1) / block_size);
    SoftMaxBackprop<<<num_of_blocks, block_size >>> ( dZ.data_device,dA.data_device,Z.data_device,dZ.shape.x, dZ.shape.y);
}

void SoftMax::setData(int* row , int* col) {
}
