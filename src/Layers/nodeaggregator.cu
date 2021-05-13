#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>
#include "nodeaggregator.hh"
//#include "csr_graph.cu"
//#include "csr_graph.h"
#include "nn_exception.hh"
#include <cusparse_v2.h>

__global__ void print_kernel_agg(float *A, int size, std::string str) {
	for(int i=1433; i<1433+size; i++) {
		if(A[i] != 0.0) {
		    printf("The value of %s[%d, %d] = %f\n", str, 1, i-1433*1, A[i]);
		}
	}
	for(int i=1433*344; i<344*1433+size; i++) {
		if(A[i] != 0.0) {
		    printf("The value of %s[%d, %d] = %f\n", str, 344, i-1433*344, A[i]);
		}
	}
}
/*

__global__ void print_kernel_agg(float* A, int size, std::string str) {

    printf("The arr is %s\n", str);

    for(int i=1433; i<(1433+size); i=i+1) {
        if(A[i] != 0.0) {
            printf("The val of [1, %d] = %f\n", i-1433, A[i]);
        }
    }
    for(int i=344*1433; i<344*1433+size; i=i+1) {
        if(A[i] != 0.0) {
            printf("The val of [344, %d] = %f\n", i-1433*344, A[i]);
        }
    }
}

*/
__global__
void  agg(float* nnz_data, int* row, int* col, float* d_B, float* d_C, int FV_size, int m, int nnz) {
    float val;
    int start_dst_ind;
    int num_dst;
    int var = 0;
    //for (int index = blockIdx.x * FV_size + threadIdx.x; index < (blockIdx.x + 1) * FV_size; index = index + blockDim.x) {
    //   start_dst_ind = row[blockIdx.x]; 
    //   num_dst = row[(blockIdx.x) + 1];
    //   val = 0;
    //   for (int i = start_dst_ind; i < num_dst; i=i+1) {
    //       val += d_B[col[i]*FV_size +blockDim.x*var + threadIdx.x]*nnz_data[i];
    //        //if(blockIdx.x == 1) {
    //        //    printf("val = %
    //        //}
    //   }
    //   d_C[index] = val;
    //   var++;
    //}
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int row_start_node = index/FV_size;
    start_dst_ind = row[row_start_node];
    num_dst = row[row_start_node + 1];
    for (int i = start_dst_ind; i < num_dst; i= i +1) {
       val += d_B[col[i]*FV_size + (index%FV_size)]*nnz_data[i]; 
    }
    d_C[index] = val;
}

 void NodeAggregator::node_SpMM(float* nnz_data, int* row, int* col, float* d_B, float* d_C, int FV_size, int m, int nnz) {
	int n = FV_size;
	cusparseHandle_t cusparse1 = NULL;
	cusparseMatDescr_t      descrA  ;
	cusparseCreateMatDescr(&descrA);
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

	const float alp = 1;
	const float bet = 0;
	const float* alpha = &alp;
	const float* beta = &bet;

	cusparseStatus_t result;
	cusparseCreate(&cusparse1);
	//A : mxn, B: mxm, C: mxn
	result = cusparseSgemmi(cusparse1, n, m , m, nnz, alpha, d_B, n, nnz_data,row, col, beta, d_C, n);

	if(result != CUSPARSE_STATUS_SUCCESS) {
	    printf("Cusparse failed\n");
	}

	  return;
}

Matrix& NodeAggregator::forward(Matrix& A,bool training,bool freeMatrix){
	//std::cout<<"Nodeagg forward\n";
	//std::cout << "A:" << A.data_device << "\n";
	//this->A = A;
	//std::cout << "A:" << A.data_device << "\n";
	//std::cout << "this.A" << this->A.data_device << "\n";
	Z.allocateCuda(A.shape);
	//Z = A;
    /*
    dim3 block_size(256);
    dim3 num_of_blocks((A.shape.x*A.shape.y + block_size.x - 1) / block_size.x);
    //print_kernel_agg<<<1,1>>>(A.data_device, 20, "A - agg in");
    agg<<<num_of_blocks,block_size>>>(nnz_data, row, col, A.data_device, Z.data_device, A.shape.y, nodes, nnz);

    cudaDeviceSynchronize();

    //print_kernel_agg<<<1,1>>>(Z.data_device, 20, "Z - agg out");
    */
	//print_kernel_agg<<<1,1>>>(A.data_device,50, "agg - in - agg layer");
	node_SpMM(nnz_data, row, col, A.data_device, Z.data_device, A.shape.y, nodes, nnz);

	//print_kernel_agg<<<1,1>>>(Z.data_device, 50, "agg - out - agg layer");

	    //std::cout << " NodeAgg forward shape.x:" << Z.shape.x << "\n";
	//    std::cout << " NodeAgg forward shape.y:" << Z.shape.y << "\n";
        NNException::throwIfDeviceErrorOccurred("Error found in NN AGG forward");
	if(freeMatrix)
	    A.freeMem();
	//std::cout<<"Nodeagg ptr:" << Z.data_device << "\n";
	return Z;
}

Matrix& NodeAggregator::backprop(Matrix& dZ, float learning_rate, bool freeMatrix) {
	this->dZ = dZ;
	//std::cout<<"Nodeagg backward\n";
//	dA.allocateCuda(dZ.shape);
	//dA = dZ;
	//std::cout<<"Nodeagg backward\n";
	//std::cout<<"dZ.Shape.x:" << dZ.shape.x << "\n";
	//std::cout<<"dZ.Shape.x:" << dZ.shape.y << "\n";
    /*dim3 block_size(256);
    dim3 num_of_blocks((dZ.shape.x*dZ.shape.y + block_size.x - 1) / block_size.x);
    agg<<<num_of_blocks,block_size>>>(nnz_data, row, col, dZ.data_device, dA.data_device, dZ.shape.y, nodes, nnz);
    */
//	SpMM(nnz_data, row, col, dZ.data_device, dA.data_device, dZ.shape.y, nodes, nnz);
	//    std::cout << " NodeAgg backward shape.x:" << dA.shape.x << "\n";
	 //   std::cout << " NodeAgg backward shape.y:" << dA.shape.y << "\n";
//        NNException::throwIfDeviceErrorOccurred("Error found in NN Agg backward 1");
//	dZ.freeMem();
//        NNException::throwIfDeviceErrorOccurred("Error found in NN Agg backward 2");
	return dZ;
}

//nn.addLayer(new NodeAggregator("nodeagg1", d_edge_data, d_row_start, d_edge_dst, 2708, nnz));
NodeAggregator::NodeAggregator(std::string name, float* nnz_data, int* row, int*col, int nodes, int nnz)
{
    this->name = name;
    this->nnz_data = nnz_data;
    this->row = row;
    this->col = col;
    this->nodes = nodes;
    this->nnz = nnz;
}

NodeAggregator::~NodeAggregator() { }
