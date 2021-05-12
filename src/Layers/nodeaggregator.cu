#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>
#include "nodeaggregator.hh"
#include "csr_graph.cu"
#include "csr_graph.h"
#include "nn_exception.hh"

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

Matrix& NodeAggregator::forward(Matrix& A,bool training,bool freeMatrix){
	//std::cout<<"Nodeagg forward\n";
	//std::cout << "A:" << A.data_device << "\n";
	//this->A = A;
	//std::cout << "A:" << A.data_device << "\n";
	//std::cout << "this.A" << this->A.data_device << "\n";
	Z.allocateCuda(A.shape);
	//Z = A;
	//print_kernel_agg<<<1,1>>>(A.data_device,50, "agg - in - agg layer");

	SpMM(nnz_data, row, col, A.data_device, Z.data_device, A.shape.y, nodes, nnz);

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
	dA.allocateCuda(dZ.shape);
	//dA = dZ;
	//std::cout<<"Nodeagg backward\n";
	//std::cout<<"dZ.Shape.x:" << dZ.shape.x << "\n";
	//std::cout<<"dZ.Shape.x:" << dZ.shape.y << "\n";
	SpMM(nnz_data, row, col, dZ.data_device, dA.data_device, dZ.shape.y, nodes, nnz);
	//    std::cout << " NodeAgg backward shape.x:" << dA.shape.x << "\n";
	 //   std::cout << " NodeAgg backward shape.y:" << dA.shape.y << "\n";
        NNException::throwIfDeviceErrorOccurred("Error found in NN Agg backward 1");
	dZ.freeMem();
        NNException::throwIfDeviceErrorOccurred("Error found in NN Agg backward 2");
	return dA;
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
