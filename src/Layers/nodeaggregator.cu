#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>
#include "nodeaggregator.hh"
#include "csr_graph.cu"
#include "csr_graph.h"

Matrix& NodeAggregator::forward(Matrix& A,bool training,bool freeMatrix){
//std::cout<<"Nodeagg forward\n";
//std::cout << "A:" << A.data_device << "\n";
//this->A = A;
//std::cout << "A:" << A.data_device << "\n";
//std::cout << "this.A" << this->A.data_device << "\n";
Z.allocateCuda(A.shape);
//Z = A;
SpMM(nnz_data, row, col, A.data_device, Z.data_device, A.shape.y, nodes, nnz);
//    std::cout << " NodeAgg forward shape.x:" << Z.shape.x << "\n";
//    std::cout << " NodeAgg forward shape.y:" << Z.shape.y << "\n";
//printf("nodeagg Z.x %d Z.y %d\n",Z.shape.x, Z.shape.y);
if(freeMatrix)
    A.freeMem();
//std::cout<<"Nodeagg ptr:" << Z.data_device << "\n";
return Z;
}

Matrix& NodeAggregator::backprop(Matrix& dZ, float learning_rate) {
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
dZ.freeMem();
return dA;
}

void NodeAggregator::setData(int* row_data, int* col_data) {
    this->row = row_data;
    this->col = col_data;
}

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
