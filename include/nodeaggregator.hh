#pragma once
#include "nn_layers.hh"

class NodeAggregator: public NNLayer{
private:
    Matrix Z;
    Matrix A;
    Matrix dZ;
    Matrix dA;
    float* nnz_data;
    int* row;
    int* col;
    int nodes;
    int nnz;
    
public:
    NodeAggregator(std::string name, float* nnz_data, int* row, int*col, int nodes, int nnz);
    ~NodeAggregator();
    
    Matrix& forward(Matrix& A, bool training, bool freeMatrix);
    Matrix& backprop(Matrix& dZ, float learning_rate, bool freeMatrix);
   
    void node_SpMM(float* nnz_data, int* row, int* col, float* d_B, float* d_C, int FV_size, int m, int nnz); 
//    int getXdim() const;
//    int getYdim() const;
    
};
