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
    void setData(int* row_data, int* col_data); 
    Matrix& forward(Matrix& A, bool training, bool freeMatrix);
    Matrix& backprop(Matrix& dZ, float learning_rate);
   
//    int getXdim() const;
//    int getYdim() const;
    
};
