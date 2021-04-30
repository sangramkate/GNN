#pragma once
#include "nn_layers.hh"

class SoftMax: public NNLayer{
private:
    const float weights_init_threshold = 0.01;
    
    Matrix Z;
    Matrix A;
    Matrix dA;
    Matrix dZ;
    
    void BackpropError(Matrix& dZ);
    void LayerOutput(Matrix& A);
    
public:
    SoftMax(std::string name);
    ~SoftMax();
    
    Matrix& forward(Matrix& A, bool training, bool freeMatrix);
    Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);
};
