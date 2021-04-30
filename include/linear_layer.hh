#pragma once
#include "nn_layers.hh"

class LinearLayer: public NNLayer{
private:
    const float weights_init_threshold = 0.01;
    
    Matrix W;
    Matrix b;
    
    Matrix Z;
    Matrix A;
    Matrix dA;
    
    void initializeBiasWithZeros();
    void initializeWeightsRandomly();
    
    void computeAndStoreBackpropError(Matrix& dZ);
    void computeAndStoreLayerOutput(Matrix& A);
    void updateWeights(Matrix& dz, float learining_rate);
    void updateBias(Matrix& dZ, float learning_rate);
    
public:
    LinearLayer(std::string name, Shape W_shape);
    ~LinearLayer();
    
    Matrix& forward(Matrix& A, bool training, bool freeMatrix);
    Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);
    
    int getXdim() const;
    int getYdim() const;
    
    Matrix getWeightsMatrix() const;
    Matrix getBiasVector() const;
    
};
