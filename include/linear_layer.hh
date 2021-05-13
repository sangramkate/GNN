#pragma once
#include "nn_layers.hh"

class LinearLayer: public NNLayer{
private:
    const float weights_init_threshold = 0.1;
    
    Matrix W;
    Matrix b;
    
    Matrix Z;
    Matrix A;
    Matrix dA;
    Matrix dW;
    Matrix stored_Z;
    int layer_num; 
    void initializeBiasWithZeros();
    void initializeWeightsRandomly();
   
    void runGEMM(Matrix& A, Matrix& B, Matrix& C, bool transposeA, bool transposeB);  
    void computeAndStoreBackpropError(Matrix& dZ);
    void computeAndStoreLayerOutput(Matrix& A);
    void updateWeights(Matrix& dz, float learining_rate);
    void updateBias(Matrix& dZ, float learning_rate);
public:
    LinearLayer(std::string name, int layer_num, Shape W_shape);
    ~LinearLayer();
    
    Matrix& forward(Matrix& A, bool training, bool freeMatrix);
    Matrix& backprop(Matrix& dZ, float learning_rate, bool freeMatrix);
    void free_matrix(); 
    
    int getXdim() const;
    int getYdim() const;
    
    Matrix getWeightsMatrix() const;
    Matrix getBiasVector() const;
    
};
