#pragma once
#include <iostream>
#include "matrix.hh"

class NNLayer{
protected:
    std::string name;
    
public:
    virtual ~NNLayer() = 0;
    virtual Matrix& forward (Matrix& A, bool training, bool freeMatrix) = 0;
    virtual Matrix& backprop (Matrix& dZ, float learning_rate) = 0;
    virtual void setData(int* row, int* col)= 0 ; 
    //void NodeAggSetData(float* row, float* col); 
    std::string getName() {return this->name;};
};

inline NNLayer::~NNLayer() {};
