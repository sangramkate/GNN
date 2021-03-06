#pragma once
#include <iostream>
#include "matrix.hh"

class NNLayer{
protected:
    std::string name;
public:
    virtual ~NNLayer() = 0;
    virtual Matrix& forward (Matrix& A, bool training, bool freeMatrix) = 0;
    virtual Matrix& backprop (Matrix& dZ, float learning_rate, bool freeMatrix ) = 0;
    std::string getName() {return this->name;};
    virtual void free_matrix()=0;
};

inline NNLayer::~NNLayer() {};
