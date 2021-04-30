#pragma once

#include "matrix.hh"
#include <stdlib.h>
#include <iostream>
#include <vector>

class Data {
private:
        int num_nodes;
        int num_training_batches;
        int num_test_batches;	
        size_t batch_size;
	size_t number_of_batches;
        size_t feature_size;
        size_t label_size;

        int* label;
        float* feature;

	std::vector<Matrix> training_batches;
	std::vector<Matrix> test_batches;
	std::vector<Matrix> training_targets;
	std::vector<Matrix> test_targets;

public:

        Matrix input_features;
        Matrix input_labels;
	Data(int num_nodes,size_t batch_size, int feature_size, int label_size, int* label, float* feature);

	int getNumOfTrainingBatches();
	int getNumOfTestBatches();
	Matrix& getTrainingBatches();
	Matrix& getTestBatches();
	Matrix& getTrainingTargets();
	Matrix& getTestTargets();

};
