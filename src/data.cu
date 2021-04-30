#include "data.hh"
#include <stdlib.h>
#include <iostream>
#include "shape.hh"

Data::Data(int num_nodes, size_t batch_size,int feature_size, int label_size, int* label, float* feature) :
	num_nodes(num_nodes),batch_size(batch_size),feature_size(feature_size),label_size(label_size),feature(feature), label(label)
{
        number_of_batches = (int) (num_nodes/batch_size);
        num_training_batches = (int) (number_of_batches * 0.4);
        num_test_batches = number_of_batches- num_training_batches;
        std::cout << "number of nodes :" << num_nodes << "\n" ;
        std::cout << "number of batches :" << number_of_batches << "\n" ;
        std::cout << "number of training batches :" << num_training_batches << "\n" ;
        std::cout << "number of test batches :" << num_test_batches << "\n";
        std::cout << "batch_size :" << batch_size << "\n" ;
        std::cout << "feature_size :" << feature_size << "\n" ;
        std::cout << "label_size :" << label_size << "\n" ;
        
        Shape input_shape(num_nodes,feature_size);
        input_features.allocateMemoryIfNotAllocated(input_shape);
        std::cout << "input_features.x:" << input_features.shape.x << "\n";
        std::cout << "input_features.y:" << input_features.shape.y << "\n";

        Shape input_label_shape(num_nodes,label_size);
        input_labels.allocateMemoryIfNotAllocated(input_label_shape);
        std::cout << "input_labels.x:" << input_labels.shape.x << "\n";
        std::cout << "input_labels.y:" << input_labels.shape.y << "\n";
	
        for (int i = 0; i < (num_nodes-1000); i++) {
        //for (int i = 0; i < num_nodes; i++) {
	    for (int j = 0; j < feature_size; j++) {
                input_features[i * feature_size + j ] = feature [i *(feature_size) + j + 7000];
               // if(feature[i * feature_size + j]){
               //    std::cout << "feature[" << i * (feature_size) + j << "]"<<feature[i * (feature_size) + j] << "\n"; 
               //    std::cout << "input features[" << i * (feature_size) + j << "]" << input_features[i * (feature_size) + j] << "\n"; 
               // }
            }
        }
        input_features.copyHostToDevice();
	for (int i = 0; i < num_nodes; i++) {
	    for (int j = 0; j < (int)(label_size); j++) {
                input_labels[i*label_size + j] = (float) label[i*label_size + j];
            }
        }
        input_labels.copyHostToDevice();
 /*
	for (int i = 0; i < num_training_batches+1; i++) {
// adding the input features into a Matrix form and pushing it.
           // std::cout << "start collecting training data for batch:" << i << "\n";
	    training_batches.push_back(Matrix(Shape(batch_size, feature_size)));
	    training_targets.push_back(Matrix(Shape(batch_size, label_size)));
	    training_batches[i].allocateMemory();
	    training_targets[i].allocateMemory();
           // std::cout << "training batch size:" << sizeof((training_batches[i].data_host)) << "\n";
           // std::cout << "test batch size:" << sizeof((training_targets[i].data_host)) << "\n";
// Set data under this for loop
            for(int j=0; j < batch_size; j++ ){
                for(int k=0; k < feature_size; k++) {
                  //  std::cout << "  training feature i j k :" << i <<" "<< j << " " << k << "\n";
                    training_batches[i][j] = feature[ k + feature_size * j  + batch_size * feature_size * i ];
                }
            }
            for(int j=0; j < batch_size; j++){
                for(int k=0; k < label_size; k++) {
                  //  std::cout << "  training label i j k :" << i <<" "<< j << " " << k << "\n";
                    training_targets[i][j] = (float) label[ k + label_size * j  + batch_size * label_size * i ];
                }
            }  
// sending the data to gpu for computation
          //  std::cout << "transferring batches to device!!\n";
	    training_batches[i].copyHostToDevice();
          //  std::cout << "transferring labels to device!!\n";
	    training_targets[i].copyHostToDevice();
          //  std::cout << "One Batch done!!\n";
	}
	for (int i = 0; i < num_test_batches+1; i++) {
// adding the input features into a Matrix form and pushing it.
	    test_batches.push_back(Matrix(Shape(batch_size, feature_size)));
	    test_targets.push_back(Matrix(Shape(batch_size, label_size)));
	    test_batches[i].allocateMemory();
	    test_targets[i].allocateMemory();
          //  std::cout << "test batch size:" << sizeof(*(test_batches[i].data_host));
          //  std::cout << "test batch size:" << sizeof(*(test_targets[i].data_host));
// Set data under this for loop
            for(int j=0; j < batch_size; j++ ){
                for(int k=0; k < feature_size; k++) {
                    test_batches[i][j] = feature[ k + feature_size * j  + batch_size * feature_size * i ];
                }
            }
            for(int j=0; j < batch_size; j++){
                for(int k=0; k < label_size; k++) {
                    test_targets[i][j] = label[ k + label_size * j  + batch_size * label_size * i ];
                }
            }  
// sending the data to gpu for computation
	    test_batches[i].copyHostToDevice();
	    test_targets[i].copyHostToDevice();
	}
*/
}

int Data::getNumOfTrainingBatches() {
	return num_training_batches;
}

int Data::getNumOfTestBatches() {
	return num_test_batches;
}

//std::vector<Matrix>& Data::getTrainingBatches() {
Matrix& Data::getTrainingBatches() {
//	return training_batches;
	return input_features;
}

Matrix& Data::getTestBatches() {
//	return test_batches;
	return input_features;
}

Matrix& Data::getTrainingTargets() {
//	return training_targets;
        return input_labels;
}

Matrix& Data::getTestTargets() {
//	return test_targets;
	return input_labels;
}
