#include <iostream>
#include <time.h>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <sstream>

#include "NeuralNetwork.hh"
#include "linear_layer.hh"
#include "activation.hh"
#include "softmax.hh"
#include "nodeaggregator.hh"
#include "nn_exception.hh"
#include "costfunction.hh"
#include "csr_graph.h"
#include "data.hh" 



float computeAccuracy(const Matrix& predictions, const Matrix& targets, int *node_array, int num_test_nodes);

int main(int argc, char **argv) {


//	std::string dataset_name = "Pubmed-Diabetes";
	std::string dataset_name = argv[1];

	std::fstream data_info;
	data_info.open("datasets/" + dataset_name + "/data_info.csv", std::ios::in);	
        std::string line, word, temp;

        int* dataset_meta = (int *) malloc(4*sizeof(int));


	int nnodes, nnz, nfeatures, nlabels;

        int i = 0;
        while(std::getline(data_info,line)) {
            std::stringstream s(line);
            while(std::getline(s,word,',')) {
                dataset_meta[i] = stoi(word);
                i++;
            }

        }

        std::cout << i << "\n";

	nnodes = dataset_meta[0];
	nnz = dataset_meta[1];
	nfeatures = dataset_meta[2];
	nlabels = dataset_meta[3];

	std::cout << "nnodes: " << nnodes << ", nedges: " << nnz << std::endl;
	std::cout << "nlabels: " << nlabels << ", nfeatures: " << nfeatures << std::endl;


        cudaSetDevice(0);
        //std::fstream myfile("/net/ohm/export/iss/inputs/Learning/cora-labels.txt", std::ios_base::in);
        std::fstream label_info;
        label_info.open("datasets/" + dataset_name + "/label_val.csv", std::ios::in);
        int* label = (int *) malloc(nnodes*nlabels*sizeof(int));
        i = 0;
        while(std::getline(label_info,line)) {
            std::stringstream s(line);
            while(std::getline(s,word,',')) {
                label[i] = stoi(word);
                i++;
            }

        }
        std::cout << i-1 << "\n";
	srand( time(NULL) );

	//CoordinatesDataset dataset(100, 21);
	CostFunction bce_cost;

//Code for extracting data from dataset files starts here
        int* d_row_start;
        int* d_edge_dst;
        float* d_edge_data;
        cudaError_t alloc;
        printf("2*nnz+nnodes = %d\n",2*nnz+nnodes);

        int* h_row_start = (int*)malloc((nnodes+1) * sizeof(int));
        int* h_edge_dst = (int*)malloc((2*nnz+nnodes) * sizeof(int));
        alloc = cudaMalloc(&d_row_start,(nnodes+1) * sizeof(int));
        if(alloc != cudaSuccess) {
            printf("malloc for row info failed\n");
        }
        alloc = cudaMalloc(&d_edge_dst,(2*nnz+nnodes) * sizeof(int));
        if(alloc != cudaSuccess) {
            printf("malloc for col info failed\n");
        }
        alloc = cudaMalloc(&d_edge_data,(2*nnz+nnodes) * sizeof(float));
        if(alloc != cudaSuccess) {
            printf("malloc failed \n");
        }
        float* d_B;

        float* h_B = (float *)malloc((nnodes) * nfeatures * sizeof(float));
	if(h_B == NULL)
	    printf("h_B malloc failed\n");
        alloc = cudaMalloc(&d_B, (nnodes) * nfeatures * sizeof(float));
        if(alloc != cudaSuccess) {
            printf("cudaMalloc failed for features matrix\n");
        }

	float* h_edge_data = (float *)malloc((2*nnz+nnodes) * sizeof(float));

//Filling up the sparse matrix info
        //graph.readFromGR(gr_file , binFile , d_row_start, d_edge_dst , d_B, nfeatures);
        std::fstream feature_info;
        feature_info.open("datasets/" + dataset_name + "/feature_val.csv", std::ios::in);
        i = 0;
        while(std::getline(feature_info,line)) {
            std::stringstream s(line);
            while(std::getline(s,word,',')) {
                h_B[i] = stof(word);
                i++;
            }

        }

        printf("node * feature = %d\n",i);
        std::fstream row_start_info;
        row_start_info.open("datasets/" + dataset_name + "/row_start.csv", std::ios::in);
        i = 0;
        while(std::getline(row_start_info,line)) {
            std::stringstream s(line);
            while(std::getline(s,word,',')) {
                h_row_start[i] = stoi(word);
                i++;
            }

        }
        printf("node + 1  = %d\n",i);

        std::fstream edge_dst_info;
        edge_dst_info.open("datasets/" + dataset_name + "/edge_dst.csv", std::ios::in);
        i = 0;
        while(std::getline(edge_dst_info,line)) {
            std::stringstream s(line);
            while(std::getline(s,word,',')) {
                h_edge_dst[i] = stoi(word);
                //printf("h_edge_dst[%d] = %d\n",i,h_edge_dst[i]);
                i++;
            }

        } 
        printf("egdges = %d\n",i);

        std::fstream edge_data_info;
        edge_data_info.open("datasets/" + dataset_name + "/edge_data.csv", std::ios::in);
        i = 0;
        while(std::getline(edge_data_info,line)) {
            std::stringstream s(line);
            while(std::getline(s,word,',')) {
                h_edge_data[i] = stof(word);
                //printf("h_edge_data[%d] = %f, h_edge_dst[%d] = %d\n",i,h_edge_data[i],i,h_edge_dst[i]);
                i++;
            }

        } 
	printf("edge data = %d\n", i);

        alloc = cudaMemcpy(d_B, h_B, (nnodes * nfeatures *sizeof(float)), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_start, h_row_start,(nnodes+1) * sizeof(int) , cudaMemcpyHostToDevice);
        cudaMemcpy(d_edge_dst, h_edge_dst, (2*nnz+nnodes) * sizeof(int) , cudaMemcpyHostToDevice);
	alloc = cudaMemcpy(d_edge_data, h_edge_data, ((2*nnz+nnodes) *sizeof(float)), cudaMemcpyHostToDevice);
        if(alloc != cudaSuccess) {
        printf("Feature matrix memcpy failed\n");
        }
	
	int hidden_size = 32;	

	if(alloc != cudaSuccess) {
    	printf("Feature matrix memcpy failed\n");
	} 
	std::cout << "Dataset captured!\n";
	
	int tmp_val = 1000;
        int num_train_nodes = 0.6 * (nnodes);
        int num_test_nodes = nnodes - num_train_nodes;

        Data dataset(nnodes,num_train_nodes,nfeatures,nlabels,label,h_B);
        printf("num_train_nodes = %d\n", num_train_nodes);
        printf("features shape_x =%d label shape_x=%d\n",dataset.train_input_features.shape.x, dataset.train_input_labels.shape.x);
        free(label);
        free(h_B);
	std::cout << "Dataset captured!\n";
        NeuralNetwork nn(0.0015);
        //-----------------------------------------------
        std::cout << "Instance of Neural Network\n";
	nn.addLayer(new NodeAggregator("nodeagg1", d_edge_data, d_row_start, d_edge_dst, nnodes, 2*nnz+nnodes));
        std::cout << "Added Nodeaggregator 1 layer\n";
	nn.addLayer(new LinearLayer("linear1", Shape(nfeatures, hidden_size)));
        std::cout << "Added Linear layer 1\n";
	nn.addLayer(new ReLUActivation("relu1"));
        std::cout << "Added relu layer 1\n";
        //-----------------------------------------------
       // nn.addLayer(new NodeAggregator("nodeagg2", d_edge_data, d_row_start, d_edge_dst, 2708, nnz));
       // std::cout << "Added Nodeaggregator layer 2\n";
       // nn.addLayer(new LinearLayer("linear2", Shape(nlabels,nlabels)));
       // std::cout << "Added Linear layer 2\n";
       // nn.addLayer(new ReLUActivation("relu2"));
       // std::cout << "Added Relu layer 2\n"; 
        //-----------------------------------------------
        nn.addLayer(new NodeAggregator("nodeagg3", d_edge_data, d_row_start, d_edge_dst, nnodes, 2*nnz+nnodes));
        std::cout << "Added Nodeaggregator layer 3\n";
	nn.addLayer(new LinearLayer("linear3", Shape(hidden_size,nlabels)));
        std::cout << "Added Linear layer 3\n";
//	nn.addLayer(new ReLUActivation("relu3"));
//        std::cout << "Added Relu layer 3\n"; 
        //-----------------------------------------------
        nn.addLayer(new SoftMax("softmax"));
        std::cout << "Added softmax layer \n";

        std::cout << "Instance of Neural Network complete\n";
	// network training
	Matrix Y;
	Matrix Y_test;
	for (int epoch = 0; epoch < 200; epoch++) {
		float cost = 0.0;

		Y = nn.forward(dataset.input_features, true);
		nn.backprop(Y,dataset.input_labels,dataset.node_array_device,num_test_nodes);

		cost += bce_cost.cost(Y,dataset.input_labels,dataset.node_array_device, num_test_nodes);
		if (epoch % 10 == 0) {
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost
						<< std::endl;
		}
                Y.freeMem();

		if(epoch %10 == 0)  {
			float accuracy = 0.0f;
			Y_test = nn.forward(dataset.input_features, false);
			Y_test.allocateHostMemory();
			//std::cout << "Y_test.host allocated:" << Y_test.host_allocated << "\n";
			Y_test.copyDeviceToHost();
			//std::cout << "Y_test copied to host "<< "\n";
			accuracy = accuracy + computeAccuracy(Y_test,dataset.input_labels, dataset.node_array, num_test_nodes);
			Y_test.freeMem();
			std::cout << "Accuracy: " << accuracy << std::endl;
		}
	}

        float accuracy = 0.0f;
        float final_accuracy = 0.0f;
//	for (int batch = 0; batch < dataset.getNumOfTestBatches(); batch++) {
		Y = nn.forward(dataset.input_features, false);
                Y.allocateHostMemory();
                std::cout << "Y.host allocated:" << Y.host_allocated << "\n";
		Y.copyDeviceToHost();
                std::cout << "Y copied to host "<< "\n";
                accuracy = accuracy + computeAccuracy(Y,dataset.input_labels, dataset.node_array, num_test_nodes);
//	}
        final_accuracy = accuracy;
	// compute accuracy

	for(int i=0; i<Y.shape.x*Y.shape.y; i++) {
		//printf("Final Y[%d] = %f\n", i, Y[i]);
	}
        
	std::cout << "Accuracy: " << final_accuracy << std::endl;
        cudaFree(d_row_start);
        cudaFree(d_edge_dst);
        cudaFree(d_B);
        cudaFree(d_edge_data);
        dataset.input_features.freeMem();
        dataset.input_labels.freeMem();
	return 0;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets, int *node_array, int num_test_nodes) {
	int correct_predictions = 0;

	for (int i = 0; i < num_test_nodes; i++) { 
        int max_class = 0;
        float max_prediction  = predictions[node_array[i] * predictions.shape.y + 0];
        for (int j = 0; j < predictions.shape.y; j++) {
            if (predictions[node_array[i] * predictions.shape.y + j] > max_prediction) {
                max_class = j;
                max_prediction = predictions[node_array[i] * predictions.shape.y + j];
            }
        }
	//printf("max class = %d, max pred = %f\n", max_class, max_prediction);

        if (targets[node_array[i] * predictions.shape.y + max_class] == 1) {
            correct_predictions++;
        }
	}
	return static_cast<float>(correct_predictions) / (num_test_nodes);
}
