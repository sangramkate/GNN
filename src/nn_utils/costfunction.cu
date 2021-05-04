#include "costfunction.hh"
#include "nn_exception.hh"

#include <math.h>
#include <iostream>
#include <assert.h>

__global__ void binaryCrossEntropyCost(float* predictions, float* target, int size,int prediction_y, float* cost, int* node_array_device, int num_test_nodes) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
        float partial_cost = 0.0f;
	if (index >= num_test_nodes && index < size) {
            for(int i = 0 ; i < prediction_y; i++){
                int index_train = node_array_device[index];
                partial_cost += (target[index_train* prediction_y + i] * logf(predictions[index_train * prediction_y + i]) 
                                                              + (1.0f - target[index_train * prediction_y + i]) 
                                                              * logf(1.0f - predictions[index_train * prediction_y + i]));
                //partial_cost += target[index* prediction_y + i] * logf(predictions[index * prediction_y + i]); 
                  //partial_cost += 0.5 * (target[index_train* prediction_y + i] - predictions[index_train * prediction_y + i])* (target[index_train* prediction_y + i] - predictions[index_train * prediction_y + i]); 
                  //printf("Partial_cost=%f,log=%f\n",partial_cost,logf(predictions[index_train * prediction_y + i]));
                }
        //printf("Partial_cost=%f,log=%f\n",partial_cost,logf(predictions[index_train * prediction_y + i]);
		  //  dY[index*prediction_y + i] = target[index * prediction_y + i];
        if (partial_cost != logf(0)) {
		    atomicAdd(cost, partial_cost);
        }
	}
}

__global__ void dBinaryCrossEntropyCost(float* predictions, float* target, float* dY, int size,int prediction_y, int* node_array_device, int num_test_nodes) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_test_nodes) {
        for (int i = 0; i < prediction_y; i++) {
            dY[node_array_device[index]*prediction_y + i] = 0;
        }
    } else if (index < size) {
            if(index >= num_test_nodes && index < num_test_nodes + 5) {
                for(int i = 0 ; i < prediction_y; i++){ 
                    int index_train = node_array_device[index];
                    dY[index_train*prediction_y + i] = (-target[index_train*prediction_y + i] + predictions[index_train * prediction_y + i] ) / 
                                                        ((1 - predictions[index_train * prediction_y + i]) * predictions[index_train * prediction_y + i]);
		            //dY[node_array_device[index]*prediction_y + i] = target[node_array_device[index] * prediction_y + i]-predictions[node_array_device[index] * prediction_y + i];
                    printf("%f:%f, ",target[node_array_device[index] * prediction_y + i],predictions[node_array_device[index] * prediction_y + i]);
                }
            printf("\n");
           }
	}
}

float CostFunction::cost(Matrix& predictions, Matrix& target, int *node_array_device, int num_test_nodes) {
       // std::cout << "predictions.x:" << predictions.shape.x <<"\n" ;
       // std::cout << "predictions.y:" << predictions.shape.y <<"\n" ;
        //std::cout << "target.x:" << target.shape.x <<"\n" ;
        //std::cout << "target.y:" << target.shape.y <<"\n" ;
	assert(predictions.shape.y == target.shape.y);

	NNException::throwIfDeviceErrorOccurred("Error already happened.");
	float* cost = nullptr;
        cudaMalloc(&cost,sizeof(float));
	NNException::throwIfDeviceErrorOccurred("Could not allocate memory.");
        cudaMemset(cost, 0.0f, sizeof(float));
	NNException::throwIfDeviceErrorOccurred("Cannot set the data.");
       // std:: cout << "pointer created\n";
       //cudaMallocManaged(&cost, sizeof(float));
       // std::cout <<"this gets printed\n";
       //   std:: cout << "Memory Allocated\n";
       //*cost = 0.0f;
       // std:: cout << "cost initialized\n";

	dim3 block_size(256);
      // std:: cout << "dim3 block size\nn";
	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
        //std::cout << "start finding cross entropy\n";
	binaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device, target.data_device,predictions.shape.x,predictions.shape.y, cost, node_array_device, num_test_nodes);
      //  std::cout << "done finding cross entropy\n";
	cudaDeviceSynchronize();
	NNException::throwIfDeviceErrorOccurred("Cannot compute binary cross entropy cost.");
        
        float* cost_value = (float*) malloc(sizeof(float));
        cudaMemcpy(cost_value,cost,sizeof(float),cudaMemcpyDeviceToHost);
        *cost_value = *cost_value * -1 / (predictions.shape.x - num_test_nodes);
	//float cost_value = *cost;
	cudaFree(cost);

	return (*cost_value);
}

Matrix& CostFunction::dCost(Matrix& predictions, Matrix& target, Matrix& dY, int *node_array_device, int num_test_nodes) {
	assert(predictions.shape.y == target.shape.y);

	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
	dBinaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device, target.data_device,dY.data_device,predictions.shape.x,predictions.shape.y,node_array_device,num_test_nodes);
	NNException::throwIfDeviceErrorOccurred("Cannot compute derivative for binary cross entropy.");

	return dY;
}
