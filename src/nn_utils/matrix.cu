#include "matrix.hh"
#include "nn_exception.hh"

Matrix::Matrix(size_t x_dim, size_t y_dim):
    shape(x_dim, y_dim), data_device(nullptr), data_host(nullptr),
    device_allocated(false), host_allocated(false)
{ }

Matrix::Matrix(Shape shape):
    Matrix(shape.x,shape.y)
{ }

void Matrix::allocateCuda(Shape shape){
        data_device = nullptr;
        this->shape = shape;
        cudaMalloc(&data_device,shape.x * shape.y * sizeof(float));
        NNException::throwIfDeviceErrorOccurred("Cannot allocate CUDA memory for tensor.");
        //data_device = std::shared_ptr<float>(device_memory,[&](float* ptr){cudaFree(ptr);});
        device_allocated = true;
       // std::cout << "Memory allocated ptr:" << data_device << "\n";
}
void Matrix::allocateCudaMemory(){
    if (!device_allocated){
        data_device = nullptr;
        cudaMalloc(&data_device,shape.x * shape.y * sizeof(float));
        NNException::throwIfDeviceErrorOccurred("Cannot allocate CUDA memory for tensor.");
        //data_device = std::shared_ptr<float>(device_memory,[&](float* ptr){cudaFree(ptr);});
        device_allocated = true;
       // std::cout << "Memory allocated ptr:" << data_device << "\n";
    }
}

void Matrix::allocateHostMemory(){
    if(!host_allocated){
        //data_host = std::shared_ptr<float>(new float[shape.x * shape.y], [&](float * ptr){delete[] ptr;});
        data_host = (float *) malloc(shape.x * shape.y * sizeof(float));
        host_allocated = true;
    }
}

void Matrix::allocateMemory(){
    allocateCudaMemory();
    allocateHostMemory();
}

void Matrix::allocateMemoryIfNotAllocated(Shape shape){
    //std::cout << "device_allocated::" << device_allocated << "\n";
    if(!device_allocated){
        this->shape = shape;
     //   std::cout << "allocating memory for new Matrix\n";
        allocateMemory();
    }
}

void Matrix::copyHostToDevice(){
    if(device_allocated && host_allocated){
      //  std::cout << "copying data from host to device \n";
      //  std::cout << "shape.x :" << shape.x << "\n";
      //  std::cout << "shape.y :" << shape.y << "\n";
        //size_t *free;
        //size_t *total;
        cudaMemcpy(data_device, data_host, shape.x * shape.y * sizeof(float),cudaMemcpyHostToDevice);
        NNException::throwIfDeviceErrorOccurred("Cannot copyhost data to CUDA device");
    }
    else
    {
     //   std::cout << "device_allocated:" << device_allocated << "\n";
     //   std::cout << "host_allocated:" << host_allocated << "\n";
     //   std::cout << "Host to device allocation failed!\n";
        throw NNException("Cannot copy host data to not allocated memory on device");
    }
}

void Matrix::copyDeviceToHost(){
    if(device_allocated && host_allocated){
        cudaMemcpy(data_host, data_device, shape.x * shape.y * sizeof(float),cudaMemcpyDeviceToHost);
        NNException::throwIfDeviceErrorOccurred("Cannot copyhost data to CUDA device");
    }
    else
    {
 //       std::cout << "Device to host allocation failed!\n";
        throw NNException("Cannot copy device data to not allocated memory on host");
    }
}
void Matrix::freeMem(){
   if(device_allocated ){
       cudaError_t errorCode = cudaFree(data_device);
     //  std::cout << "Free Error:" << errorCode << "\n";
     //  std::cout << "Free pointer:" << data_device << "\n";
       NNException::throwIfDeviceErrorOccurred("Can not delete CUDA memory");
       device_allocated = false;
     }
   if(host_allocated){
       free(data_host);
       host_allocated = false;
     //  std::cout << "Free memory device_allocation:" << device_allocated << "\n";
   }
}

float& Matrix::operator[](const int index){
    return data_host[index];
}

const float& Matrix::operator[](const int index) const {
    return data_host[index];
}

