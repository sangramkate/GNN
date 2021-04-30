#pragma once
#include <exception>
#include <iostream>

class NNException : std::exception{
private:
    const char* exception_msg;
    
public:
    NNException(const char* exception_msg):
        exception_msg(exception_msg)
    { }
    
    virtual const char* what() const throw()
    {
        return exception_msg;
    }
       
    static void throwIfDeviceErrorOccurred( const char* exception_msg){
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess){
            std::cout << "-------------------------------------------\n";
            std::cerr << error << ":" << exception_msg << "\n";
            std::cout << "-------------------------------------------\n";
            throw NNException(exception_msg);
        }
    }
};
