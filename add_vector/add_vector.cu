#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iostream>

__global__ void addVector(float *x, float *y, float *z, int VEC_SIZE){
     int globalId=blockIdx.x * blockDim.x + threadIdx.x
       
       if(globalId < VEC_SIZE){
        z[globalId]=x[globalId]+y[globalId];
       }

   }



   int main(){

    const int VEC_SIZE=2048; // vector size 
    const int size= VEC_SIZE * sizeof(int); // size of the vector in bytes

     // create memories for the vector
    float  *x=new float[VEC_SIZE]; 
    float  *y=new float[VEC_SIZE];
    float *z=new float[VEC_SIZE];

    // initialize the vector with numbers strating from 1
    for (int i=1; i < VEC_SIZE;i++){
        x[i-1]=i;
        y[i-1]=i;

    }

    // declare the device vector variables
    float *x_d;
    float *y_d;
    float *z_d;

   // create the device memory
    cudaMalloc((void**)&x_d,size);
    cudaMalloc((void**)&y_d,size);
    cudaMalloc((void**)&z_d,size);

    // transfer memory by copying the contents of the host vector to the device
    cudaMemcpy(x_d,x,size,cudaMemcpyHostToDevice);
    cudaMemcpy(y_d,y,size,cudaMemcpyHostToDevice);

     // initialize our thread blocks
    int threadPerBlock=512;
    int numberOfBlocks=(VEC_SIZE + threadPerBlock -1 )/(threadPerBlock);
     
     // call the device kernel
     addVector <<< numberOfBlocks, threadPerBlock >>>(x_d,y_d,z_d,VEC_SIZE);

     // transfer memory from the device to host by copying the contents of the device vector to the host
     cudaMemcpy(z,z_d,size,cudaMemcpyDeviceToHost);

    // print out the result of our addition
     for (int i=0;i<VEC_SIZE;i++){
        std::cout << z[i]<< std::endl;
     }

     // free the device memories
     cudaFree(x_d);
     cudaFree(y_d);
     cudaFree(z_d);

    // free the host memories
     delete[] x;
     delete[] y;
     delete[] z;

     return 0;

   }