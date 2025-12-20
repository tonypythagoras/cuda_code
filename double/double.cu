#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iostream>

// kernel method that can only be called from another kernel method
__device__ int two(int a){
  return a*2;
}

// kernel method that does the computation
__global__ void times(int *array, int N,int width){
    unsigned int row=blockIdx.y*blockDim.y+threadIdx.y;
    unsigned int col=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int globalId=row * width +col;
    if(globalId < N){
      array[globalId]=two(array[globalId]);
       
    }  

}


int main(){

 // the size of the array
  const int N= 256;
  // the width of the grid
  const int width = 16;
  // the size of the array in bytes
  const int size = N * sizeof(int);

   // the host arrays 
  int *x = new int[N];
  int *y =new int[N];
   // array init
  for(int i=0;i < N; i++ ){
    x[i]=i+1;
  }

// device array
  int *x_d;
  // allocate memory to the device array
  cudaMalloc((void**)&x_d,size);
// copy memory from the host to device
  cudaMemcpy(x_d,x, size,cudaMemcpyHostToDevice);

 // initialize the thread 
 // I am using two dimensional thread grid
  dim3 threadPerBlock(8,8);
  dim3 numberOfBlocks((width + threadPerBlock.x-1/threadPerBlock.x),(width + threadPerBlock.x-1/threadPerBlock.x));

  // call my kernel function
  times <<< numberOfBlocks, threadPerBlock >>>(x_d,N,width);

// copy memory from device to host
  cudaMemcpy(y,x_d,size,cudaMemcpyDeviceToHost);

// prints out my results
  for(int i=0;i<N;i++){
            std::cout << y[i]<< std::endl;
  }
// free device memory
 cudaFree(x_d);

// free host memory
 delete[] x;
 delete[] y;

 return 0;











}



