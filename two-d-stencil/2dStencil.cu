#include "cuda_runtime.h"
#include "device_launch_parameter.h"
#include "iostream"
#include <stdio.h>
#include <stdlib.h>


unsigned const int N = 16;
unsigned const int width = 4;


// computes 2 dimentional stencil
__global__ void stencil(float *input, float *output){
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;

     if((row >= 1) && (row < (width -1)) && (col >= 1) && (col < width -1)){
        output[row * width + col] = input[(row-1) * width + col] + input[(row + 1) * width + col] + input[(row) * width + col + 1]+ input[(row-1) * width + col - 1];
     }

}


// initializes array
void initArray(float *input){
      
      for(int i=0;i<N;i++){
        input[i]=1;

      }
        

}

int main(){
  size_t size = N * sizeof(float);
  
  // allocates memory on the host
  float *input=(float*)malloc(size);
  float *output=(float*)malloc(size);

  initArray(input);

  float *input_d;
  float *output_d;

  cudaError_t err;

// allocates memory on the device
  err=cudaMalloc((void**)&input_d,size);

  if( err != cudaSuccess){
     fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

}

err=cudaMalloc((void**)&,output_d,size);

if( err != cudaSuccess){
     fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

}

// copy memory from host to device
cudaMemcpy(input_d,input,size,cudaMemcpyHostToDevice);

// initializes threads and thread blocks
dim3 threadPerBlock(2,2);
dim3 numberOfBlocks(2,2);

// calls the kernel code
stencil <<<numberOfBlocks, threadPerBlock >>>(input_d, output_d);

// copy memory from the device to the host
cudaMemcpy(output,output_d,size,cudaMemcpyDeviceToHost);


// output display
 for (int i=0;i<N;i++){
    std::cout<< output[i]<< std::endl;
   }


// free allocated memories on the device 
cudaFree(input_d);
cudaFree(output_d);

// free allocated memories on the host
delete[] input;
delete[] output;


    return 0;
}