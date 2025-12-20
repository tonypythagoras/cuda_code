#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>

unsigned const int N = 64;

unsigned const int width = 4;

// kernel code that computes three dimensional stencil
__global__ void stancilKernel(float *input, float *output){
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= 1 && (i <= (width-1)) && j >= 1 && (j <= (width-1)) && k >= 1 && (k <= (width-1))){
        output[i * width * width + j * width + k] = input[i * width * width + j * width + k] +
         input[(i-1) * width * width + j * width + k] + 
         input[(i+1) * width * width + j * width + k] + 
         input[i * width * width + (j-1) * width + k] + 
         input[i * width * width + (j+1) * width + k] + 
         input[i * width * width + j * width + k + 1] + 
         input[i * width * width + j * width + k-1];
                                  
    }


}

// initilizes the array
void initArray(float *input){
   for( int i=0; i< N; i++){
     input[i]=1;
   }

}


int main(){
     size_t size = N * sizeof(float);
     // allocate memory on the  host
     float *input =(float*)malloc(size);
     float *output = (float*)malloc(size);

     initArray(input);

     float *input_d;
     float *output_d;

     cudaError_t err;

// allocates memories on the device
      err = cudaMalloc((void**)&input_d,size);
      
       if( err != cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

      }
      
      err = cudaMalloc((void**)&output_d,size);
      if( err != cudaSuccess){
     fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

     }
     // copy memory from host to device
     cudaMemcpy(input_d,input,size,cudaMemcpyHostToDevice);

// iniliazes thread and thread blocks
     dim3 threadPerBlock(4,4,4);
     dim3 numberOfBlocks(2,2,2);

// calls the kernel code
     stancilKernel <<<numberOfBlocks, threadPerBlock >>> (input_d,output_d);

// copy memory from the device to the host
     cudaMemcpy(output,output_d,size,cudaMemcpyDeviceToHost);

     // output display
 for (int i=0;i<N;i++){
    std::cout<< output[i]<< std::endl;
   }

// free allocated memory on the device
     cudaFree(input_d);
     cudaFree(output_d);
// free allocated memory on the host
     delete[] output;
     delete[] input;

    return 0;
}