#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>

// some costatnts
unsigned const int N = 4096;
unsigned const int in_t_width = 8;
unsigned const int out_t_width = 6;
unsigned const int width = 16;


unsigned const int width = 16;


// computes three dimensional stecil using shared memory
__global__ void stancilKernel(float *input, float *output){
    int i = blockIdx.z * out_t_width + threadIdx.z - 1;
    int j = blockIdx.y * out_t_width + threadIdx.y - 1;
    int k = blockIdx.x * out_t_width + threadIdx.x - 1;

    __shared__ float shared_m[in_t_width][in_t_width][in_t_width];

    if(i >= 0 && i < width && j >= 0 && j< width && k >= 0 && k < width){

           shared_m[threadIdx.z][threadIdx.y][threadIdx.x]=input[i * width * width + j * width + k];

    }
           __syncthreads();

    if(i >= 1 && (i <= (width-1)) && j >= 1 && (j <= (width-1)) && k >= 1 && (k <= (width-1))){
        
      if(threadIdx.z >= 1 && threadIdx.z < in_t_width && threadIdx.y >=1 && threadIdx.y < in_t_width && threadIdx.x >= 1 && threadIdx.x < in_t_width){

 output[i * width * width + j * width + k] = shared_m[threadIdx.z][threadIdx.y][threadIdx.x] +
         shared_m[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
         shared_m[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
         shared_m[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
         shared_m[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
         shared_m[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
         shared_m[threadIdx.z + 1][threadIdx.y][threadIdx.x]
      }
    }
}

// initializes array
void initArray(float *input){
   for( int i=0; i< N; i++){
     input[i]=1;
   }

}


int main(){

      size_t size = N * sizeof(float);
      // allocates some memories on the host
     float *input =(float*)malloc(size);
     float *output = (float*)malloc(size);

     initArray(input);

     float *input_d;
     float *output_d;

     cudaError_t err;

// allocates some memories on the device
      err = cudaMalloc((void**)&input_d,size);
      
       if( err != cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

      }
      
      err = cudaMalloc((void**)&output_d,size);
      if( err != cudaSuccess){
     fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

     }

     // copy memory from host to the device
     cudaMemcpy(input_d,input,size,cudaMemcpyHostToDevice);

// initializes threads and thread blocks
     dim3 threadPerBlock(8,8,8);
     dim3 numberOfBlocks(2,2,2);

// invoke the kernel 
     stancilKernel <<<numberOfBlocks, threadPerBlock >>> (input_d,output_d);

// copy memory from device to host
     cudaMemcpy(output,output_d,size,cudaMemcpyDeviceToHost);

     // output display
 for (int i=0;i<N;i++){
    std::cout<< output[i]<< std::endl;
   }
// free some allocated memory on the device
     cudaFree(input_d);
     cudaFree(output_d);
// free allocated memory on the host
     delete[] output;
     delete[] input;

    return 0;

}