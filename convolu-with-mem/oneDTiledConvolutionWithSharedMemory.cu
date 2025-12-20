#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdio.h>
#include <stdlib.h>


    // constants
    unsigned const int N=64;
    unsigned const int MASK_WIDTH=5;
    unsigned const int TILE_WIDTH=16;
    unsigned const int radius=MASK_WIDTH/2;


// This kernel computes one dimensional convolution using shared memory. 
// The offssets are  still accessed from the global memory    
    __global__ void tiledConvolution(float *input, float *output, float *mask){

        // shared memory
      __shared__ float tile[TILE_WIDTH];
       int m=blockIdx.x * blockDim.x;
       int i=m + threadIdx.x;

      // loads to the shared memory
       tile[threadIdx.x]= input[i];
       __syncthreads();

       int start = i - radius;
       float sum=0.0;

       for(int j=0;j<MASK_WIDTH;j++){

        int index=start +j;
        if(index >=0 && index < N){
            int t_index=index - m;

            if(t_index >=0 && t_index < blockDim.x){
                
                sum += tile[t_index] * mask[j];

            }else{

                sum +=input[index] * mask[j];
            }
        }
       }
       output[i]=sum;
    }

// initializes input and the mask
void initArray(float *input, float *mask){
    for(int i=0;i<N; i++){
        if(i< 32){
       input[i]=1;
        }else{
        input[i]=2;
        }

        for(int i=0;i<MASK_WIDTH; i++){
            mask[i]=2;
        }
}
}

int main(){

    size_t size= N * sizeof(float);
    size_t mask_size= MASK_WIDTH * sizeof(float);

// allocates memory on the host
    float *input=(float*)malloc(size);
    float *output=(float*)malloc(size);
    float *mask=(float*)malloc(mask_size);


    initArray(input,mask);
     
     float *input_d;
     float *output_d;
     float *mask_d;

     cudaError_t err;

// allocate memory on the device
     err=cudaMalloc((void**)&input_d,size);
     if( err !=cudaSuccess){
                       fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
     }
// allocate memory on the device

     err=cudaMalloc((void**)&output_d,size);

     if(err !=cudaSuccess){
                               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

     }
// allocate memory on the device

     err=cudaMalloc((void**)&mask_d,mask_size);

     if(err != cudaSuccess){
                               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

     }

// copy memory from host to device
     cudaMemcpy(input_d,input,size,cudaMemcpyHostToDevice);
     cudaMemcpy(mask_d,mask,mask_size,cudaMemcpyHostToDevice);


// declare thread and threadblocks
     dim3 threadPerBlock(16,1);
     dim3 numberOfBlocks(4,1);


// call the kernel
     tiledConvolution <<<numberOfBlocks, threadPerBlock >>>(input_d, output_d, mask_d);


// copy memory from device to host
     cudaMemcpy(output,output_d,size,cudaMemcpyDeviceToHost);

// prints the output 
     for (int i=0;i<N;i++){
    std::cout<< output[i]<< std::endl;
   }

// free allocated device memory
cudaFree(input_d);
cudaFree(output_d);
cudaFree(mask_d);


// free allocated host memory
delete[] input;
delete[] output;
delete[] mask;

return 0;

}