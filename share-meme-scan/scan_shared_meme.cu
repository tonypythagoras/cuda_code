#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdio.h>
#include <stdlib.h>

unsigned const int N=16;
unsigned const int P = 4;
unsigned const int NUM_BLOCK=4;
unsigned const int THREAD_PER_BLOCK=4;

// This kernel computes the inclusive scan using koge-stoe using shared memory
__global__ void scanKernel(float *input, float *vector, float * partial_sum){
   int i = blockIdx.x * blockDim.x + threadIdx.x;

     __shared__ float share_memory[NUM_BLOCK];
     share_memory[threadIdx.x]=input[i];
     __syncthreads();
     for(int stride=1; stride <= THREAD_PER_BLOCK/2; stride *=2){
         float temp;
          if(threadIdx.x >= stride){
            temp=share_memory[threadIdx.x-stride];            
          }
          __syncthreads();
          if(threadIdx.x >= stride){
            share_memory[threadIdx.x] +=temp;
          }
          __syncthreads();
     }
     if(threadIdx.x == blockDim.x-1){
     partial_sum[blockIdx.x]=share_memory[threadIdx.x];
     }
     vector[i]=share_memory[threadIdx.x];

}



__global__ void scanPartial(float * partial_sum){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i < blockDim.x-1){
     for(int stride=1; stride <= (blockDim.x/2); stride *=2){

            float temp;
          if(threadIdx.x >= stride){
            temp=partial_sum[i-stride];            
          }
          __syncthreads();
          if(threadIdx.x >= stride ){
            partial_sum[i] +=temp;
          }
          __syncthreads();

     }}
    
}

void addPartialOutput(float *output, float *partial_sum){

 

    for( int i=0;i< N; i++){
        if((i/4)==1){
            output[i] +=partial_sum[0];
        }else if((i/4)==2){
            output[i] +=partial_sum[1];
        }else if((i/4)==3){
            output[i] +=partial_sum[2];
        }

    }

}

void initArray(float *vector_h){

    for(int i=0;i < N; i++){
        vector_h[i]=1;
    }
}


int main(){
size_t size= N * sizeof(float);
size_t p_size= P * sizeof(float);

float *vector_h;
float *partial_sum_h;
float *output_h;
// allocates memory on the host
vector_h=(float*)malloc(size);
output_h=(float*)malloc(size);

partial_sum_h=(float*)malloc(p_size);

initArray(vector_h);


float *vector_d;
float *partial_sum_d;
float *input_d;
cudaError_t err;
// allocates memories on the device
err=cudaMalloc((void**)&input_d,size);
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }

err=cudaMalloc((void**)&vector_d,size);
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }
err = cudaMalloc((void**)&partial_sum_d,p_size);
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }

// copy memory fro host to device
    cudaMemcpy(input_d,vector_h,size,cudaMemcpyHostToDevice);
     
     // initializes thread and threadblocks
     dim3 threadPerBlock(4);
     dim3 numberOfBlocks(4);


// kernel launch that computes scan
     scanKernel <<<numberOfBlocks, threadPerBlock >>> (input_d, vector_d, partial_sum_d);
     cudaDeviceSynchronize();
      
      //scan partial sum
     scanPartial <<<numberOfBlocks, threadPerBlock >>> (partial_sum_d);
           //cudaDeviceSynchronize();

           

// copy memories from device to host
     cudaMemcpy(output_h, vector_d,size,cudaMemcpyDeviceToHost);
     cudaMemcpy(partial_sum_h,partial_sum_d,p_size,cudaMemcpyDeviceToHost);
     
     // This adds the partial sum array to the previously scanned array for final result
     addPartialOutput(output_h, partial_sum_h);
    

   // output display
 for (int i=0;i<N;i++){
    std::cout<< output_h[i]<< std::endl;
   }

// free allocated device memories
cudaFree(vector_d);
cudaFree(partial_sum_d);
cudaFree(input_d);

// free allocated host memories
delete[] vector_h;
delete[] partial_sum_h;
delete[] output_h;

    return 0;
}