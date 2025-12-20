#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdio.h>
#include <stdlib.h>

unsigned const int N=16;
unsigned const int P = 2;
unsigned const int NUM_BLOCK=2;
unsigned const int THREAD_PER_BLOCK=4;
unsigned const int SEG_BLOCK=8;

// This kernel computes the inclusive scan of an array using brent-kung approach
__global__ void scanKernel(float *input, float *vector, float * partial_sum){
   int seg = blockIdx.x * blockDim.x * 2;
   __shared__ float buf[SEG_BLOCK];
    buf[threadIdx.x]=input[seg + threadIdx.x];
    buf[threadIdx.x + THREAD_PER_BLOCK]=input[seg + threadIdx.x + THREAD_PER_BLOCK];
    __syncthreads();

    for(int s=1; s <= THREAD_PER_BLOCK; s *=2){
        int i = (threadIdx.x+1)*2*s-1;
        if(i < SEG_BLOCK){
            buf[i] +=buf[i-s];
        }
        __syncthreads();
        }
         
         for(int s=THREAD_PER_BLOCK/2; s>=1; s /=2){
           
           int i=(threadIdx.x + 1)*2*s-1;
           if((i+s)<(2*THREAD_PER_BLOCK)){
            buf[i+s] +=buf[i];
           }
         __syncthreads();

         }
        if(threadIdx.x == 0){
            partial_sum[blockIdx.x]=buf[2 * THREAD_PER_BLOCK -1];
        }

        vector[seg + threadIdx.x] = buf[threadIdx.x];
        vector[seg +threadIdx.x + THREAD_PER_BLOCK]=buf[threadIdx.x+THREAD_PER_BLOCK];

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
        if((i/8)==1){
            output[i] +=partial_sum[0];
        }else if((i/8)==2){
            output[i] +=partial_sum[1];
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
// allocates some memories on the host
vector_h=(float*)malloc(size);
output_h=(float*)malloc(size);

partial_sum_h=(float*)malloc(p_size);

initArray(vector_h);


float *vector_d;
float *partial_sum_d;
float *input_d;





// allocates some memories on the device
cudaError_t err;

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

   // copy memory from host to device
    cudaMemcpy(input_d,vector_h,size,cudaMemcpyHostToDevice);
     //initializes thread and thread blocks
     dim3 threadPerBlock(4);
     dim3 numberOfBlocks(2);

      // launch the kernel that computes the scan using brent-kung approach
     scanKernel <<<numberOfBlocks, threadPerBlock >>> (input_d, vector_d, partial_sum_d);
     cudaDeviceSynchronize();
      
      // scan the partial sum
     scanPartial <<<numberOfBlocks, threadPerBlock >>> (partial_sum_d);
           //cudaDeviceSynchronize();

           

// copy memory from device to host
     cudaMemcpy(output_h, vector_d,size,cudaMemcpyDeviceToHost);
     cudaMemcpy(partial_sum_h,partial_sum_d,p_size,cudaMemcpyDeviceToHost);
     
     // This adds the partial sum array to the previously scanned array for final result
     addPartialOutput(output_h, partial_sum_h);
    

   // output display
 for (int i=0;i<N;i++){
    std::cout<< output_h[i]<< std::endl;
   }

// free memories on the device
cudaFree(vector_d);
cudaFree(partial_sum_d);
cudaFree(input_d);

// free memories on the host
delete[] vector_h;
delete[] partial_sum_h;
delete[] output_h;

    return 0;
}