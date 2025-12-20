#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>



    unsigned const int NUMBER_OF_BLOCKS=2;
    unsigned const int N=32;
    unsigned const int NUM_THREAD=8;
    unsigned const int P=NUMBER_OF_BLOCKS;
    unsigned const int THREAD_PER_BLOCK=8;

// This performs the reduction on the array
    __global__ void reductionKernel(float *array, float *partial_sum){
         int segment = blockIdx.x * blockDim.x * 2;
         int i = segment + threadIdx.x;
         __shared__ float share_memory[THREAD_PER_BLOCK];

           share_memory[threadIdx.x]=array[i]+array[i+blockDim.x];
         
         __syncthreads();
           for (int stride = blockDim.x/2; stride >0; stride /=2){
                if(threadIdx.x < stride){
                    share_memory[threadIdx.x] += share_memory[threadIdx.x + stride];
                }
                __syncthreads();
           }

        if(threadIdx.x == 0){
        partial_sum[blockIdx.x]=share_memory[threadIdx.x];
        }

    }

    __global__ void reductionPartialSumKernel(float *partial_sum){

         int i = blockIdx.x * blockDim.x + threadIdx.x*2;
        for(int stride = 1; stride <=NUMBER_OF_BLOCKS; stride *= 2){
            if(threadIdx.x % stride == 0){
              partial_sum[i] = partial_sum[i] + partial_sum[i+stride];
            }
            __syncthreads();
        }
        }

    

    void initArray(float *array){

        for(int i=0;i<N;i++){
            array[i]=2;
        }
    }


    int main(){

       size_t size = N * sizeof(float);
       size_t p_size = P * sizeof(float);
// allocates memories on the host
       float *array_h = (float*)malloc(size);
       float *partial_sum_h = (float*)malloc(p_size);

       initArray(array_h);

       float *array_d;
       float *partial_sum_d;

       cudaError_t err;

       // allocates memories on the device
       err=cudaMalloc((void**)&array_d,size);
       if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }

       err = cudaMalloc((void**)&partial_sum_d,p_size);
       if( err != cudaSuccess){

                       fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

       }

       // copy memory from host to the device
       cudaMemcpy(array_d,array_h,size,cudaMemcpyHostToDevice);
       int n =  N/(2*NUMBER_OF_BLOCKS);
     // initializes thread and threadblocks
       dim3 threadPerBlock1(n);
       dim3 numberOfBlocks1(NUMBER_OF_BLOCKS);



// kernel that performs the array reduction
    reductionKernel <<<numberOfBlocks1, threadPerBlock1 >>>(array_d, partial_sum_d);
    cudaDeviceSynchronize();

     
     // initializes the thread and threadblocks
     int n2 = NUMBER_OF_BLOCKS/2;
      dim3 threadPerBlock2(n2);
      dim3 numberOfBlocks2(1);

     // reduction of partial sum
    reductionPartialSumKernel <<< numberOfBlocks2, threadPerBlock2 >>>(partial_sum_d);
    cudaMemcpy(partial_sum_h,partial_sum_d,p_size,cudaMemcpyDeviceToHost);

    std::cout<< "The reduction is : "<< partial_sum_h[1]<< std::endl;


// free memories on the device
    cudaFree(array_d);
    cudaFree(partial_sum_d);


// feee memories on the host
    delete[] array_h;
    delete[] partial_sum_h;
        return 0;
    }

