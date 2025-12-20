#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>



    unsigned const int NUMBER_OF_BLOCKS=2;
    unsigned const int N=16;
    unsigned const int P=NUMBER_OF_BLOCKS;
    unsigned const int THREAD_PER_BLOCK=8;

//  The naive reduction kernel. This computes the reduction of numbers ina an array
    __global__ void reductionKernel(float *array, float *partial_sum){
         int segment = blockIdx.x * blockDim.x * 2;
         int i = segment + threadIdx.x * 2;

         if( i < N){

             for(int stride = 1; stride <=blockDim.x; stride *= 2){
            if(threadIdx.x % stride == 0){
              array[i] = array[i] + array[i+stride];
            }
            __syncthreads();
        }

        if(threadIdx.x == 0){
        partial_sum[blockIdx.x]=array[i];
        }
         }


    }

    __global__ void reductionPartialSumKernel(float *partial_sum){

        int i = blockIdx.x * blockDim.x + threadIdx.x*2;
        if(i < P){
        for(int stride = 1; stride <=NUMBER_OF_BLOCKS; stride *= 2){
            if(threadIdx.x % stride == 0){
              partial_sum[i] = partial_sum[i] + partial_sum[i+stride];
            }
            __syncthreads();
        }
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

       // initializes the thread and threadblocks
       dim3 threadPerBlock(4);
       dim3 numberOfBlocks(2);


// kernel launch that performs the reduction
    reductionKernel <<<numberOfBlocks, threadPerBlock >>>(array_d, partial_sum_d);
    cudaDeviceSynchronize();
    // reduce partial sum
    reductionPartialSumKernel <<< numberOfBlocks, threadPerBlock >>>(partial_sum_d);
    // copy memory from device to host
    cudaMemcpy(partial_sum_h,partial_sum_d,p_size,cudaMemcpyDeviceToHost);

      // output display
    std::cout<< "The reduction is : "<< partial_sum_h[0]<< std::endl;
 
// free memories on the device
    cudaFree(array_d);
    cudaFree(partial_sum_d);

//free some memories on the host
    delete[] array_h;
    delete[] partial_sum_h;
        return 0;
    }

