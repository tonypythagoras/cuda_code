#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdio.h>
#include <stdlib.h>

//This uses both shared memory and global memories. Shared memory for the the tiles and global memory for the halos.
// This computes the convolution for two dimensional array. It uses shared memory but loads the halos from the global memory

//define constatnts
unsigned const int N=64;
unsigned const int M_N=25;
unsigned const int width = 8;
unsigned const int tile_width=4;
unsigned const int m_width=5;
unsigned const int radius=m_width/2;


// this computes the convolution for two dimensional array. It uses shared memory but loads the halos from global memory
__global__ void tiledConvolution(float *input, float *output, float *mask){
       __shared__ float tile[tile_width][tile_width];

       int mx = blockIdx.x * blockDim.x;
       int my = blockIdx.y * blockDim.y;
       int col= mx + threadIdx.x;
       int row= my + threadIdx.y;
       int start_x = col - radius;
       int start_y = row - radius;

         tile[threadIdx.y][threadIdx.x]=input[row * width +col];
         __syncthreads();


         float sum=0.0;
       for(int i=0; i<m_width; i++){
        int index_i = i + start_x;

        for (int j=0; j<m_width; j++){

          int index_j = j + start_y;

          if(index_i >= 0 && index_j >= 0 && index_i < width && index_j < width){

            int tx = index_i - mx;
            int ty = index_j - my;
            if(tx >=0 && ty >=0 && tx < blockDim.x && ty < blockDim.y){
             
                  sum += tile[ty][tx] * mask[(i * m_width) +j];

            }else{

              sum += input[row * width +col] * mask[(i * m_width) +j];
            }
          }
        } 
       }

       output[row * width + col]=sum;
          

}

// This initializes the array
void initArray(float *input, float *mask){

     for(int i=0; i<N; i++){
        if(i<32){
            input[i]=1
        }else{
            input[i]=2;
        }
     }

     for(int i=0;i<M_N; i++){
             mask[i]=1;
     }


}


int main(){

    size_t size =N * sizeof(float);
    size_t mask_size= M_N * sizeof(float);


    // allocates memories on the host
    float *input=(float*)malloc(size);
    float *output=(float*)malloc(size);
    float *mask=(float*)malloc(mask_size);

    initArray(input,mask);


    float *input_d;
    float *output_d;
    float *mask_d;

   //Allocates memories on the device
    cudaError_t err;
    err=cudaMalloc((void**)&input_d, size);
    if(err !=cudaSuccess){

         fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

    }

    err=cudaMalloc((void**)&output_d,size);

    if(err !=cudaSuccess){

      fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

    }


  err=cudaMalloc((void**)&mask_d,mask_size);

  if(err !=cudaSuccess){
    fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

  }

   // Copy memories from host to device
   cudaMemcpy(input_d,input,size,cudaMemcpyHostToDevice);
   cudaMemcpy(mask_d,mask,mask_size,cudaMemcpyHostToDevice);

  // initialize thread and thread blocks
   dim3 threadPerBlock(4,4);
   dim3 numberOfBlocks(2,2);

// invokes the kernel code
   tiledConvolution <<<numberOfBlocks, threadPerBlock>>>(input_d, output_d, mask_d);

     cudaMemcpy(output,output_d,size,cudaMemcpyDeviceToHost);

     // prints the output 
     for (int i=0;i<N;i++){
    std::cout<< output[i]<< std::endl;
   }


// free allocated memories on the device
cudaFree(input_d);
cudaFree(output_d);
cudaFree(mask_d);

// free allocated memories on the host
delete[] output;
delete[] input;
delete[] mask;

return 0;

}