#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>



// This is the kernel that does the tiled matrix multplication
const int tileWidth=2;

__global__ void tileMux(float *a, float *b, float *c, int width){
             // shared memories. shared between threads in the same block
           __shared__ float M[tileWidth][tileWidth];
           __shared__ float N[tileWidth][tileWidth];
           int tx = threadIdx.x;
            int ty = threadIdx.y;
            int bx = blockIdx.x;
            int by = blockIdx.y;
            int row =tileWidth * by + ty;
            int col = tileWidth *bx + tx;
            int pv=0;
            for (int m=0; m<(width/tileWidth); m++){
               M[ty][tx]=a[row * width + (m * tileWidth) +tx];
               N[ty][tx]= b[(m * tileWidth + ty)* width + col];
               // synchronization mechanism among threads in a block
               // it is also called barrier synchronization
               // This allows all the thread to finish loading before proceeding with the computation
                __syncthreads();
                for(int k=0; k<tileWidth; k++){
                  pv += M[ty][k]* N[k][tx];
                }
                // This barrier synchronization allows threads in a block to finish computation before loading the second block
                __syncthreads();
            }
            // placed in the output
            c[row * width + col]=pv;

        }


// this function initializes the matrix
void initMatrix (float *a, float *b, int N){

     for (int i=0;i<N; i++){
      if(i==4||i==13||i==15){
       a[i]=2.0;
      }else{
         a[i]=1.0;
      }
       b[i]=2.0;
     }

}




int main(){

unsigned const int N=16;
const size_t size =N * sizeof(float);
unsigned int width=4;
unsigned int TILE_WIDTH=2;
// allocate memory on the host
float *a= (float*)malloc(size);
float *b= (float*)malloc(size);
float *c= (float*)malloc(size);
   if(!a || !b || !c){
    exit(EXIT_FAILURE);
   }

   initMatrix(a,b,N);


   float *a_d;
   float *b_d;
   float *c_d;

// allocate memories on the device with error checks
   cudaError_t err;
   err=cudaMalloc((void**)&a_d,size);
   if(err != cudaSuccess){
       fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

   }
   err=cudaMalloc((void**)&b_d,size);
    if(err != cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

    }

    err=cudaMalloc((void**)&c_d,size);


    if(err !=cudaSuccess){

        fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

    }
   
   // copy from host to device
   cudaMemcpy(a_d,a,size,cudaMemcpyHostToDevice);
   cudaMemcpy(b_d,b,size,cudaMemcpyHostToDevice);


   // initialize thread blocks
   dim3 threadPerBlock(2,2);
   dim3 numberOfBlocks((width + threadPerBlock.x - 1)/threadPerBlock.x,(width + threadPerBlock.y -1)/threadPerBlock.y);
   tileMux<<<numberOfBlocks, threadPerBlock>>>(a_d,b_d,c_d, width);
   cudaMemcpy(c,c_d,size,cudaMemcpyDeviceToHost);
   
   for (int i=0;i<N;i++){
    std::cout<< c[i]<< std::endl;
   }

// free device memories
   cudaFree(a_d);
   cudaFree(b_d);
   cudaFree(c_d);

// free host memories
   delete[] a;
   delete[] b;
   delete[] c;

   return 0;

}