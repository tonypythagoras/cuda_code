#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


// This kernel computes matrix multplication with any arbitrary width or length
const int tileWidth=2;

__global__ void tileMux_with_bound_check(float *a, float *b, float *c, int width, int Num){
      
      // shared memories
    __shared__ float M[tileWidth][tileWidth];
    __shared__  float N[tileWidth][tileWidth];

    int tx=threadIdx.x;
    int ty=threadIdx.y;
    int bx=blockIdx.x;
    int by=blockIdx.y;
    int row = tileWidth * by + ty;
    int col = tileWidth * bx + tx;

    float sum=0.0;

    for(int i=0;i < ((width-1)/tileWidth)+1; i++){
           
           if(row < width && (i * tileWidth + tx) < width ){
                M[ty][tx]=a[row * width + (i * tileWidth + tx)];
           }else{

            M[ty][tx]=0.0;

           }
           if((i * tileWidth +ty)< width && col < width){
           N[ty][tx]=b[(i * tileWidth +ty)* width + col];
           }else{
            N[ty][tx]=0.0;
           }
           // the synchronization mechanism that allow all the threads to finish loading before the computation
           __syncthreads();

           for(int j=0; j<tileWidth; j++){
                
                sum +=M[ty][j]*N[j][tx];
                // Synchronize to finish computation before another load
                __syncthreads();
           }
         
    }

     // output result
    if(row < width && col < width){
    c[row * width + col]=sum;
    }

}

// initialize the matrix
void initMatrix(float *a,float *b, int N){

     for (int i=0;i<N; i++){
      if(i==3||i==4||i==5){
       a[i]=2.0;
      }else{
         a[i]=1.0;
      }
       b[i]=2.0;
     }


}


int main(){
 unsigned const int N=9;
 const size_t size= N * sizeof(float);
 const int width=3;
 const int tileWidth=2;

// memory allocations on the host
 float *a =(float*)malloc(size);
 float *b= (float*)malloc(size);
 float *c= (float*)malloc(size);

if(!a || !b || !c){
    exit(EXIT_FAILURE);

}

initMatrix(a,b, N);

float *a_d;
float *b_d;
float *c_d;



// memory allocations on the device with error checking
cudaError_t err;
err = cudaMalloc((void**)&a_d,size);
if(err !=cudaSuccess){
           fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

}
 err= cudaMalloc((void**)&b_d,size);
 if(err !=cudaSuccess){
           fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

 }

 err=cudaMalloc((void**)&c_d, size);
 if(err !=cudaSuccess){
           fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

 }

// copy memories from host to the device
cudaMemcpy(a_d,a,size,cudaMemcpyHostToDevice);
cudaMemcpy(b_d,b,size,cudaMemcpyHostToDevice);
 
 // initialize your threads and thread blocks
 dim3 threadPerBlock(2,2);
 dim3 numberOfBlocks((width + threadPerBlock.x -1) /threadPerBlock.x, (width + threadPerBlock.y -1)/threadPerBlock.y);
   
   // launch the kernel
   tileMux_with_bound_check <<< numberOfBlocks,threadPerBlock >>>(a_d,b_d,c_d,width,N);
    
    // copy memory from the device to host
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