#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>

unsigned const int N=16;
unsigned const int MASK_WIDTH=3;
unsigned const int radius=MASK_WIDTH/2;
unsigned const int MASK_SIZE=9;
unsigned const int TILE_SIZE=9;
unsigned const int TILE_WIDTH=4;
unsigned const int TILE_HEIGTH=4;

unsigned const int BLOCK_WIDTH=8;
unsigned const int BLOCK_HEIGTH=8;


static __constant__ float Mc[MASK_SIZE];


// kernel function
__device__ int blockNumber(int x,int y){
    if(x==0 && y==0){
        return 0;
    }else if(x==0 && y==1){
        return 1;
    }else if(x==1 && y== 0){
        return 2;
    }else if(x==1 && y==1){
        return 3;
    }else{
        return -1;
    }
}

// kernel function
__device__ int number(int x,int y){
    if(x==1 && y==1){
        return 0;
    }else if(x==1 && y==2){
        return 1;
    }else if(x==2 && y== 1){
        return 2;
    }else if(x==2 && y==2){
        return 3;
    }else{
        return -1;
    }
}


//This computes 2D tiled convolution using shared memory
__global__ void tiledConvolution(float *input, float *output, int *test){

       __shared__ float tile[TILE_WIDTH][TILE_HEIGTH];
       int col = blockIdx.x * blockDim.x + threadIdx.x;
       int row = blockIdx.y * blockDim.y + threadIdx.y;
       int bx = blockIdx.x;
       int by = blockIdx.y;
       int r1 = row - (2 * by);
       int c1 = col - (2 * bx);
       int col1 = c1 - radius;
       int row1 = r1 - radius;
       int index1= row1 * TILE_WIDTH + col1;
       
       if(col <= 0){
        tile[threadIdx.y][threadIdx.x]=0;
       } else if(row <=0){
         tile[threadIdx.y][threadIdx.x]=0;
       }else if(col >= (BLOCK_HEIGTH-1)){
          tile[threadIdx.y][threadIdx.x]=0;

       }else if(row >= (BLOCK_WIDTH-1)){
           tile[threadIdx.y][threadIdx.x]=0;

       }else{
         tile[threadIdx.y][threadIdx.x]=input[index1];
       }

    __syncthreads();
     int ind=blockNumber(blockIdx.y,blockIdx.x) * 4 + number(threadIdx.y, threadIdx.x);
       
       float sum=0.0;
            int num=0;
            int start_i=threadIdx.y - radius;
            int start_j=threadIdx.x - radius;

            if(number(threadIdx.x, threadIdx.y)>=0){
          
                  for(int i=start_i;i< (3+start_i); i++){

                    for( int j=start_j; j<(3+start_j); j++){
                       sum +=tile[i][j] * Mc[num];
                       num++;

                    }
                  }
                output[ind]=sum;

            }
            
            
    

}

// initializes the array
void initArray(float *input, float *mask){

    for(int i=0;i<N; i++){
        input[i]=i+1;
    }

    for(int i=0;i<MASK_SIZE; i++){
        mask[i]=1;
    }

}




int main(){

size_t size= N * sizeof(float);
size_t mask_size = MASK_SIZE * sizeof(float);

// memory allocation on the host
float *input=(float*)malloc(size);
float *output=(float*)malloc(size);
float *mask= (float*)malloc(mask_size);

initArray(input,mask);

float *input_d;
float *output_d;
float *mask_d;

cudaError_t err;

// memory allocation on the device
err=cudaMalloc((void**)&input_d,size);
if( err != cudaSuccess){
     fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

}

err = cudaMalloc((void**)&output_d,size);

if( err != cudaSuccess){
     fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

}

err = cudaMalloc((void**)&mask_d,mask_size);
if(err != cudaSuccess){
         fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

}

// copy memory to the device
cudaMemcpy(input_d,input,size,cudaMemcpyHostToDevice);
// copy memory to the constant memory
cudaMemcpyToSymbol(Mc,mask,mask_size);

// allocates threads and threadblocks
dim3 threadPerBlock(4,4);
dim3 numberOfBlocks(2,2);

// calls the tiled convolution kernel
tiledConvolution <<< numberOfBlocks, threadPerBlock >>>(input_d,output_d,test_d);

// copy memory to the host
cudaMemcpy(input,input_d,size,cudaMemcpyDeviceToHost);

cudaMemcpy(output,output_d,size,cudaMemcpyDeviceToHost);


// output display
 for (int i=0;i<N;i++){
    std::cout<< output[i]<< std::endl;
   }


// free device allocated memories
   cudaFree(input_d);
   cudaFree(output_d);

// free host allocated memories
   delete[] input;
   delete[] output;
   delete[] mask;

    return 0;
}