#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>


// constants
unsigned const int N=16;
unsigned const int MASK_WIDTH=5;
unsigned const int radius=2;
unsigned const int BLOCK_SIZE=8;
unsigned const int GRID_SIZE=32;
//constant memory
static __constant__ float Mc[MASK_WIDTH];



// computes one dimensional tiled convolution using shared memory
__global__ void  tiledConvolution(float *input, float *output){
    // shared memory array
    __shared__ float tile[BLOCK_SIZE];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int bl  = blockIdx.x;


    int f1 = col - (radius + (bl * radius));
        f1= f1-bl * radius;
    int f2= f1-(radius * radius * bl);


    if(f1 < 0){

        tile[threadIdx.x]=0;
    }else if(f1>=(GRID_SIZE - 1)){

         tile[threadIdx.x]=0;
    }else{

        tile[threadIdx.x]=input[f1];
    }
    __syncthreads();
    float sum=0.0;


    if(f2 >=0 && f2 < 4){
                   
        for(int j=0; j<MASK_WIDTH; j++){
             sum +=tile[j+f2] * Mc[j];
        }
                output[f1]=sum;
    }
}


// initilizes the array
void initArray(float *input, float *mask){
    for(int i=0;i<N; i++){
       if(i<9){
        input[i]=i+1;
       }else{
        input[i]=1;
       }
    }

    for(int i=0;i<MASK_WIDTH; i++){
        mask[i]=1;
    }

   

}


int main(){
    size_t size = N * sizeof(float);
    size_t mask_size= MASK_WIDTH * sizeof(float);

    size_t sd= 32 * sizeof(int);
    int *test= (int*)malloc(sd);

// allocates memory on the host
    float *input= (float*)malloc(size);
    float *output =(float*)malloc(size);
    float *mask =(float*)malloc(mask_size);

    initArray(input,mask);

    float *output_d;
    float *input_d;
    


// allocates memory on the device
    cudaError_t err;
    err = cudaMalloc((void**)&output_d,size);
    if(err !=cudaSuccess){

              fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
    }
     
     err = cudaMalloc((void**)&input_d,size);

     if( err !=cudaSuccess){
              fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

     }

// copy allocated memory to the device
     cudaMemcpy(input_d,input,size,cudaMemcpyHostToDevice);
     cudaMemcpyToSymbol(Mc,mask,mask_size);
     // allocates threads and threadblocks
     dim3 threadPerBlock(8,1);
     dim3 threadblocks(4,1);

      

// calls the tiled convolution kernel
     tiledConvolution <<<threadblocks, threadPerBlock >>>(input_d,output_d);
     // copy memory to the host
     cudaMemcpy(output,output_d,size,cudaMemcpyDeviceToHost);

// output display
     for (int i=0;i<N;i++){
    std::cout<< output[i]<< std::endl;
   }

// free memory allocated on the device
cudaFree(input_d);
cudaFree(output_d);

// free memories allocated on the host
delete[]input;
delete[] output;
delete[] mask;
    
    return 0;
}