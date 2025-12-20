#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "iostream"



// This kernel computes two dimensional convolution
__global__ void twoDConvoKernel(float *input, float *output, float *mask,int width, int N, int m_width,int m_r){
       
       int row= blockIdx.y * blockDim.y + threadIdx.y;
       int col= blockIdx.x * blockDim.x + threadIdx.x;
       float sum=0.0;
       for (int i=0; i < m_width; i++ ){
         for (int j=0; j< m_width; j++){
            int in_r = row - m_r +i;
            int in_c = col - m_r +j;

              if((in_r) >= 0 &&(in_r < width) && (in_c) >= 0 &&(in_c < width)){
                sum += input[(in_r  *width)+(in_c)] * mask[i * m_width + j ];

              }
         }
       }
       output[row * width + col]=sum;
}

// intilizes the arrays
void initArray(float *input, float *mask){

  for(int i=0;i<9;i++){
    input[i]=2;
    mask[i]=1;
  }
}



int main(){

    unsigned int N = 9;
    unsigned int width = 3;
    unsigned int M_WIDTH = 3;
    unsigned int M_RADIUS = 1;
    size_t size = N * sizeof(float);

   // create and allocates memories on the host
    float *input=(float*)malloc(size);
    float *output=(float*)malloc(size);
    float *mask = (float*)malloc(size);
     
     initArray(input,mask);


     float *input_d;
     float *mask_d;
     float *output_d;
     
     // creates and allocates device memories
     cudaError_t err;
     err=cudaMalloc((void**)&input_d,size);
     if(err !=cudaSuccess){
                   fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
     }

     err=cudaMalloc((void**)&mask_d,size);
     if(err !=cudaSuccess){
                           fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

     }

     err=cudaMalloc((void**)&output_d,size);

     if(err !=cudaSuccess){
                           fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

     }

// copy memories to the device
     cudaMemcpy(input_d,input,size,cudaMemcpyHostToDevice);
     cudaMemcpy(mask_d,mask,size,cudaMemcpyHostToDevice);


// initializes threads and thread blocks
 dim3 threadPerBlock(3,3);
 dim3 numberOfBlocks(1,1);

// kernel launch
    twoDConvoKernel <<<numberOfBlocks,threadPerBlock >>>(input_d,output_d,mask_d,width,N,M_WIDTH,M_RADIUS);

 
 // copy memory fro device to host
     cudaMemcpy(output,output_d,size,cudaMemcpyDeviceToHost);


for (int i=0;i<N;i++){
    std::cout<< output[i]<< std::endl;
   }

// free and reclaim device allocated memories
cudaFree(input_d);
cudaFree(mask_d);
cudaFree(output_d);

// free and reclaim host allocated memories
delete[] input;
delete[] output;
delete[] mask;


    return 0;

}
