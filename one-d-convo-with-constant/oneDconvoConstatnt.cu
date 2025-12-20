#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>




const int m_width=5;

//constant memory
static __constant__ float Mc[m_width];

// This kernel computes the one dimensional convolution array
 __global__ void oneDconvoConstant(float *input, float *output, int width, int radius){
     int row= blockIdx.y * blockDim.y +threadIdx.y;
     int col= blockIdx.x * blockDim.x +threadIdx.x;

     int start=col - radius;
     float sum=0.0;

     for(int i=0; i< m_width; i++){
         if((start +i)>= 0 && (start + i)< width){
            sum += input[start+i] * Mc[i];
         }

     }

     output[col]=sum;

 }

// initializes the input array and the mask
void initArray( float *a, float *m){
    
   a[0]=1;
   a[1]=2;
   a[2]=3;
   a[3]=1;
   a[4]=2;
   a[5]=3;
   a[6]=1;
   a[7]=2;
   a[8]=3;
   a[9]=1;



   m[0]=1;
   m[1]=2;
   m[2]=1;
   m[3]=2;
   m[4]=1;

}



int main(){


unsigned int N=10;
unsigned int radius=2;

size_t size= N * sizeof(float);
size_t mask_size =m_width * sizeof(float);

// allocate host memory
float *input =(float*)malloc(size);
float *mask= (float*)malloc(mask_size);
float *output =(float*)malloc(size);



initArray(input,mask);

float *input_d;
float *output_d;

cudaError_t err;

// allocates memory in the device
err=cudaMalloc((void**)&input_d,size);
 if(err != cudaSuccess){
                          fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
 }


 err=cudaMalloc((void**)&output_d,size);

 if(err !=cudaSuccess){
                       fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

 }

 // copies memory to constant memory
 cudaMemcpyToSymbol(Mc,mask,mask_size);
 // copies memory to the device
 cudaMemcpy(input_d,input,size,cudaMemcpyHostToDevice);

// this invokes the kernel
 oneDconvoConstant <<<1,10 >>>(input_d,output_d,N,radius);
// copies memory from device to host
 cudaMemcpy(output,output_d,size, cudaMemcpyDeviceToHost);



for (int i=0;i<N;i++){
    std::cout<< output[i]<< std::endl;
   }

// free memories allocated on the device
cudaFree(input_d);
cudaFree(output_d);

// free memories on the host
delete[] input;
delete[] output;
delete[] mask;
    return 0;

}