#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>



// This kernel computes the convolution for one dimensional array
__global__ void convo1DArray(float *a, float *p, float *m, int N, int m_width, int rad){

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int start = i - rad;

    float sum=0.0;

    for(int j=0;j<m_width; j++){

        if((start +j)>=0 && (start + j)<N){
          
         sum += a[j+start]*m[j];   
        }
    }

    p[i]=sum;
}

// initializes the array
void initArray(float *a){
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
}

// initializes the convolution masks
void initMask(float *m){
   m[0]=1;
   m[1]=2;
   m[2]=1;
   m[3]=2;
   m[4]=1;


}



int main(){
    unsigned const int N = 10;
    unsigned const int MASK_WIDTH=5;
    unsigned const int M_RADIUS=MASK_WIDTH/2;
    const int size = N * sizeof(float);
    const int mask_size = MASK_WIDTH * sizeof(float);

   
   // allocate memories on the host
   float *a= (float*)malloc(size);
    float *m= (float*)malloc(mask_size);
    float *p= (float*)malloc(size);

    

    initArray(a);
    initMask(m);

    float *a_d;
    float *m_d;
    float *p_d;
     


     // allocate memories on the device with error checks
     cudaError_t err;
     err=cudaMalloc((void**)&p_d,size);

     if(err !=cudaSuccess){

                   fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

     }

     err=cudaMalloc((void**)&a_d,size);

     if(err !=cudaSuccess){

                   fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

     }

     err= cudaMalloc((void**)&m_d,mask_size);

     if(err !=cudaSuccess){
                   fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 

     }

// copy memories from host to the device
     cudaMemcpy(a_d,a,size,cudaMemcpyHostToDevice);
     cudaMemcpy(m_d,m,mask_size,cudaMemcpyHostToDevice);

     // launch the kernel
     convo1DArray <<< 1, 10 >>>(a_d,p_d,m_d,N,MASK_WIDTH,M_RADIUS);
     
     // copy memory from the device to host
     cudaMemcpy(p,p_d,size,cudaMemcpyDeviceToHost);


for (int i=0;i<N;i++){
    std::cout<< p[i]<< std::endl;
   }

//free allocated device memories
   cudaFree(a_d);
   cudaFree(m_d);
   cudaFree(p_d);


// free allocated host memories
   delete[] a;
   delete[] m;
   delete[] p;

   return 0;

}