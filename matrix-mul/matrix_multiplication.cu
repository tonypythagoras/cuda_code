#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iostream>


//this is the kernel that does the matrix multiplication
__global__ void matrix_m(float *a, float *b, float *c, int width,int N){

    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
     int globalId=row*width+col;


if(globalId < N){
    // assigs the row multiplicant
     int j=globalId % width;
     // checks the row multiplicant
     int myRow=globalId/width;
     int sum=0;
     for(int i=0;i < width; i++){
             sum +=a[myRow*width+i]*b[j];
             j +=width;

     }
     c[globalId]=sum;
}

}

int main(){
    unsigned int N = 256;
    unsigned int size= N * sizeof(int);
    unsigned int width=16;

    // allocates  memory in the host
    float *a=new float[N];
    float *b=new float[N];
    float *c=new float[N];
     

// initialize the array in the host
     for (int i=0;i<N; i++){
       a[i]=i+2;
       b[i]=i*2;


     }
    


    float *a_d;
    float *b_d;
    float *c_d;
    // allocates device memory
    cudaMalloc((void**)&a_d,size);
    cudaMalloc((void**)&b_d,size);
    cudaMalloc((void**)&c_d,size);

    // copy memory to the device
    cudaMemcpy(a_d,a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(b_d,b,size,cudaMemcpyHostToDevice);


    // assigs thread per block
    dim3 threadPerBlock(8,8);
    
    // assigns thread blocks
    dim3 numberOfBlocks((width + threadPerBlock.x -1)/threadPerBlock.x, (width + threadPerBlock.y - 1)/threadPerBlock.y);

    matrix_m <<<numberOfBlocks, threadPerBlock>>>(a_d, b_d, c_d, width,N);

// copy memory back to the host
    cudaMemcpy(c,c_d,size,cudaMemcpyDeviceToHost);
// prints the result
    for(int i=0;i<N;i++){
        std::cout<<c[i]<<std::endl;
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    delete[] a;
    delete[] b;
    delete[] c;

    return 0;

}