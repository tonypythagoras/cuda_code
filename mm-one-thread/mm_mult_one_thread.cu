#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <iostream>


//this is the kernel that does the matrix multiplication but with only one thread
__global__ void mm_one_tread(float *a, float *b, float *c, int width,int N){

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

    unsigned int N= 256;
    unsigned int size= N*sizeof(int);
    unsigned int width= 16;

    float *a=new float[N];
    float *b= new float[N];
    float *c = new float[N];


    // initialize the array in the host
     for (int i=0;i<N; i++){
       a[i]=i+2;
       b[i]=i+1;
     }

     float *a_d;
     float *b_d;
     float *c_d;
     
     // allocate memories on the device
     cudaMalloc((void**)&a_d,size);
     cudaMalloc((void**)&b_d,size);
     cudaMalloc((void**)&c_d,size);


// copy memories from host to device
     cudaMemcpy(a_d,a,size,cudaMemcpyHostToDevice);
     cudaMemcpy(b_d,b,size,cudaMemcpyHostToDevice);

      // allocate thread and thread blocks
     dim3 threadPerBlock(1,1);
     dim3 numberOfBlocks(threadPerBlock.x, threadPerBlock.y);

//kernel call
mm_one_tread <<<numberOfBlocks, threadPerBlock >>>(a_d,b_d,c_d,width, N);
// copy memory from device to host
 cudaMemcpy(c,c_d,size,cudaMemcpyDeviceToHost);


// print out the result in the the host
 for(int i=0;i<N;i++){
    std::cout<<c[i]<<std::endl;
 }

// free allocated device memories
cudaFree(a_d);
cudaFree(b_d);
cudaFree(c_d);

// free allocated host memories
delete[] a;
delete[] b;
delete[] c;

return 0;









}