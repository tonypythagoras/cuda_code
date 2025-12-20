#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>


unsigned const int N = 16;
 
 // This computes the frequency of numbers in an array
__global__ void hisKernel(int *numbers, int *frequency){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride= blockDim.x * gridDim.x;
    while(i < N){
      atomicAdd(&(frequency[numbers[i]]),1);
      i=i+stride;
    }



}


// init array
void initArray(int *numbers){

numbers[0]=9;
numbers[1]=9;
numbers[2]=8;
numbers[3]=8;

numbers[4]=1;
numbers[5]=1;
numbers[6]=1;
numbers[7]=1;

numbers[8]=3;
numbers[9]=3;
numbers[10]=3;
numbers[11]=3;
numbers[12]=3;

numbers[13]=1;
numbers[14]=1;
numbers[15]=8;

    
}





int main(){

size_t size = N * sizeof(int);
// allocates some host memories
int *numbers =(int*)malloc(size);
int *frequency =(int*)malloc(size);

initArray(numbers);

int *numbers_d;
int *frequency_d;


//allocates some device memories
cudaError_t err;
err = cudaMalloc((void**)&numbers_d,size);

if(err !=cudaSuccess){
     fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
}

err = cudaMalloc((void**)&frequency_d,size);

if(err !=cudaSuccess){
     fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
}
//copy memory from host to the device
cudaMemcpy(numbers_d,numbers,size,cudaMemcpyHostToDevice);


// initializes thread and threadblocks
dim3 threadPerBlock(8);
dim3 numberOfBlocks(2);

// kernel launch
hisKernel <<<numberOfBlocks, threadPerBlock >>> (numbers_d, frequency_d);

//copty memory from device to host
cudaMemcpy(frequency,frequency_d,size,cudaMemcpyDeviceToHost);
//display result
for (int i=0;i<N;i++){
    std::cout<< frequency[i]<< std::endl;
   }


//de-allocate device memories
cudaFree(frequency_d);
cudaFree(numbers_d);


//de-allocates host memories
delete[] frequency;
delete[] numbers;

    return 0;
}
