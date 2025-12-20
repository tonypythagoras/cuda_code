#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include  <cstdio>
#include  <iostream>



// the kernel that does the actual of mixing red, green and blue, turning the mixture to gray.
__global__ void rgb_to_gray(float *red, float *green, float *blue , float *gray, int width, int height){

   unsigned int row=blockIdx.y * blockDim.y+threadIdx.y;
   unsigned int col= blockIdx.x * blockDim.x + threadIdx.x;
   int globalId =row * width + col;
   if(row < height && col < width){
      
      gray[globalId]=red[globalId]*3/10 + green[globalId]*6/10 + blue[globalId]*1/10;

   }


}




int main(){

// the size of the color array
const unsigned int N=256;
const int size = N*sizeof(int);

const int width=16;
const int height=16;

// allocate memories in the host for the clor arrays
float *red =new float[N];
float *green = new float[N];
float *blue =new float[N];
float *gray= new float[N];

// initialize the colors
for(int i=0;i<N; i++){
       red[i]=255;
       green[i]=128;
       blue[i]=255;
}

float *red_d;
float *blue_d;
float *green_d;
float *gray_d;

// allocate memory in the device
cudaMalloc((void**)&red_d,size);
cudaMalloc((void**)&green_d,size);
cudaMalloc((void**)&blue_d,size);
cudaMalloc((void**)&gray_d,size);

// copy memory to device
cudaMemcpy(red_d,red,size,cudaMemcpyHostToDevice);
cudaMemcpy(blue_d,blue,size,cudaMemcpyHostToDevice);
cudaMemcpy(green_d,green,size,cudaMemcpyHostToDevice);

// allocate the threads and the thread blocks
dim3 numThreadPerBlock(8,8);
dim3 numberOfBlocks((width + numThreadPerBlock.x -1)/numThreadPerBlock.x,(height + numThreadPerBlock.y -1)/numThreadPerBlock.x);

// calls the kernel tha does the job of mixing the colors of red, green and blue, turning them to gray
  rgb_to_gray <<< numberOfBlocks, numThreadPerBlock >>>(red_d, green_d, blue_d, gray_d, width, height);

cudaMemcpy(gray,gray_d,size,cudaMemcpyDeviceToHost);


// prints out the final output which is the gray
for(int i=0;i < N; i++){
    std::cout<<gray[i]<<std::endl;
}

// free allocated device memory
cudaFree(red_d);
cudaFree(blue_d);
cudaFree(green_d);
cudaFree(gray_d);

// free allocated host memory
delete[] red;
delete[] blue;
delete[] green;
delete[] gray;


return 0;






}
