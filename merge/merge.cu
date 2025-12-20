#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdlib.h>
#include <stdio.h>


unsigned const int N_A = 10;
unsigned const int N_B = 6;
unsigned const int N_C = 16;
unsigned const int COARSE_FACTOR=4;
 
 // This computes the frequency of numbers in an array
__device__ int coRank(int *a, int *b, int m, int n, int k){

     int low=(k>n)? (k-n):0;
     int high=(k<m)?k:m;
     while(true){
        int i=(low+high)/2;
        int j= k- 1;

        if(i>00 && j< n && a[i-1] > b[j]){
            high=i;
        }else if(j >0 && i<m &&b[j-1] > a[i]){
             low=i;
        }else{
            return i;
        }
     }
    }

    __global__ void mergeKernel(int *a, int *b, int *c, int m, int n){

        int k= (blockIdx.x * blockDim.x + threadIdx.x)* COARSE_FACTOR;
          if(k < (m+n)){

            int i= coRank(a,b,m,n,k);
            int j= k-i;
            int kNext= (k+COARSE_FACTOR < m+n)? (k + COARSE_FACTOR): (m+n);
            int iNext= coRank(a,b,m,n,kNext);
            int jNext = kNext - iNext;
            mergeSeque(&a[i], &b[j], &c[k],iNext-i, jNext- j);
          }
    }

  __device__  void mergeSeque(int *a, int *b, int *c, int m, int n){
    int i=0;
    int j=0;
    int k=0;
    while(i < m && j < n){
        if(a[i] < b[j]){
            c[k++]=a[i++];
        }else{
            c[k++]=b[j++];
        }
    }

    while(i< m){
        c[k++]=a[i++];
    }

    while(j<n){
        c[k++]=b[j++];
    }
  }


// init array
void initArray(int *a, int *b){

a[0]=1;
a[1]=2;
a[2]=3;
a[3]=5;
a[4]=6;
a[5]=10;
a[6]=11;
a[7]=12;
a[8]=13;
a[9]=16;

b[0]=4;
b[1]=7;
b[2]=8;
b[3]=9;
b[4]=14;
b[5]=15;
}





int main(){

size_t size = N_C * sizeof(int);
size_t size_a = N_A * sizeof(int);
size_t size_b = N_B * sizeof(int);

// allocates some host memories
int *c =(int*)malloc(size);
int *a =(int*)malloc(size_a);
int *b =(int*)malloc(size_b);


initArray(a,b);

int *a_d;
int *b_d;
int *c_d;


//allocates some device memories
cudaError_t err;

err = cudaMalloc((void**)&c_d,size);

if(err !=cudaSuccess){
     fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
}


err = cudaMalloc((void**)&a_d,size_a);

if(err !=cudaSuccess){
     fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
}

err = cudaMalloc((void**)&b_d,size_b);

if(err !=cudaSuccess){
     fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
}
//copy memory from host to the device
cudaMemcpy(a_d,a,size_a,cudaMemcpyHostToDevice);
cudaMemcpy(b_d,b,size_b,cudaMemcpyHostToDevice);



// initializes thread and threadblocks
dim3 threadPerBlock(2);
dim3 numberOfBlocks(2);

// kernel launch
mergeKernel <<<numberOfBlocks, threadPerBlock >>> (a_d, b_d, c_d,N_A,N_B);

//copy memory from device to host
cudaMemcpy(c,c_d,size,cudaMemcpyDeviceToHost);
//display result
for (int i=0;i<N_C;i++){
    std::cout<< c[i]<< std::endl;
   }


//de-allocate device memories
cudaFree(a_d);
cudaFree(b_d);
cudaFree(c_d);



//de-allocates host memories
delete[] a;
delete[] b;
delete[] c;

    return 0;
}
