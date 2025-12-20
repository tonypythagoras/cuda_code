#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdio.h>
#include <stdlib.h>

unsigned const int N=8;
unsigned const int P = 4;
unsigned const int NUM_BLOCK=4;
unsigned const int THREAD_PER_BLOCK=4;


struct CSR {  
  int numRow;           
  int *rowPtr;
  int *col;
  int *values;
};




// This performs a sparse matrix multiplication using csr format
__global__ void csrSparseMatrixKernel(CSR *csr, int *invector, int *outvector){
   int row = blockIdx.x * blockDim.x + threadIdx.x;
   if(row < (csr->numRow)){
    int sum=0;
    for( int i = csr->rowPtr[row]; i< csr->rowPtr[row+1]; i++){
    int col = csr->col[i];
    int value = csr->values[i];
     sum +=invector[col]*value;
   }
   outvector[row]=sum;
   }
      

}






// initializes data
void initArray(int *rowPtr, int *col, int *values, int *vect){
  values[0]=1;
  values[1]=7;
  values[2]=5;
  values[3]=3;
  values[4]=9;
  values[5]=2;
  values[6]=8;
  values[7]=6;

  col[0]=0;
  col[1]=1;
  col[2]=0;
  col[3]=2;
  col[4]=3;
  col[5]=1;
  col[6]=2;
  col[7]=3;


  rowPtr[0]=0;
  rowPtr[1]=2;
  rowPtr[2]=5;
  rowPtr[3]=7;
  rowPtr[4]=8;
 
  vect[0]=2;
  vect[1]=1;
  vect[2]=3;
  vect[3]=1;





    
}


int main(){
size_t size= N * sizeof(int);
size_t size_row= 5 * sizeof(int);

size_t size_p= P * sizeof(int);
size_t coo_size= (3 * size) + 1;

int *invector_h;
int *outvector_h;
int *row_h;
int *col_h;
int *values_h;
// allocate memories on the host
row_h=(int*)malloc(size_row);
col_h=(int*)malloc(size);
values_h=(int*)malloc(size);
invector_h=(int*)malloc(size_p);
outvector_h=(int*)malloc(size_p);



initArray(row_h, col_h, values_h,invector_h);
     CSR myCsr;


     myCsr.numRow=4;
     myCsr.rowPtr=row_h;
     myCsr.col=col_h;
     myCsr.values=values_h;

     CSR *myCSR_D;




int *invector_d;
int *outvector_d;




cudaError_t err;
// allocates memories on the device

err = cudaMalloc((void**)&invector_d,size_p);
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }

err = cudaMalloc((void**)&outvector_d,size_p);
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }


err=cudaMalloc((void**)&myCSR_D, sizeof(CSR));
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }


  

//copy memory from host to device
    cudaMemcpy(myCSR_D,&myCsr,sizeof(CSR),cudaMemcpyHostToDevice);

    cudaMemcpy(invector_d,invector_h,size_p,cudaMemcpyHostToDevice);

     


     
     // inilizes thread and threadblocks
     dim3 threadPerBlock(8);
     dim3 numberOfBlocks(1);

    // launches kernel for execution
     csrSparseMatrixKernel <<< numberOfBlocks, threadPerBlock >>> (myCSR_D, invector_d, outvector_d);

    // copy memories fro device to host
     cudaMemcpy(outvector_h, outvector_d,size_p,cudaMemcpyDeviceToHost);
         

   // output display
 for (int i=0;i<P;i++){
    std::cout<< outvector_h[i]<< std::endl;
   }

// free some device memories
cudaFree(myCSR_D);
cudaFree(outvector_d);
cudaFree(invector_d);



// free some host memories
delete[] outvector_h;
delete[] invector_h;
delete[] col_h;
delete[] row_h;
delete[] values_h;



    return 0;
}