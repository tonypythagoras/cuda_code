#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdio.h>
#include <stdlib.h>

unsigned const int N=8;
unsigned const int P = 4;
unsigned const int NUM_BLOCK=4;
unsigned const int THREAD_PER_BLOCK=4;


struct COO {  
  int len;           
  int *row;
  int *col;
  int *values;
};




// This performs a sparse matrix multiplication using coo(cordinate format)
__global__ void cooSparseMatrixKernel(COO *coo, int *invector, int *outvector){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int toAdd=0;
   if(i < (coo->len)){
    int row = coo->row[i];
    int col = coo->col[i];
    int value = coo->values[i];
     toAdd=invector[col]*value;
     atomicAdd(&outvector[row], toAdd);
   }
      

}






// initializes data
void initArray(int *row, int *col, int *values, int *vect){
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


  row[0]=0;
  row[1]=0;
  row[2]=1;
  row[3]=1;
  row[4]=1;
  row[5]=2;
  row[6]=2;
  row[7]=3;

  vect[0]=2;
  vect[1]=1;
  vect[2]=3;
  vect[3]=1;





    
}


int main(){
size_t size= N * sizeof(int);
size_t size_p= P * sizeof(int);
size_t coo_size= (3 * size) + 1;

int *invector_h;
int *outvector_h;
int *row_h;
int *col_h;
int *values_h;
// allocate memories on the host
row_h=(int*)malloc(size);
col_h=(int*)malloc(size);
values_h=(int*)malloc(size);
invector_h=(int*)malloc(size_p);
outvector_h=(int*)malloc(size_p);



initArray(row_h, col_h, values_h,invector_h);
     COO myCOO;


     myCOO.len=8;
     myCOO.row=row_h;
     myCOO.col=col_h;
     myCOO.values=values_h;

     COO *myCOO_D;




int *invector_d;
int *outvector_d;
int *row_d;
int *col_d;
int *values_d;




cudaError_t err;
// allocates memories on the device

err=cudaMalloc((void**)&myCOO_D, sizeof(COO));
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }


err=cudaMalloc((void**)&row_d,size);
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }


err=cudaMalloc((void**)&row_d,size);
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }

err=cudaMalloc((void**)&col_d,size);
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }
err = cudaMalloc((void**)&values_d,size);
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }

       err = cudaMalloc((void**)&invector_d,size_p);
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }

err = cudaMalloc((void**)&outvector_d,size_p);
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }


//copy memory from host to device
    cudaMemcpy(myCOO_D,&myCOO,sizeof(COO),cudaMemcpyHostToDevice);

    cudaMemcpy(values_d,values_h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(col_d,col_h,size,cudaMemcpyHostToDevice);
    cudaMemcpy(row_d,row_h,size,cudaMemcpyHostToDevice);

    cudaMemcpy(invector_d,invector_h,size_p,cudaMemcpyHostToDevice);

     


     
     // inilizes thread and threadblocks
     dim3 threadPerBlock(8);
     dim3 numberOfBlocks(1);

    // launches kernel for execution
     cooSparseMatrixKernel <<< numberOfBlocks, threadPerBlock >>> (myCOO_D, invector_d, outvector_d);

    // copy memories fro device to host
     cudaMemcpy(outvector_h, outvector_d,size_p,cudaMemcpyDeviceToHost);
         

   // output display
 for (int i=0;i<P;i++){
    std::cout<< outvector_h[i]<< std::endl;
   }

// free some device memories
cudaFree(myCOO_D);
cudaFree(outvector_d);
cudaFree(invector_d);
cudaFree(col_d);
cudaFree(row_d);
cudaFree(values_d);



// free some host memories
delete[] outvector_h;
delete[] invector_h;
delete[] col_h;
delete[] row_h;
delete[] values_h;



    return 0;
}