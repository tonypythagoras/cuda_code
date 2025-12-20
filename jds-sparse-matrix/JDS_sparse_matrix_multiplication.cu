#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdio.h>
#include <stdlib.h>

unsigned const int N=8;
unsigned const int P = 4;
unsigned const int NUM_BLOCK=4;
unsigned const int THREAD_PER_BLOCK=4;


struct JDS {  
  int numRow;
  int *iterPr;           
  int *row;
  int *col;
  int *values;
};

// This performs a sparse matrix multiplication using JDS format
__global__ void jdsSparseMatrixKernel(JDS *jds, int *invector, int *outvector){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int row = jds->row[i];

    int sum=0;
    for( int iter = 0; iter < jds->numRow; iter++){
        int df=jds->iterPr[iter];
        int df2=jds->iterPr[iter+1];

        if(i < (df2-df)){
      int pt=jds->iterPr[iter];
      int col = jds->col[i+pt];
      int value = jds->values[i+pt];
       outvector[row] +=invector[col]*value;
        }

    
   }
    
}

// initializes data
void initArray(int *iterPtr, int *row, int *col, int *values, int *vect){
  values[0]=5;
  values[1]=1;
  values[2]=2;
  values[3]=6;
  values[4]=3;
  values[5]=7;
  values[6]=8;
  values[7]=9;



  
iterPtr[0]=0;
iterPtr[1]=4;
iterPtr[2]=7;
iterPtr[3]=8;

  col[0]=0;
  col[1]=0;
  col[2]=1;
  col[3]=3;
  col[4]=2;
  col[5]=1;
  col[6]=2;
  col[7]=3;
 
  row[0]=1;
  row[1]=0;
  row[2]=2;
  row[3]=3;
 
  vect[0]=2;
  vect[1]=1;
  vect[2]=3;
  vect[3]=1;
    
}


int main(){
size_t size= N * sizeof(int);
size_t size_row= 5 * sizeof(int);
size_t size_v= 12 * sizeof(int);
size_t size_non= 4 * sizeof(int);



size_t size_p= P * sizeof(int);
size_t coo_size= (3 * size) + 1;

int *invector_h;
int *outvector_h;
int *iterPr;
int *num;

int *row_h;
int *col_h;
int *values_h;
// allocate memories on the host
iterPr=(int*)malloc(size_p);
num=(int*)malloc(size_p);

row_h=(int*)malloc(size_p);
col_h=(int*)malloc(size);
values_h=(int*)malloc(size);
invector_h=(int*)malloc(size_p);
outvector_h=(int*)malloc(size_p);



initArray(iterPr,row_h, col_h, values_h,invector_h);
     JDS myJds;


     myJds.numRow=4;
     myJds.row=row_h;
     myJds.col=col_h;
     myJds.iterPr=iterPr;
     myJds.values=values_h;

     JDS *myJDS_D;




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


err=cudaMalloc((void**)&myJDS_D, sizeof(JDS));
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }  

//copy memory from host to device
    cudaMemcpy(myJDS_D,&myJds,sizeof(JDS),cudaMemcpyHostToDevice);

    cudaMemcpy(invector_d,invector_h,size_p,cudaMemcpyHostToDevice);

     


     
     // inilizes thread and threadblocks
     dim3 threadPerBlock(8);
     dim3 numberOfBlocks(1);

    // launches kernel for execution
     jdsSparseMatrixKernel <<< numberOfBlocks, threadPerBlock >>> (myJDS_D, invector_d, outvector_d);

    // copy memories fro device to host
     cudaMemcpy(outvector_h, outvector_d,size_p,cudaMemcpyDeviceToHost);
         

   // output display
 for (int i=0;i<P;i++){
    std::cout<< outvector_h[i]<< std::endl;
   }

// free some device memories
cudaFree(myJDS_D);
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