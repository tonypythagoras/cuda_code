#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include <stdio.h>
#include <stdlib.h>

unsigned const int N=8;
unsigned const int P = 4;
unsigned const int NUM_BLOCK=4;
unsigned const int THREAD_PER_BLOCK=4;


struct GRAPH {  
  int numVertex;
  int *rowPtr;           
  int *dest;
};

// This performs a bfs search using CSR format
//It starts from a particular vertext of choice and calculates the distance of other reachable vertices,
// and put the values (the number of hops) inside levels array.
// This one uses botom up aproach. 
__global__ void bsfCSRKernelBottomUp(GRAPH *graph, int *levels, int *again, int currentLevel){
   int vertex = blockIdx.x * blockDim.x + threadIdx.x;

   if(vertex < graph->numVertex){

    if(levels[vertex] == -1){

        for(int i=graph->rowPtr[vertex]; i < graph->rowPtr[vertex + 1]; i++){

           int neighbor=graph->dest[i];
           if(levels[neighbor]==currentLevel-1){
            levels[vertex]=currentLevel;
            *again=1;
            break;
           }    

        }
    }
   }
   
    
}



// This performs a bfs search using CSR format
//It starts from a particular vertext of choice and calculates the distance of other reachable vertices,
// and put the values (the number of hops) inside levels array.
//This uses top down approach
__global__ void bsfCSRKernelTopDown(GRAPH *graph, int *levels, int *again, int currentLevel){
   int vertex = blockIdx.x * blockDim.x + threadIdx.x;

   if(vertex < graph->numVertex){

    if(levels[vertex] == currentLevel-1){

        for(int i=graph->rowPtr[vertex]; i < graph->rowPtr[vertex + 1]; i++){

           int neighbor=graph->dest[i];
           if(levels[neighbor]==-1){
            levels[neighbor]=currentLevel;
            *again=1;
           }    

        }
    }
   }
   
    
}

// initializes data
void initArray(int *rowPtr, int *dest, int *level){

rowPtr[0]=0;
rowPtr[1]=2;
rowPtr[2]=5;
rowPtr[3]=7;
rowPtr[4]=8;


  dest[0]=1;
  dest[1]=2;
  dest[2]=0;
  dest[3]=2;
  dest[4]=3;
  dest[5]=0;
  dest[6]=1;
  dest[7]=1;

  level[0]=0;
  level[1]=-1;
  level[2]=-1;
  level[3]=-1;
    
}


int main(){
size_t size= N * sizeof(int);
size_t size_row= 5 * sizeof(int);
size_t size_l= 4 * sizeof(int);



int *rowPtr;

int *dest;
int *levels;
// allocate memories on the host
rowPtr=(int*)malloc(size_row);
dest=(int*)malloc(size);
levels=(int*)malloc(size_l);



initArray(rowPtr,dest,levels);
     GRAPH graph;

     graph.numVertex=4;
     graph.dest=dest;
     graph.rowPtr=rowPtr;

     GRAPH *myGRAPH_D;

int *level_d;
int *again_d;


cudaError_t err;
// allocates memories on the device

err = cudaMalloc((void**)&again_d,sizeof(int));
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }



err = cudaMalloc((void**)&level_d,size_l);
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }


err=cudaMalloc((void**)&myGRAPH_D, sizeof(GRAPH));
 if(err !=cudaSuccess){
               fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); 
       }  

//copy memory from host to device
    cudaMemcpy(myGRAPH_D,&graph,sizeof(GRAPH),cudaMemcpyHostToDevice);

    cudaMemcpy(level_d,levels,size_l,cudaMemcpyHostToDevice);

     


     
     // inilizes thread and threadblocks
     dim3 threadPerBlock(4);
     dim3 numberOfBlocks(1);

    // launches kernel for execution
    // starting from a particular vertext, it searches for its neighbors and assign values to them.
     int again_h=1;
     int currentLevel=1;
     while(again_h==1){
        again_h=0;
        cudaMemcpy(again_d,&again_h,sizeof(int),cudaMemcpyHostToDevice);
        if(currentLevel==1){
        bsfCSRKernelTopDown <<< numberOfBlocks, threadPerBlock >>> (myGRAPH_D, level_d, again_d, currentLevel);
        }else{
         bsfCSRKernelBottomUp<<< numberOfBlocks, threadPerBlock >>> (myGRAPH_D, level_d, again_d, currentLevel); 
        }
        
        cudaMemcpy(&again_h,again_d,sizeof(int),cudaMemcpyDeviceToHost);
        currentLevel=currentLevel+1;

     }
    // copy memories fro device to host
     cudaMemcpy(levels,level_d,size_l,cudaMemcpyDeviceToHost);
         

   // output display
 for (int i=0;i<4;i++){
    std::cout<< levels[i]<< std::endl;
   }

// free some device memories
cudaFree(myGRAPH_D);
cudaFree(level_d);
cudaFree(again_d);



// free some host memories
delete[] levels;
delete[] rowPtr;
delete[] dest;

    return 0;
}