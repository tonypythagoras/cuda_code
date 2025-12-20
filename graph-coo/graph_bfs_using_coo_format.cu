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
  int numEdge;
  int *source;           
  int *dest;
};

// This performs a bfs search using COO format
//It starts from a particular vertext of choice and calculates the distance of other reachable vertices,
// and put the values (the number of hops) inside levels array.
__global__ void bsfCOOKernel(GRAPH *graph, int *levels, int *again, int currentLevel){
   int i = blockIdx.x * blockDim.x + threadIdx.x;

   if(i < graph->numEdge){

    int srcVertex = graph->source[i];
    int neighbor = graph->dest[i];

    if(levels[srcVertex] == currentLevel-1){
        if(levels[neighbor] == -1){
            levels[neighbor]=currentLevel;
            *again=1;
        }
    } 
}
}

// initializes data
void initArray(int *src, int *dest, int *level){

  src[0]=0;
  src[1]=0;
  src[2]=1;
  src[3]=1;
  src[4]=1;
  src[5]=2;
  src[6]=2;
  src[7]=3;



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
size_t size_l= 4 * sizeof(int);



int *src;

int *dest;
int *levels;
// allocate memories on the host
src=(int*)malloc(size);
dest=(int*)malloc(size);
levels=(int*)malloc(size_l);



initArray(src,dest,levels);
     GRAPH graph;

     graph.numEdge=8;
     graph.dest=dest;
     graph.source=src;

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
     dim3 threadPerBlock(8);
     dim3 numberOfBlocks(1);

    // launches kernel for execution
    // starting from a particular vertext, it searches for its neighbors and assign values to them.
     int again_h=1;
     int currentLevel=1;
     while(again_h==1){
        again_h=0;
        cudaMemcpy(again_d,&again_h,sizeof(int),cudaMemcpyHostToDevice);
        bsfCOOKernel <<< numberOfBlocks, threadPerBlock >>> (myGRAPH_D, level_d, again_d, currentLevel);
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
delete[] dest;
delete[] src;

    return 0;
}