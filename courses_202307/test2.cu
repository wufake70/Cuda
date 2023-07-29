/*
实现求包含1000000个元素的数组的最大的两个值
*/

#include <stdio.h>
#define N 1000000
#define BLOCK_SIZE 256
#define GRID_SIZE 32
#define TOPK 2

__managed__ int a[N];
__managed__ int result_gpu[TOPK]={0};
__managed__ int result_block[TOPK*GRID_SIZE];

// 排序算法
__device__ __host__ void sort(int* a,int length,int data){
    for(int i=0;i<length;i++){
        if(a[i]==data){
            return;
        }
    }
    if(a[length-1]>data){
        return;
    }
    for(int i=length-2;i>=0;i--){
        if(data>a[i]){

            a[i+1] = a[i];
        }else{
            a[i+1]=data;
            return;
        }
    }
    a[0]=data;
}

__global__ void topkByGPU(int* a,int* b,int length,int topk)
{
    // 每个线程维护一个数组
    // __shared__ int share_memory[BLOCK_SIZE*topk];
    // int top_arr[topk];
    __shared__ int share_memory[BLOCK_SIZE*TOPK];
    int top_arr[TOPK];
    for(int i=0;i<topk;i++){
        top_arr[i]=INT_MIN;
    }

    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<length;i+=blockDim.x*gridDim.x){
        sort(top_arr,topk,a[i]);
    }
    for(int i=0;i<topk;i++){
        share_memory[topk*threadIdx.x+i]=top_arr[i];
    }
    __syncthreads();
    for(int i=BLOCK_SIZE/2;i>=1;i/=2){
        if(threadIdx.x<i){
            for(int j=0;j<topk;j++){
                sort(top_arr,topk,share_memory[topk*(threadIdx.x+i)+j]);
            }
        }
        __syncthreads();
        if(threadIdx.x<i){
            for(int j=0;j<topk;j++){
                
                share_memory[topk*threadIdx.x+j] = top_arr[j];
            }
        }
        __syncthreads();

    }
    if(blockIdx.x*blockDim.x<length){
        if(threadIdx.x == 0){
            for(int i=0;i<topk;i++){
                b[TOPK*blockIdx.x+i]=share_memory[i];
            }
        }
    }
}


int main()
{
    for(int i=0;i<N;i++){
        a[i]=rand()%20;
    }
    topkByGPU<<<GRID_SIZE,BLOCK_SIZE>>>(a,result_block,N,TOPK);
    topkByGPU<<<1,BLOCK_SIZE>>>(result_block,result_gpu,TOPK*GRID_SIZE,TOPK);
    cudaDeviceSynchronize();
    for(int i=0;i<TOPK;i++) printf("%d ",result_gpu[i]);
    

    

    return 0;
}