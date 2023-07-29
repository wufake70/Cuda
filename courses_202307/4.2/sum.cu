#include <stdio.h>
/*
一维数组求和
    例如 arr[]=[ 0，1,2,3，4,5]
    cpu: for循环
    gpu: (规约法)        total_threads
        thread ,step 0: tid0: arr[0] + arr[3] --> arr[0]
                        tid1: arr[1] + arr[4] --> arr[1]
                        tid2: arr[2] + arr[5] --> arr[2]

                step 1: tid0: arr[0]+arr[1] --> arr[0]
                        tid1: arr[2]+0      --> arr[2]

                step 2: tid0: arr[0]+arr[2] --> arr[0]


*/
#define N 1000000
#define BLOCK_SIZE 256
#define GRID_SIZE 32

__managed__ int source[N];
__managed__ int result_gpu[1]={0};

__global__ void sum_gpu(int* a,int n,int* result)
{
    __shared__ int share_memory[BLOCK_SIZE];
    int tmp=0;
    // grid_loop
    // 线程数不够 小于 N，
    // 一个线程处理多个数据相加在存储share memory
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=BLOCK_SIZE*GRID_SIZE){
        tmp+=a[i];
    }
    // 载入 shared memory并等待完成
    share_memory[threadIdx.x]=tmp;
    __syncthreads();

    // 规约法 将share memory中元素两辆对应相加存储前一个地址，
    // 直到total_threads为0即可
    for(int total_threads=BLOCK_SIZE/2;total_threads>=1;total_threads/=2){
        
        // 大于total_threads的线程停止(防止数组越界)
        if(threadIdx.x<total_threads){
            share_memory[threadIdx.x]=share_memory[threadIdx.x]+share_memory[threadIdx.x+total_threads];
        }
        __syncthreads();
    }

    // 不同线程块的share memory不能访问
    if(blockIdx.x*blockDim.x<n){
        if(threadIdx.x==0){
            // result[0]+=share_memory[0]
            // 原子操作
            atomicAdd(result,share_memory[0]);

        }
    }
}


int main()
{
    int result_cpu=0;
    // 初始化
    for(int i=0;i<N;i++){
        source[i]=rand()%10; // 仅在10以内，避免大数处理
    }

    // 事件
    cudaEvent_t start,stop_cpu,stop_gpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&stop_gpu);

    // 计时
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    // 重复20次
    for(int i=0;i<20;i++){
        result_gpu[0]=0;
        sum_gpu<<<GRID_SIZE,BLOCK_SIZE>>>(source,N,result_gpu);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    for(int i=0;i<N;i++){
        result_cpu +=source[i];
    }
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    float time_cpu,time_gpu;
    cudaEventElapsedTime(&time_cpu,stop_gpu,stop_cpu);
    cudaEventElapsedTime(&time_gpu,start,stop_gpu);
    printf("cpu_time:%.2f gpu_time:%.2f\n",time_cpu,time_gpu/20);
    printf(result_gpu[0]==result_cpu?"Pass!!":"Error!!");
    printf("\n%d ,%d",result_cpu,result_gpu[0]);
    return 0;
}