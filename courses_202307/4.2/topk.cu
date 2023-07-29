#include <stdio.h>
/*
数组最大值排序(重复的不算)
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
#define TOPK 40    // 最大值前20

__managed__ int source[N];
__managed__ int result_gpu[TOPK]={0};
/*
cuda 并行思想，原子操作 串行的，有相违背

不采用原子操作，
将每一个block内最大值前二十个保存作比较
*/
__managed__ int result_block[GRID_SIZE*TOPK];

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

__global__ void sum_gpu(int* a,int* b,int length,int topk)
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


void topkByCPU(int* a,int *b,int length,int topk){
    // 冒泡排序法
    int t1;
    for(int i=0,I=0;i<topk;i++){
        for(int j=length-1;j>I;j--){
            
            if(a[j]>a[j-1]){
                t1=a[j-1];
                a[j-1]=a[j];
                a[j]=t1;
            }
        }
        if(i&&b[i-1]==a[I]){
            i--;
            I++;
            continue;;
        }
        b[i]=a[I];
        I++;
        
    }


}

int main()
{
    int result_cpu[TOPK];
    for(int i=0;i<TOPK;i++){
        result_gpu[i]=INT_MIN;
        result_cpu[i]=INT_MIN;
    }
    // 初始化
    for(int i=0;i<N;i++){
        source[i]=rand(); // 仅在10以内，避免大数处理
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
        // result_gpu[0]=0;
        sum_gpu<<<GRID_SIZE,BLOCK_SIZE>>>(source,result_block,N,TOPK);
        sum_gpu<<<1,BLOCK_SIZE>>>(result_block,result_gpu,TOPK*GRID_SIZE,TOPK);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    topkByCPU(source,result_cpu,N,TOPK);
    
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);
    float time_cpu,time_gpu;
    cudaEventElapsedTime(&time_cpu,stop_gpu,stop_cpu);
    cudaEventElapsedTime(&time_gpu,start,stop_gpu);
    printf("cpu_time:%.2f gpu_time:%.2f\n",time_cpu,time_gpu/20);

    // 验证
    int ok = 1;
    for(int i=0;i<TOPK;i++) {
        if(result_gpu[i]!=result_cpu[i]){
            ok = 0;
            break;
        }
    }
    printf(ok?"Pass!!\n":"Error!!\n");
    // printf("\n%d ,%d",result_cpu,result_gpu[0]);
    for(int i=0;i<TOPK;i++) printf("%d ",result_cpu[i]);
    printf("\n");
    for(int i=0;i<TOPK;i++) printf("%d ",result_gpu[i]);
    
    return 0;
}