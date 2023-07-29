#include <stdio.h>
#include <math.h>
/*
矩阵转置(任意矩阵)

*/
// a[2][1] 第二行第三列元素

//              matrix transpose
//                                                     t56 t57 t58
//               in        b00 b01 b02 | b03 b04 b05 | b06 b07 b08  B[6][9]
//                         b10 b11 b12 | b13 b14 b15 | b16 b17 b18
//                         b20 b21 b22 | b23 b24 b25 | b26 b27 b28
//                         ------------+-------------+------------
//                         b30 b31 b32 | b33 b34 b35 | b36 b37 b38
//                         b40 b41 b41 | b43 b44 b45 | b46 b47 b48
//                         b50 b51 b52 | b53 b54 b55 | b56 b57 b58   threadIdx.x=1, threadIdx.y=2;
//                                                                   block 1, 2
//
//                         
//               out       b00 b10 b20 | b30 b40 b50
//                         b01 b11 b21 | b31 b41 b51
//                         b02 b12 b22 | b32 b42 b52
//                         ------------+------------
//                         b03 b13 b23 | b33 b43 b53
//                         b04 b14 b24 | b34 b44 b54
//                         b05 b15 b25 | b35 b45 b55
//                         ------------+------------
//                         b06 b16 b26 | b36 b46 b56
//                         b07 b17 b27 | b37 b47 b57
//                         b08 b18 b28 | b38 b48 b58                 block 2, 1
// shared memory 
// t57 read b57 from global memroy to shared memroy
// t57 read b48 from shared memory
// t57 write b48 to global memory
#define M 3000
#define N 3000
#define BLOCK_SIZE 32
__managed__ int a[M][N];
__managed__ int b[N][M];
__managed__ int c[N][M];
__managed__ int count=0;

__global__ void matrixMulByGPU(int a[M][N],int b[N][M])
{   
    // 访存合并是指在GPU编程中，多个线程对全局内存的一次访问请求（读或写）导致最少数量的数据传输。
    // 当前线程在grid中的坐标
    int thread_x=threadIdx.x+blockIdx.x*blockDim.x;
    int thread_y=threadIdx.y+blockIdx.y*blockDim.y;
    __shared__ int shared_memory[BLOCK_SIZE+1][BLOCK_SIZE+1];   // 避免bank冲突
    
    if(thread_x<N&&thread_y<M){
        // 载入shared memory中
        shared_memory[threadIdx.y][threadIdx.x]=a[thread_y][thread_x];
    }
    // 等待shared memory载入完成
    __syncthreads();
    // 写入数据
    // 只需要将block在grid的坐标切换即可，线程在block中的坐标 threadIdx不改变
    int thread_x2=threadIdx.x + blockDim.y * blockIdx.y;
    int thread_y2=threadIdx.y + blockDim.x * blockIdx.x;
    if(thread_x2<M&&thread_y2<N){
        // shared memory进行转置(即交换下标变量)
        b[thread_y2][thread_x2]=shared_memory[threadIdx.x][threadIdx.y];

    }


}

void cpu_matrix_transpose(int in[M][N], int out[N][M])
{
    for(int x=0;x<M;x++){
        for(int y=0;y<N;y++){
            out[y][x]=in[x][y];
        }

    }

}

int main()
{
    // 创建事件
    cudaEvent_t start, stop_gpu, stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&stop_gpu);


    // 数组初始化
    for(int x=0;x<M;x++){
        for(int y=0;y<N;y++){
            a[x][y]=rand()%1024;
            // printf("%d %s",a[x][y],(y+1)%N==0?"\n":"");
        }
    }

    // 设置线程
    unsigned int grid_x = (M + BLOCK_SIZE -1)/BLOCK_SIZE;
    unsigned int grid_y = (N + BLOCK_SIZE -1)/BLOCK_SIZE;
    dim3 dimGrid(grid_x,grid_y);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    cudaEventRecord(start);

    matrixMulByGPU<<<dimGrid, dimBlock>>>(a, b);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    cpu_matrix_transpose(a, c);

    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    bool errors = false;
    float time_cpu, time_gpu;

    for(int y=0; y<M; y++)
    {
        for(int x=0; x<N; x++)
        {
            if(fabs(b[y][x] - c[y][x]) > (1.0e-10))
            {
                errors = true;
                
            }
        }
    }
    // for(int x=0;x<N;x++){
    //     for(int y=0;y<M;y++){
    //         if(fabs(c[x][y]-b[x][y])>(1.0e-10)){
    //             errors=true;
    //         }
    //     }
    // }

    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);
    cudaEventElapsedTime(&time_gpu, start ,stop_gpu);

    printf("Result: %s\n", errors?"Error":"Pass");
    printf("CPU time: %.2f; GPU time: %.2f;\n", time_cpu, time_gpu);
    // printf("共有 %d线程\n",count);
    
    // for(int x=0;x<N;x++){
    //     for(int y=0;y<M;y++){
    //         printf("%d %s",c[x][y],(y+1)%M==0?"\n":"");
    //     }
    // }
    printf("\n");
    for(int x=0;x<N;x++){
        for(int y=0;y<M;y++){
            printf("%d %s",b[x][y],(y+1)%M==0?"\n":"");
        }
    }
    

    return 0;
}