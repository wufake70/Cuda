#include <stdio.h>
#include <math.h>

// a[m][n] * b[n][k] = c[m][k]
// 
//                         b00 b01 b02 b03
//                         b10 b11 b12 b13
//                         b20 b21 b22 b23
//                         b30 b31 b32 b33
//
// a00 a01 a02 a03         c00 c01 c02 c03
// a10 a11 a12 a13         c10 c11 c12 c13     block(1, 0) -> shared memory
// a20 a21 a22 a23         c20 c21 c22 c23     c20 c21
// a30 a31 a32 a33         c30 c31 c32 c33     c30 c31
//
//                              b00 b01->  sub_b_step_0
//                              b10 b11
//
//                              b20 b21->  sub_b_step_1
//                              b30 b31
// sub_a_step_0 sub_a_step_1    sub_c
// a20 a21      a22 a23         c20 c21
// a30 a31      a32 a33         c30 c31
//
// sub_c = sub_a_step_0 * sub_b_step_0 + sub_a_step_1 * sub_b_step_1;
//
// for(int step =0; step < N/block_size; step++ )
//      load sub_a_step to shared memory;
//      load sub_b_step to shared memory;
//      tmp += sub_a_step_on_sharedmemory * sub_b_step_on_sharedmemory;
// sub_c = tmp;
//
// cudaMalloc -> global memory
// data global memory(慢) -> shared memory(快)
// threads shared memory -> register
// shared memory SM(stream multi-processor，流多处理器) 
// same block same shared memory
// 同一个线程块共享一个共享内存

// c21 = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31
// a00 a01 a02 a03 a10 a11 a12 a13 a20 a21 a22 a23 a30 a31 a32 a33
// 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
// b00 b01 b02 b03 b10 b11 b12 b13 b20 b21 b22 b23 b30 b31 b32 b33

#define M 100
#define N 100
#define K 100
#define BLOCK_SIZE 16
__managed__ int a[M*N];
__managed__ int b[N*K];
__managed__ int c[M*K];
__managed__ int d[M*K];
__managed__ int count=0;
__global__ void matrixMulByGPU(int* a,int* b,int* c,int m,int n,int k)
{   
    // 计算线程数
    // atomicAdd(&count,1);

    // a矩阵 m行n列，b矩阵 n行k列，c矩阵 m行k列

    // 创建两个共享内存，同线程块一样维度 BLOCK_SIZE行BLOCK_SIZE列 ！！！！
    // (同一线程块的共享内存都指向一个地方)
    // 分别存储a，b矩阵
    __shared__ int share_mem1[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int share_mem2[BLOCK_SIZE][BLOCK_SIZE];
    // 获取当前线程在grid的坐标
    int thread_x=threadIdx.x+blockDim.x*blockIdx.x; 
    int thread_y=threadIdx.y+blockDim.y*blockIdx.y; 

    // 将元素载入shared memory
    // shared memory(空间有限),必须将两个矩阵同时分割出 子矩阵 存入shared memory中，
    // 对应相乘在累加
    int tmp=0;
    int idx=0;
    // 子矩阵(方阵)的长度为 BLOCK_SIZE
    for(int step=0;step<=n/BLOCK_SIZE;step++){
        // atomicAdd(&count,1);
        // a矩阵分块 载入shared memory
        // step_x,step_y表示分块后元素的全局坐标
        // a矩阵按行读取，y轴不变 step_y=thread_y
        int step_x=step*BLOCK_SIZE+threadIdx.x;
        int step_y=thread_y;
        // 逻辑二维数组 转为 物理一维数组(下标转换)
        idx=step_y*n+step_x;
        if(step_x>=n||step_y>=m){   // 越界赋值0
            share_mem1[threadIdx.y][threadIdx.x]=0;

        }else{
            share_mem1[threadIdx.y][threadIdx.x]=a[idx];
        }

        // b矩阵操作
        step_x=thread_x;
        step_y=step*BLOCK_SIZE+threadIdx.y;
        idx=step_y*k+step_x;
        if(step_x>=k||step_y>=n){   // 越界赋值0
            share_mem2[threadIdx.y][threadIdx.x]=0;
        }else{
            share_mem2[threadIdx.y][threadIdx.x]=b[idx];
        }

        // 同步线程块，即等待 数据载入shared memory完成
        // 它的作用是确保线程块中的每个线程都执行完 __syncthreads() 前面的语句后，才会执行下一条语句。
        __syncthreads();
        // 
        for(int i=0;i<BLOCK_SIZE;i++){
            // a的行，b的列相乘累加
            tmp+=share_mem1[threadIdx.y][i]*share_mem2[i][threadIdx.x];
        }
        __syncthreads();
    }
    // 防止越界
    if(thread_x<k&&thread_y<m){
        c[thread_y*k+thread_x]=tmp;
    }

}

void cpu_matrix(int* a, int* b, int* c, int m, int n, int k)
{
    for( int y = 0; y < m; y++)
    {
        for(int x = 0; x < k; x++)
        {
            int tmp = 0;
            for(int step =0; step < n; step++)
            {
                tmp += a[y*n + step] * b[step*k + x];
            }
            c[y * k + x] = tmp;
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
    for(int y=0; y<M; ++y)
    {
        for(int x=0; x<N; ++x)
        {
            a[y * N + x] = rand()%1024;
        }
    }
    for(int y=0; y<N; ++y)
    {
        for(int x=0; x<K; ++x)
        {
            b[y*K + x] = rand()%1024;
        }
    }

    // 设置线程
    unsigned int grid_x = (K + BLOCK_SIZE -1)/BLOCK_SIZE;
    unsigned int grid_y = (M + BLOCK_SIZE -1)/BLOCK_SIZE;
    dim3 dimGrid(grid_x,grid_y);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    cudaEventRecord(start);

    matrixMulByGPU<<<dimGrid, dimBlock>>>(a, b, c, M, N, K);

    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    cpu_matrix(a, b, d, M, N, K);

    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    bool errors = false;
    float time_cpu, time_gpu;

    for(int y=0; y<M; y++)
    {
        for(int x=0; x<K; x++)
        {
            if(fabs(d[y*K + x] - c[y*K+x]) > (1.0e-10))
            {
                errors = true;
                // printf("c: %d. d: %d", c[y*K + x], d[y*K+x]);
                // break;
            }
        }
    }

    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);
    cudaEventElapsedTime(&time_gpu, start ,stop_gpu);

    printf("Result: %s\n", errors?"Error":"Pass");
    printf("CPU time: %.2f; GPU time: %.2f;\n", time_cpu, time_gpu);
    printf("共有 %d线程\n",count);


    return 0;
}