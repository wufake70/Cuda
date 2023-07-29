#include <stdio.h>
/*

按照矩阵转置的公式，我们设定（按照下图所示）

输入矩阵为: A[16][16]    M=16

输出矩阵为: A’[16][16]

保证：A ‘[y][x] = A[x][y]

*/

#define M 300
#define N 300
#define BLOCK_SIZE 16
__managed__ int a[M][M];
__managed__ int b[M][M];


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
        // b[thread_y2][thread_x2]=shared_memory[threadIdx.x][threadIdx.y];

        // 直接换为a即可实现在原数组操作
        a[thread_y2][thread_x2]=shared_memory[threadIdx.x][threadIdx.y];

    }


}


int main()
{
    for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            a[i][j]=rand()%10;
            // printf("%d ",a[i][j]);

        }
        printf("\n");
    }
    // for(int i=0;i<M*M;i++) printf("%d %s",a[i],(i+1)%M==0?"\n":"");

    // 设置线程
    unsigned int grid_x = (M + BLOCK_SIZE -1)/BLOCK_SIZE;
    unsigned int grid_y = (N + BLOCK_SIZE -1)/BLOCK_SIZE;
    dim3 dimGrid(grid_x,grid_y);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

    
    matrixMulByGPU<<<dimGrid,dimBlock>>>(a,b);
    cudaDeviceSynchronize();
    printf("\n");

     for(int i=0;i<M;i++){
        for(int j=0;j<M;j++){
            printf("%d ",a[i][j]);

        }
        printf("\n");
    }



    return 0;
}