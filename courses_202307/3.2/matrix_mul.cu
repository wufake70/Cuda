#include <stdio.h>
#include <math.h>
#include "error.cuh"
// 编码可能影响win端运行。
#define BLOCK_SIZE 16
#define X 100
#define Y 100 
#define Z 100
// 统一内存变量,去掉cudaMalloc() 和 cudaMemcpy()
__managed__ int a[X*Y];
__managed__ int b[Y*Z];
__managed__ int c[X*Z];
// __managed__ int d[X*Z];


__global__ void gpu_matrix_mult(int *a,int *b, int *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 

void cpu_matrix_mult(int *h_a, int *h_b, int *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            int tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_a[i * n + h] * h_b[h * k + j];
            }
            h_result[i * k + j] = tmp;
        }
    }
}

int main(int argc, char const *argv[])
{
    int *d = (int*)malloc(sizeof(int)*X*Z);
    /*
    int m=100;
    int n=100;
    int k=100;

    int *h_a, *h_b, *h_c, *h_cc;
    cudaMallocHost((void **) &h_a, sizeof(int)*m*n);
    cudaMallocHost((void **) &h_b, sizeof(int)*n*k);
    cudaMallocHost((void **) &h_c, sizeof(int)*m*k);
    cudaMallocHost((void **) &h_cc, sizeof(int)*m*k);
    */

    // cuda事件
    float time;
    cudaEvent_t start,stop;
    // 创建事件
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    for (int i = 0; i < X; ++i) {
        for (int j = 0; j < Y; ++j) {
            a[i * Y + j] = rand() % 1024;
        }
    }

    for (int i = 0; i < Y; ++i) {
        for (int j = 0; j < Z; ++j) {
            b[i * Z + j] = rand() % 1024;
        }
    }
    
    /*
    int *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **) &d_a, sizeof(int)*m*n));
    CHECK(cudaMalloc((void **) &d_b, sizeof(int)*n*k));
    CHECK(cudaMalloc((void **) &d_c, sizeof(int)*m*k));
    */
    
    // copy matrix A and B from host to device memory
    // CHECK(cudaMemcpy(d_a, h_a, sizeof(int)*m*n, cudaMemcpyHostToDevice));
    // CHECK(cudaMemcpy(d_b, h_b, sizeof(int)*n*k, cudaMemcpyHostToDevice));

    // 设置线程
    unsigned int grid_rows = (X + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (Z + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
   
   
    // 添加事件到当前执行流：
    CHECK(cudaEventRecord(start));
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(a, b, c, X,Y,Z);    
    //CHECK(cudaMemcpy(h_c, d_c, sizeof(int)*m*k, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&time, start, stop));
    printf("Time = %g ms.\n", time);
    // 销毁事件
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    //cudaThreadSynchronize();

    cpu_matrix_mult(a, b, d,X,Y,Z);

    int ok = 1;
    for (int i = 0; i < X; ++i)
    {
        for (int j = 0; j < Y; ++j)
        {
            if(fabs(d[i*Z + j] - c[i*Z + j])>(1.0e-10))
            {
                
                ok = 0;
            }
        }
    }

    if(ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }

    // free memory
    /*
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFreeHost(h_cc);
    */
    return 0;
}