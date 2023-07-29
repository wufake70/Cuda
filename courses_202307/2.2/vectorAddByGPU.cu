#include <math.h>
#include <stdlib.h>
#include <stdio.h>
/*
实现一亿次向量加法，GPU并行计算
数组1 元素全为1，1亿个
数组2 全为0,1亿个
数组3 1亿个，
数组1 数组2 对应相加，对应保存在数组3 中

*/
void __global__ add(const double *x, const double *y, double *z, const int N)
{
    // 每个线程绑定一个任务
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if(n<N){
        z[n]=x[n]+y[n];
    }
}
// 检查计算的精度是否出错
void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - 3) > (1.0e-10))
        {
            has_error = true;
        }
        if(has_error) printf("%d %f\n",n,z[n]);
    }
    printf("%s\n", has_error ? "Errors" : "Pass");
}


int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *c_x = (double*) malloc(M);
    double *c_y = (double*) malloc(M);
    double *c_z = (double*) malloc(M);
    
    // 申请gpu内存空间
    double *g_x,*g_y,*g_z;
    cudaMalloc((void **)&g_x,M),
    cudaMalloc((void **)&g_y,M),
    cudaMalloc((void **)&g_z,M);

    // 对x，y数组初始化
    for (int n = 0; n < N; ++n)
    {
        c_x[n] = 1;
        c_y[n] = 2;
    }
    
    // 将cup内存值复制到gpu
    cudaMemcpy(g_x,c_x,M,cudaMemcpyHostToDevice),
    cudaMemcpy(g_y,c_y,M,cudaMemcpyHostToDevice);

    int block_size=128;
    int grid_size=(N+block_size-1)/block_size;
    add<<<grid_size,block_size>>>(g_x, g_y, g_z, N);
    //将gpu计算的值复制到cpu内存
    cudaMemcpy(c_z,g_z,M,cudaMemcpyDeviceToHost);
    check(c_z, N);
    printf("%f\n",c_z[185]);
    

    // 释放内存空间
    free(c_x);
    free(c_y);
    free(c_z);
    cudaFree(g_x);
    cudaFree(g_y);
    cudaFree(g_z);
    return 0;
}