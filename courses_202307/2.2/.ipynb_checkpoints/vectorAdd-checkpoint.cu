#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void __global__ add(const double *x, const double *y, double *z, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        z[n] = x[n] + y[n];
    }
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - 3) > (1.0e-10))
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Errors" : "Pass");
}


int main(void)
{
    const int N = 1000;
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
    printf("%f\n",c_z[1]);
    

    // 释放内存空间
    free(c_x);
    free(c_y);
    free(c_z);
    cudaFree(g_x);
    cudaFree(g_y);
    cudaFree(g_z);
    return 0;
}