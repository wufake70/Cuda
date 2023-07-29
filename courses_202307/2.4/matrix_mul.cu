#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
/*
矩阵乘法

*/
__global__ void matrixMulByGPU(int *arr1,int *arr2,int *arr3,int x,int y,int z)
{ 
    // 获取当前线程在所有线程的坐标，两个坐标值就相当于两层for循环的循环变量
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if( col < z && row < x) // 防止数组越界
    {
        for(int i = 0; i < y; i++) 
        {
            sum += arr1[row * y + i] * arr2[i * z + col];
        }
        arr3[row * z + col] = sum;

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

int main()
{
    // 行列数
    int x=1000,
        y=1000,
        z=1000;
    // 数组定义并初始化
    int *c_arr1=(int *)malloc(sizeof(int)*x*y),
        *c_arr2=(int *)malloc(sizeof(int)*y*z),
        *c_arr3=(int *)malloc(sizeof(int)*x*z),
        *c_arr4=(int *)malloc(sizeof(int)*x*z);
    for(int i=0;i<x;i++){
        for(int j=0;j<y;j++){
            c_arr1[i*y+j]=rand();
        }
    }
    for(int i=0;i<y;i++){
        for(int j=0;j<z;j++){
            c_arr1[i*z+j]=rand();
        }
    }
    
    
    // 申请gpu内存并赋值
    int *g_arr1,*g_arr2,*g_arr3;
    cudaMalloc((void**)&g_arr1,sizeof(int)*x*y);
    cudaMalloc((void**)&g_arr2,sizeof(int)*y*z);
    cudaMalloc((void**)&g_arr3,sizeof(int)*x*z);
    
    cudaMemcpy(g_arr1,c_arr1,sizeof(int)*x*y,cudaMemcpyHostToDevice);
    cudaMemcpy(g_arr2,c_arr2,sizeof(int)*y*z,cudaMemcpyHostToDevice);
    
    
    // 设置动态线程,
    int block_size=8;   // 线程数设置尽量小些，防止数组越界
    // 向上取整
    int grid_rows=(x+block_size-1)/block_size;
    int grid_cols=(y+block_size-1)/block_size;
    dim3 dimGrid(grid_cols,grid_rows);
    dim3 dimBlock(block_size,block_size);
    
    // 计时器
    clock_t t1,t2;
    
    // gpu并行计算
    printf("开始gpu并行计算\n");
    t1=clock();
    matrixMulByGPU<<<dimGrid,dimBlock>>>(g_arr1,g_arr2,g_arr3,x,y,z);
    // cudaMemcpy会等待 GPU线程执行完所有的任务
    cudaMemcpy(c_arr3,g_arr3,sizeof(int)*x*z,cudaMemcpyDeviceToHost);
    t2=clock();
    
    printf("gpu并行计算共耗时 %f\n",(double)(t2-t1));
    
    // cpu for循环计算
    printf("开始cpu for循环计算\n");
    t1=clock();
    cpu_matrix_mult(c_arr1,c_arr2,c_arr4,x,y,z);
    t2=clock();
    printf("cpu for循环计算共耗时 %f\n",(double)(t2-t1));
    
    // 验算
    int ok=1;
    for(int i=0;i<x;i++){
        for(int j=0;j<z;j++){
            if(c_arr3[i*z+j]-c_arr4[i*z+j]!=0){
                ok=0;
                break;
            }
        }
    }
    
    printf(ok?"Pass!!\n":"Error!!\n");

    free(c_arr1),
    free(c_arr2),
    free(c_arr3),
    free(c_arr4);
    cudaFree(g_arr1),
    cudaFree(g_arr2),
    cudaFree(g_arr3);    
    return 0;
}