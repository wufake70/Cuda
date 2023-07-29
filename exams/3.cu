/*

3.利用GPU完成原地矩阵(3000*3000)转置操作:
要求:输入矩阵和输出矩阵在同一存储空间内, 不得另外申请空间.

*/

#include <stdio.h>
#include <stdlib.h>
#include "error.cuh"

#define TILE_DIM 32   //Don't ask me why I don't set these two values to one
#define BLOCK_SIZE 32
#define N 3000

__managed__ int input_M[N * N];      //input matrix & GPU result
int cpu_result[N * N];   //CPU result
//in-place matrix transpose

__global__ void ip_transpose(int* a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N&&i<j) {
        /*
        i<j,保证只有对角线下部分的线程能进行转换(也可以i>j),
        对角线上的线程对结果没影响，但影响整体速率，可直接优化掉
        1 2 3  第一列可进行交换的线程，对应的元素 4 7
        4 5 6  i=0,j=1
        7 8 9  i=0,j=2

        1 4 7
        2 5 8
        3 6 9
        */
        int tmp = a[i * N + j];
        a[i * N + j] = a[j * N + i];
        a[j * N + i] = tmp;
    }
}




void cpu_transpose(int* A, int* B)
{
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < N; i++)
        {
            B[i * N + j] = A[j * N + i];
        }
    }
}

int main(int argc, char const* argv[])
{

    cudaEvent_t start, stop_gpu;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop_gpu));


    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            input_M[i * N + j] = rand() % 1000;
        }
    }
    cpu_transpose(input_M, cpu_result);

    CHECK(cudaEventRecord(start));
    unsigned int grid_rows = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    unsigned int grid_cols = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    ip_transpose <<<dimGrid, dimBlock >>>(input_M);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));

    float elapsed_time_gpu;
    CHECK(cudaEventElapsedTime(&elapsed_time_gpu, start, stop_gpu));
    printf("Time_GPU = %g ms.\n", elapsed_time_gpu);

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop_gpu));

    int ok = 1;
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (fabs(input_M[i * N + j] - cpu_result[i * N + j]) > (1.0e-10))
            {
                ok = 0;
            }
        }
    }


    if (ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }
   

    return 0;
}