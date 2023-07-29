#include <stdio.h>
#include <stdlib.h>
#include <time.h>


__global__ void matrixTranspose(int *in, int *out, int width)
{
    int y=threadIdx.x+blockIdx.x*blockDim.x,
        x=threadIdx.y+blockIdx.y*blockDim.y;
        
        if(x<width&&y<width){
        
        out[y*width+x]=in[x*width+y];
        }
        
}



void cpu_matrix_transpose(int *in, int *out, int width)
{
    for(int y = 0; y < width; y++)
    {
        for(int x = 0; x < width; x++)
        {
            out[x * width + y] = in[y * width + x];
        }
    }
}

int main()
{
    int width=1000;
    int *c_arr1,
        *c_arr2,
        *c_arr3;
    c_arr1=(int*)malloc(sizeof(int)*width*width),
    c_arr2=(int*)malloc(sizeof(int)*width*width),
    c_arr3=(int*)malloc(sizeof(int)*width*width);
    for(int i=0;i<width;i++){
        for(int j=0;j<width;j++){
            c_arr1[i*width+j]=rand();
        }
    }
    // 申请GPU内存空间
    int *g_arr1,*g_arr2;
    cudaMalloc((void**)&g_arr1,sizeof(int)*width*width),
    cudaMalloc((void**)&g_arr2,sizeof(int)*width*width);
    cudaMemcpy(g_arr1,c_arr1,sizeof(int)*width*width,cudaMemcpyHostToDevice);
    
    //设置线程
    int block_size=16;
    int gridrows=(width+block_size-1)/block_size,
        gridcols=(width+block_size-1)/block_size;
    dim3 gridDim(gridcols,gridrows);
    dim3 gridBlock(block_size,block_size);
    
    // 计时器
    clock_t t1,t2;
    
    // GPU计算
    printf("开始GPU计算\n");
    t1=clock();
    matrixTranspose<<<gridDim,gridBlock>>>(g_arr1,g_arr2,width);
    cudaMemcpy(c_arr3,g_arr2,sizeof(int)*width*width,cudaMemcpyDeviceToHost);
    t2=clock();
    printf("GPU计算共消耗 %d\n",(int)(t2-t1));
    
    // cpu 计算
    printf("开始CPU计算\n");
    t1=clock();
    cpu_matrix_transpose(c_arr1,c_arr2,width);
    t2=clock();
    printf("CPU计算共消耗 %d\n",(int)(t2-t1));
    
    // 验算
    int ok=1;
    for(int i=0;i<width*width;i++){
        if(c_arr2[i]!=c_arr3[i]){
            ok=0;
            printf("%d %d\n",c_arr2[i],i);
            break;
        }
    }
    printf(ok?"Pass!!\n":"Error!!\n");
    
    return 0;
}