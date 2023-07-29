#include <stdio.h>

/*
要求，输入一个1000x1000的方阵：判定如果x,y坐标都为偶数的元素，则对该元素做平方，否则该元素值减一。

*/
#define N 10
#define BLOCK_SIZE 16

__managed__ int a[N*N];
__managed__ int b[N*N];


__global__ void test(int* a,int* b,int len){
    // 获取全局坐标(x,y),化一维  y*(一行多少元素)+x
    int x=threadIdx.x+blockDim.x*blockIdx.x;
    int y=threadIdx.y+blockDim.y*blockIdx.y; 
    if(x<len&&y<len){
    // printf("%d %d\n",x,y);
        // if(x%2==0&&y%2==0){
        //     b[x*len+y]=a[x*len+y]*a[x*len+y];
        // }else{
        //     b[ x* len + y] = a[x * len + y] - 1;
        // }

        if(a[x*len+y]<100){
            b[x*len+y]=a[x*len+y]*a[x*len+y];
        }else{
            b[ x* len + y] = a[x * len + y] - 1;
        }
    }
    

}

int main()
{
    
    for(int i=0;i<N*N;i++){
        a[i]=rand()%10;
    }

    int grid_x=(N+BLOCK_SIZE-1)/BLOCK_SIZE;
    int grid_y=(N+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 Dimgrid(grid_x,grid_y);
    dim3 Dimblock(BLOCK_SIZE,BLOCK_SIZE);
    test<<<Dimgrid,Dimblock>>>(a,b,N);
    cudaDeviceSynchronize();
    for(int i=0;i<N*N;i++) printf("%d %s",a[i],(i+1)%N==0?"\n":"");
    printf("\n");
    for(int i=0;i<N*N;i++) printf("%d %s",b[i],(i+1)%N==0?"\n":"");
    

    


    return 0;
}