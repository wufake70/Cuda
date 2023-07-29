#include <stdio.h>
__managed__ int i=0;
__global__ void test(int* a){
    //i++;
    atomicAdd(&i, 1);
}
int main()
{
    dim3 grid(7,7);
    dim3 block(16,16);
    test<<<grid,block>>>(&i);
    cudaDeviceSynchronize();
    printf("%d\n",i);
    

}