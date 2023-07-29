#include <stdio.h>

__global__ void helloFromGpu(void)
{

    // const int tid=threadIdx.x;
    printf("threadx=%d blockIdx=%d blockDim=%d gridDim=%d\n",threadIdx.x,blockIdx.x,blockDim.x,gridDim.x);

}

int main()
{
    helloFromGpu<<<5,65>>>();
    cudaDeviceSynchronize();

    
    return 0;
}