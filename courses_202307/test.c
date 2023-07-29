#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define M 10
#define N 10

int a[10][10];

int main()
{
    // srand(time(NULL));
    
    // printf("%d",rand());
    // return 0;
    for(int y=0; y<M; ++y)
    {
        for(int x=0; x<N; ++x)
        {
            a[y][x] = rand()%1024;
            printf("%d",a[y][x]);
            printf((N%(x+1))==0?"\n":" 0");


        }
    }

}