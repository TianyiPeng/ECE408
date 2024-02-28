#include <sys/time.h>
#include <stdio.h>
#include <stdint.h>

const int BLK = 1024;
uint64_t n = 1024L * 1024 * 1024;
int tt = 100;

__global__ void scalamul(int n, int *s)
{
    int idx = threadIdx.x + blockIdx.x * BLK;
    
    if (idx < n)
	//s[idx] = 1;
        s[idx] = s[idx] + 1;
    // for () {
    // }
}

int main()
{
    int *a, *s;
    cudaMalloc(&a, n * sizeof(int));
    cudaMalloc(&s, n * sizeof(int));

    struct timeval start, end;
    float kernel_time;

    gettimeofday(&start, NULL);

    for (int i = 0; i < tt; ++i) {
        scalamul<<<n/BLK, BLK>>>(n, s);
    }
    cudaDeviceSynchronize();

    gettimeofday(&end, NULL);

    kernel_time = end.tv_sec - start.tv_sec
        + (float) (end.tv_usec - start.tv_usec) / 1e6;

    printf("time = %.5f\n", kernel_time);
    printf("bandwidth = %.5f\n", n * sizeof(int) * tt / kernel_time / 1e9);


    return 0;
}
