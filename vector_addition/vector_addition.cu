#include <cuda.h>
#include <stdio.h>

unsigned int cdiv(int a, int b)
{
    return (a + b - 1) / b;
}

__global__ void vecAddKernal(float *A, float *B, float *C, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}
void vecAdd(float *A, float *B, float *C, int n)
{
    float *A_d, *B_d, *C_d;

    size_t size = n * sizeof(float);

    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    const unsigned int numThreads = 256;
    unsigned int numBlocks = cdiv(n, numThreads);

    vecAddKernal<<<numBlocks, numThreads>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
int main()
{
    const int n = 1000;

    float A[n];
    float B[n];
    float C[n];

    for (int i = 0; i < n; i++)
    {
        A[i] = float(i);
        B[i] = float(i);
    }

    vecAdd(A, B, C, n);

    for (int i = 0; i < n; i += 1)
    {
        if (i > 0)
        {
            printf(", ");
        }
        if (i % 10 == 0)
        {
            printf("\n");
        }
        printf("%8.3f", C[i]);
    }
    printf("\n");
    return 0;
}