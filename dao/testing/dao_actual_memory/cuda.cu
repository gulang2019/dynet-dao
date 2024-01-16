#include <cuda_runtime.h>

__global__ 
void set_constant(int *a, int value, size_t size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        a[i] = value;
}
#define BLOCK_SIZE 256
void set_tensor_value(void * start, size_t size, int value) {
    int block_num = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    set_constant<<<block_num, BLOCK_SIZE>>>((int *)start, value, size);    
}

bool all_equal(void * start, size_t size, int value) {
    int* a = (int*)malloc(size*sizeof(int));
    cudaMemcpy(a, start, size*sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) {
        if (a[i] != value) {
            return false;
        }
    }
    free(a);
    return true;
}

