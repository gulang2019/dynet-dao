#ifndef CUDA_H 
#define CUDA_H 

void set_tensor_value(void * start, size_t size, int value); 
bool all_equal(void * start, size_t size, int value);
void print_tensor_value(void * start, size_t size, int n);
#endif 