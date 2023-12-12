#include <DAO/globals.h>

namespace DAO {
DAO_API int verbose = 0;
DAO_API bool async_enabled = false; 
DAO_API bool offload_enabled = false; 
DAO_API size_t gpu_mem_limit = 0; 
DAO_API size_t cpu_mem_limit = 0;
DAO_API size_t gpu_mem = GPU_MEM_SIZE; 
DAO_API size_t cpu_mem = CPU_MEM_SIZE;
DAO_API cudaStream_t default_stream = cudaStreamDefault; 
} //DAO