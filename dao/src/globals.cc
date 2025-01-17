#include <DAO/globals.h>

namespace DAO {
DAO_API int verbose = 0;
DAO_API bool async_enabled = false; 
DAO_API bool offload_enabled = false; 
DAO_API size_t gpu_mem_limit = 0; 
DAO_API size_t cpu_mem_limit = 0;
DAO_API double gpu_mem = 128; // 128MB
DAO_API double cpu_mem = 128; // 128MB
DAO_API cudaStream_t default_stream = cudaStreamDefault; 
DAO_API int default_device_id = 0;
DAO_API bool debug_mode = false;
DAO_API bool use_dao = false;
DAO_API int offload_profiling = false;

} //DAO