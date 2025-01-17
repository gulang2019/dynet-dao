#ifndef _DAO_GLOBAL_H_
#define _DAO_GLOBAL_H_

#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <dynet/dynet.h>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

namespace DAO {

typedef dynet::Tensor* TensorUID; 
typedef size_t logical_time_t;

#define DAO_EXPORT __attribute__((__visibility__("default")))
#define DAO_HIDDEN __attribute__((__visibility__("hidden")))
#define DAO_API DAO_EXPORT

#define GPU_MEM_SIZE (128*1024*1024)
#define CPU_MEM_SIZE (128*1024*1024)
extern bool profile_enabled; 
extern int verbose; 
extern bool async_enabled; 
extern bool offload_enabled;
extern size_t gpu_mem_limit; 
extern size_t cpu_mem_limit;
extern double gpu_mem; 
extern double cpu_mem;
extern cudaStream_t default_stream;
extern int default_device_id;
extern bool debug_mode;
extern bool enabled; // whether to do the symbolic update;
extern bool use_dao; // whether DAO is enabled for memory allocation;
extern int offload_profiling; // whether to do profiling for DAO; 0, no-profile; 1 profile memory&time; 2 profile time

#define PRINT_MSG(...) do{printf(ANSI_COLOR_MAGENTA "%s:%d,%d " ANSI_COLOR_RESET, __FILE__, __LINE__,gettid()); printf(__VA_ARGS__); printf("\n");}while(0)
#define DAO_INFO(...) do{ if (DAO::verbose) { printf(ANSI_COLOR_GREEN "[DAO::INFO]:\t" ANSI_COLOR_RESET); PRINT_MSG(__VA_ARGS__); }} while(0)
#define DAO_INFO_LEVEL(level, ...) do{ if (DAO::verbose >= level) { printf(ANSI_COLOR_GREEN "[DAO::INFO]:\t" ANSI_COLOR_RESET); PRINT_MSG(__VA_ARGS__); }} while(0)
#define DAO_WARNING(...) do{ printf(ANSI_COLOR_RED "[DAO::WARNING]:\t" ANSI_COLOR_RESET); PRINT_MSG(__VA_ARGS__); } while(0)
#define DAO_COND_WARNING(cond, ...) do{ if (cond) {printf(ANSI_COLOR_RED "[DAO::WARNING]:\t" ANSI_COLOR_RESET); PRINT_MSG(__VA_ARGS__);} } while(0)
#define DAO_ERROR(...) do{ { printf(ANSI_COLOR_RED "[DAO::ERROR]:\t" ANSI_COLOR_RESET); PRINT_MSG(__VA_ARGS__); exit(1);} } while(0)
#define DAO_ASSERT(cond,...) do{ if (!(cond)) { printf(ANSI_COLOR_RED "[DAO::ASSERT]:\t" ANSI_COLOR_RESET); PRINT_MSG(__VA_ARGS__); assert(cond); }} while(0)

} // DAO 

#endif 