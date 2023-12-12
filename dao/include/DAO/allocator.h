#ifndef GLOBAL_MEM_ALLOCATOR_H
#define GLOBAL_MEM_ALLOCATOR_H

// #include <DAO/memory_manager.h>
#include <DAO/globals.h>
#include <DAO/generator.h>

#include <cuda_runtime.h>
#include <queue>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

namespace DAO {



struct MemBlock;
class MemStorage; 

enum tensor_status_t{
    ONCPU,
    ONGPU,
    UNINITIALIZED
};

struct SmartCudaEvent {
    cudaEvent_t event;
    std::string name;
    SmartCudaEvent(cudaStream_t stream, const std::string& name);
    cudaEvent_t get();
    ~SmartCudaEvent();
};

// A cuda event that destroys itself when it goes out of scope
struct TensorRecord {
    std::string name; 
    TensorUID tensor_id;
    tensor_status_t status;  
    size_t tensor_size;
    std::shared_ptr<MemBlock> block_ptr;
    std::queue<logical_time_t> access_pattern;
    std::shared_ptr<SmartCudaEvent> event; 
    logical_time_t last_access;
    void display(std::ostream& o, bool display_block = true) const;
};


class GlobalMemRecord {
    std::unordered_map<TensorUID, TensorRecord> tensor_record_table;
public:
    void add_tensor_access(TensorUID tensor_id, size_t size, logical_time_t init_time, bool is_last_access);
    //if no such tensor id exist, create new, otherwise, lookup and update time access
    void delete_tensor(TensorUID tensor_id);
    TensorRecord& lookup_tensor(TensorUID tensor_id);
    void display(std::ostream& o) const;
};

class Allocator{
private:
    cudaStream_t H2D_stream;
    cudaStream_t D2H_stream;
    cudaStream_t compute_stream;
    std::unique_ptr<GlobalMemRecord> global_memory_record;
    std::unique_ptr<MemStorage> cpu_manager;
    std::unique_ptr<MemStorage> gpu_manager;
    void* prepare(TensorUID tensor_id);
    std::shared_ptr<MemBlock> allocate_on_gpu (size_t size);
    logical_time_t logical_time = 0;
    void free(TensorUID tensor_id);
public:
    logical_time_t& get_current_time() {return logical_time;}
    void set_compute_stream(cudaStream_t stream);
    Allocator(
        size_t cpu_mem = CPU_MEM_SIZE, 
        size_t gpu_mem = GPU_MEM_SIZE,
        size_t cpu_mem_limit = size_t(-1),
        size_t gpu_mem_limit = size_t(-1),
        size_t cpu_grow_size = 1024*1024,
        size_t gpu_grow_size = 1024*1024);
    void prepare(const DAO::Kernel& kernel);
    void complete(const DAO::Kernel& kernel);
    void display(std::ostream& o) const;
    template<typename... Args>
    void free(Args... args) {
        (free(args), ...); 
    }
    ~Allocator();
};

}

#endif