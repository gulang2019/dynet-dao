#ifndef GLOBAL_MEM_ALLOCATOR_H
#define GLOBAL_MEM_ALLOCATOR_H

// #include <DAO/memory_manager.h>
#include <DAO/globals.h>
#include <DAO/generator.h>
#include <DAO/memory_manager.h>
#include <DAO/utils.h>

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
template<typename T>
struct noPopQueue {
    std::deque<T> q;
    size_t head = 0;
    void push(const T& t) {
        q.push_back(t);
    }
    bool empty() const {
        return q.size() == head;
    }
    T front() const {
        assert(head < q.size());
        return q[head];
    }
    void pop() {
        assert(head < q.size());
        head++;
    }
    T last() {
        if (q.empty() || head == 0) return T();
        return q[head - 1];
    }
    T back() {
        if (q.empty()) return T();
        return q.back();
    }
};

struct TensorRecord {
    enum record_type_t {
        INTERMIDIATE, 
        GLOBAL
    }record_type;
    std::string name;
    TensorUID tensor_id;
    tensor_status_t status;  
    size_t tensor_size;
    std::shared_ptr<MemBlock> block_ptr;
    noPopQueue<logical_time_t> access_pattern;
    std::shared_ptr<SmartCudaEvent> event; 
    logical_time_t last_access = size_t(-1);
    void display(std::ostream& o, bool display_block = true) const;
};


class GlobalMemRecord {
    std::unordered_map<TensorUID, TensorRecord> tensor_record_table;
public:
    //if no such tensor id exist, create new, otherwise, lookup and update time access
    void delete_tensor(TensorUID tensor_id);
    TensorRecord& lookup_tensor(TensorUID tensor_id);
    void display(std::ostream& o) const;
    void Register(std::unordered_set<TensorUID>& tensor_ids, logical_time_t t);
    void cal_last_access();
    void self_check() const;
    const std::unordered_map<TensorUID, TensorRecord>& get_table() const {
        return tensor_record_table;
    }
};


class Allocator{
private:
    cudaStream_t H2D_stream;
    cudaStream_t D2H_stream;
    cudaStream_t compute_stream;
    std::unique_ptr<GlobalMemRecord> global_memory_record;
    std::unique_ptr<MemStorage> cpu_manager;
    std::unique_ptr<MemStorage> gpu_manager;
    
    std::shared_ptr<MemBlock> allocate_on_gpu (size_t size);
    logical_time_t logical_time = 0;
    logical_time_t registered_time = 0; 
    
    Kernel last_kernel;
    std::unordered_set<TensorUID> freed_tensors;
    // tensor_values for debug usage 
    std::unordered_map<TensorUID, std::vector<float> > tensor_values;


    struct AllocatorStatistics {
        size_t max_gpu_usage = 0;
        size_t max_cpu_usage = 0;        
    };

    AllocatorStatistics statistics;
public:
    Allocator() = default;
    Timer timer; 
    void init(size_t cpu_mem = CPU_MEM_SIZE, 
        size_t gpu_mem = GPU_MEM_SIZE,
        size_t cpu_grow_size = 1024*1024,
        size_t gpu_grow_size = 1024*1024, 
        DoubleLinkedListStorage::allocation_strategy_t allocation_strategy = DoubleLinkedListStorage::FIRST_FIT,
        DoubleLinkedListStorage::eviction_strategy_t evict_strategy = DoubleLinkedListStorage::WEIGHTED_BELADY); 

    std::vector<std::unordered_set<TensorUID> > all_accesses;
    std::unordered_set<TensorUID> zero_init_tensors; 
    void set_compute_stream(cudaStream_t stream);
    TensorRecord& lookup_tensor(TensorUID tensor_id);
    void finish_register();
    void Register(Kernel&& kernel);
    void prepare();
    void* prepare(TensorUID tensor_id, bool initialize = false);
    void complete();
    void display(std::ostream& o) const;
    bool check_on_gpu(const dynet::Tensor* t) const;
    void free(TensorUID tensor_id);
    // set the tensor_id to be a global tensor (i.e. never evict)
    void set_global(TensorUID tensor_id);
    std::vector<float> get_values(TensorUID tensor_id);
    // free all intermidiate tensors
    void free_intermidiates();
    Kernel& get_last_kernel();
    ~Allocator();
};

extern Allocator dao_allocator;

}

#endif