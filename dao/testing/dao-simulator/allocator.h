#ifndef _MEM_ALLOCATOR_H
#define _MEM_ALLOCATOR_H

#include "globals.h"
#include "devices.h"
#include "kernel.h"
#include "memory_manager.h"

#include <assert.h>
#include <queue>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

enum tensor_status_t{
    ONCPU,
    ONGPU,
    UNINITIALIZED
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
        if (q.empty() || head == 0) return 0;
        return q[head - 1];
    }
};

struct TensorRecord {
    TensorUID tensor_id;
    tensor_status_t status;  
    size_t tensor_size;
    std::shared_ptr<MemBlock> block_ptr;
    noPopQueue<logical_time_t> access_pattern;
    event_t event = 0;
    logical_time_t last_access;
    void display(std::ostream& o, bool display_block = true) const;
};


class GlobalMemRecord {
    std::unordered_map<TensorUID, size_t>& tensor_sizes;
    std::unordered_map<TensorUID, TensorRecord> tensor_record_table;
public:
    GlobalMemRecord(Trace& traces);
    void add_tensor_access(TensorUID tensor_id, size_t size, logical_time_t init_time);
    //if no such tensor id exist, create new, otherwise, lookup and update time access
    void delete_tensor(TensorUID tensor_id);
    TensorRecord& lookup_tensor(TensorUID tensor_id);
    void display(std::ostream& o) const;
};

struct report_t {
    double compute_time;
    double h2d_time;
    double d2h_time;
};

class Allocator{
private:
    stream_t H2D_stream;
    stream_t D2H_stream;
    stream_t compute_stream;
    std::unique_ptr<GlobalMemRecord> global_memory_record;
    std::unique_ptr<MemStorage> cpu_manager;
    std::unique_ptr<MemStorage> gpu_manager;
    void* prepare(TensorUID tensor_id);
    std::shared_ptr<MemBlock> allocate_on_gpu (size_t size);
    logical_time_t logical_time = 0;
    void free(TensorUID tensor_id);
public:
    Allocator(
        Trace& trace, 
        Device& device,
        DoubleLinkedListStorage::allocation_strategy_t allocation_strategy,
        DoubleLinkedListStorage::eviction_strategy_t evict_strategy
    ); 
    real_time_t get_h2d_time() const {return H2D_stream.get_copy_time();}
    real_time_t get_compute_time() const {return compute_stream.current();}
    real_time_t get_d2h_time() const {return D2H_stream.get_copy_time();}
    stream_t& get_compute_stream() {return compute_stream;}
    real_time_t get_total_time() const;
    double get_idle_rate() const {return compute_stream.idle_time / compute_stream.current();}
    void prepare(const Kernel& kernel);
    void complete(const Kernel& kernel);
    void display(std::ostream& o) const;
    template<typename... Args>
    void free(Args... args) {
        (free(args), ...); 
    }
    ~Allocator();
};


#endif