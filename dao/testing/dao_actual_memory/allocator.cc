#include "memory_manager.h"
#include <cuda_runtime.h>
#include <assert.h>

#include <unordered_set>

#include "allocator.h"
#include "globals.h"


#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


SmartCudaEvent::SmartCudaEvent(cudaStream_t stream, const std::string& name): name(name) {
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(event, stream));
}

cudaEvent_t SmartCudaEvent::get() {
    return event;
}

SmartCudaEvent::~SmartCudaEvent() {
    CUDA_CHECK(cudaEventDestroy(event));
}



TensorRecord& GlobalMemRecord::lookup_tensor(TensorUID uid) {
    if (tensor_record_table.count(uid) == 0) {
        auto& record = tensor_record_table[uid];
        record.tensor_id = uid;
        record.status = UNINITIALIZED;
        // is this correct? 
        assert(tensor_sizes.count(uid));
        record.tensor_size = tensor_sizes[uid];
        record.block_ptr = NULL;
    }
    return tensor_record_table[uid];
}



Allocator::Allocator(
    Trace& trace,
    Device& device,
    DoubleLinkedListStorage::allocation_strategy_t allocation_strategy,
    DoubleLinkedListStorage::eviction_strategy_t evict_strategy
) {
    //cout << "GPU memory initial size" << device.mem_limit << endl;
    cout << "size of size_t: " << sizeof(size_t) << endl;
    cpu_manager = std::make_unique<DoubleLinkedListStorage>(
        MemStorage::CPU, ((size_t) 1) << 31, logical_time, ((size_t)1) << 31, 0,
        DoubleLinkedListStorage::allocation_strategy_t::FIRST_FIT,
        DoubleLinkedListStorage::eviction_strategy_t::EVI_FIRST_FIT,
        1024, 256);
    cout << "GPU memory initial size" << device.mem_limit << endl;
    gpu_manager = std::make_unique<DoubleLinkedListStorage>(
        MemStorage::GPU, device.mem_limit, logical_time, device.mem_limit, 0, 
        allocation_strategy,
        evict_strategy,
        1024, 256);
    global_memory_record = std::make_unique<GlobalMemRecord>(trace);
    //initialize compute stream, H2D stream and D2H stream
    this->compute_stream = cudaStreamDefault;
    cudaStream_t H2D_stream;
    cudaStream_t D2H_stream;
    
    CUDA_CHECK(cudaStreamCreate(&H2D_stream));
    CUDA_CHECK(cudaStreamCreate(&D2H_stream));
    this->H2D_stream = H2D_stream;
    this->D2H_stream = D2H_stream; 
    //assert(false);
    //H2D_stream.bandwidth = device.mem_bandwidth;
    //D2H_stream.bandwidth = device.mem_bandwidth;
}



GlobalMemRecord::GlobalMemRecord(Trace& trace): tensor_sizes(trace.tensor_sizes) {
    logical_time_t t = 0;
    unordered_map<TensorUID, logical_time_t> last_accesses;
    for (auto& kernel : trace.kernels) {
        for (auto& tensor_id : kernel.accesses) {
            TensorRecord& record = lookup_tensor(tensor_id);
            record.access_pattern.push(t);
            last_accesses[tensor_id] = t;
        }
        t++; 
    }
    for (auto& last_access: last_accesses) {
        trace.kernels[last_access.second]._free_list.push_back(last_access.first);
    }
}




void* Allocator::prepare(TensorUID uid){
    TensorRecord& record = global_memory_record -> lookup_tensor(uid);
    //this->wait_for_event_if_have(record);
    DAO_ASSERT(gpu_manager->check_MemStorage(),  "gpu linked list invalid before prepare");
    DAO_ASSERT(cpu_manager->check_MemStorage(),  "cpu linked list invalid before prepare");
    if (record.status == ONCPU){
        DAO_INFO("moving record from CPU to GPU");
        std::shared_ptr<MemBlock> block = this->allocate_on_gpu(record.tensor_size);
        if (block != nullptr) {
            assert(!block->allocated && block->record == nullptr);

            CUDA_CHECK(cudaStreamWaitEvent(H2D_stream, record.event->get()));
            CUDA_CHECK(cudaMemcpyAsync(block->physical_location_start, record.block_ptr->physical_location_start, record.tensor_size, cudaMemcpyHostToDevice, H2D_stream));
            record.event = std::make_shared<SmartCudaEvent>(H2D_stream, "H2D::" + std::to_string(logical_time) + "::" + record.name);
            CUDA_CHECK(cudaStreamWaitEvent(compute_stream, record.event->get()));

            //H2D_stream.wait_for(record.event);
            //H2D_stream.copy(record.tensor_size);
            //record.event = H2D_stream.current();
            //compute_stream.wait_for(record.event);
            
            record.status = ONGPU; 
            cpu_manager->free(record.block_ptr);
            record.block_ptr = block; 
            block->record = &record;
            block->allocated = true;  
        } else {
            DAO_ERROR("Insufficient CPU Memory for offloading");
        }
    }
    else if (record.status == UNINITIALIZED) {
        DAO_INFO("record status need to be initialized");
        std::shared_ptr<MemBlock> block = this -> allocate_on_gpu(record.tensor_size);
        assert(record.block_ptr == nullptr);
        if (block == nullptr) {
            DAO_ERROR("Insufficient CPU Memory for offloading");
        }
        assert(!block->allocated && block->record == nullptr);
        record.status = ONGPU;
        record.block_ptr = block;
        block->record = &record;
        block->allocated = true;
    }
    DAO_ASSERT(gpu_manager->check_MemStorage(),  "gpu linked list invalid after prepare");
    DAO_ASSERT(cpu_manager->check_MemStorage(),  "cpu linked list invalid after prepare");
    return record.block_ptr -> physical_location_start; 
}



std::shared_ptr<MemBlock> Allocator::allocate_on_gpu (size_t size) {
    //assert(gpu_manager->check_MemStorage());
    //assert(cpu_manager->check_MemStorage());
    std::shared_ptr<MemBlock> block = gpu_manager->allocate(size);
    if (block != nullptr) {
        return std::move(block);
    }
    std::vector<std::shared_ptr<MemBlock>> to_evict_blocks;
    std::shared_ptr<MemBlock> front, back;
    std::tie(front, back) = gpu_manager->evict(size, to_evict_blocks);
    if (front == nullptr) {
        display(std::cout);
        DAO_ERROR("GPU memory insufficient when allocating %d", size); 
    }
    for (auto & evicted_block : to_evict_blocks) {
        assert(evicted_block != nullptr);
        assert(evicted_block->allocated);
        assert(evicted_block->record != nullptr);
        auto evict_record = evicted_block->record;
        std::shared_ptr<MemBlock> CPU_block = cpu_manager -> allocate(evict_record->tensor_size);
        if (CPU_block == nullptr) {
            DAO_ERROR("No memory left on CPU");
        } else {
            //cout << "eviction " << endl;
            //D2H_stream.wait_for(evict_record->event);
            //D2H_stream.copy(evict_record->tensor_size);
            //evict_record->event = D2H_stream.current();
            //compute_stream.wait_for(evict_record->event);

            CUDA_CHECK(cudaStreamWaitEvent(D2H_stream, evict_record->event->get()));
            DAO_INFO("copying %lu bytes from GPU %p to CPU %p using stream %d", evict_record->tensor_size, evicted_block->physical_location_start, CPU_block->physical_location_start, D2H_stream);
            CUDA_CHECK(cudaMemcpyAsync(CPU_block->physical_location_start, evicted_block->physical_location_start, evict_record->tensor_size, cudaMemcpyDeviceToHost, D2H_stream)); 
            evict_record->event = std::make_shared<SmartCudaEvent>(D2H_stream, "D2H::" + std::to_string(logical_time) + "::" + evict_record->name);
            CUDA_CHECK(cudaStreamWaitEvent(compute_stream, evict_record->event->get()));

            CPU_block->record = evict_record; 
            CPU_block->allocated = true;
            evict_record->block_ptr = CPU_block;
            evict_record->status = ONCPU;
            evicted_block->record = nullptr;
            evicted_block->allocated = false;
        }
    }
    std::shared_ptr<MemBlock> GPU_block = gpu_manager -> mergeAndAllocate(size, front, back);
    return std::move(GPU_block);
}



void Allocator::prepare(const Kernel& kernel) {
    DAO_ASSERT(gpu_manager->check_MemStorage(), "gpu linked list invalid before prepare");
    DAO_ASSERT(cpu_manager->check_MemStorage(), "cpu linked list invalid before prepare");
    std::unordered_set<TensorUID> tensor_ids(kernel.accesses.begin(), kernel.accesses.end());
    for (auto& tensor_id : tensor_ids) {
        TensorRecord& record = global_memory_record -> lookup_tensor(tensor_id);
        while(!record.access_pattern.empty() && record.access_pattern.front() < logical_time) {
            record.access_pattern.pop();
        }
        if (record.access_pattern.empty() || record.access_pattern.front() > logical_time) {
            record.access_pattern.push(logical_time);
        }
    }
    for (auto& tensor_id : tensor_ids) {
        prepare(tensor_id);
    }
    DAO_ASSERT(gpu_manager->check_MemStorage(), "gpu linked list invalid after prepare");
    DAO_ASSERT(cpu_manager->check_MemStorage(), "cpu linked list invalid after prepare");
}



void Allocator::complete(const Kernel& kernel) {
    assert(gpu_manager->check_MemStorage());
    assert(cpu_manager->check_MemStorage());
    std::unordered_set<TensorUID> tensor_ids(kernel.accesses.begin(), kernel.accesses.end());
    //tensor_ids.insert(kernel._outputs.begin(), kernel._output.end());
    
    //replace fake events with true events
    //auto event = compute_stream.current();

    std::shared_ptr<SmartCudaEvent> event = std::make_shared<SmartCudaEvent>(compute_stream, "compute::" + std::to_string(logical_time));

    for (auto& tensor_id : tensor_ids) {
        TensorRecord& record = global_memory_record -> lookup_tensor(tensor_id);
        assert(!record.access_pattern.empty() && record.access_pattern.front() == logical_time);
        record.access_pattern.pop(); 
        record.event = event; 
    }
    for (auto& tensor_id: kernel._free_list) {
        free(tensor_id);
    }
    logical_time++;
    assert(gpu_manager->check_MemStorage());
    assert(cpu_manager->check_MemStorage());
}

void Allocator::display(std::ostream& o) const {
    o << "-------------Allocator----------------" << std::endl;
    o << "Records: " << std::endl;
    global_memory_record->display(o);
    o << "CPU Memory: " << std::endl;
    cpu_manager->display(o);
    o << "GPU Memory: " << std::endl;
    gpu_manager->display(o);
    o << "------------Allocator END-------------" << std::endl;
}

Allocator::~Allocator() {
    CUDA_CHECK(cudaStreamDestroy(H2D_stream));
    CUDA_CHECK(cudaStreamDestroy(D2H_stream));
    global_memory_record.release();
    cpu_manager.release();
    gpu_manager.release();
}

void TensorRecord::display(std::ostream& o, bool display_block) const {
    o << "<" ;
    o << "ID: " << ((size_t)tensor_id & 0xff) << ",";
    std::string status_tag = status == ONCPU ? "CPU" : (status == ONGPU ? "GPU" : "UND");
    o << "S: " << status_tag << ",";
    assert(status == UNINITIALIZED || block_ptr != nullptr);
    if (display_block){
        if (block_ptr) {
            o << "B: ";
            block_ptr->display(o);
        }
    }
    size_t na = access_pattern.empty()? -1: access_pattern.front();
    o << "A: " << na << ",";
    o << ">,";
}

void GlobalMemRecord::display(std::ostream& o) const {
    for (auto& pair : tensor_record_table) {
        if (pair.second.status == UNINITIALIZED) {
            continue;
        }
        pair.second.display(o);
        o << std::endl; 
    }
}


void GlobalMemRecord::delete_tensor(TensorUID tensor_id) {
    tensor_record_table.erase(tensor_id);
}



void Allocator::free(TensorUID tensor_id) { 
    auto & record = global_memory_record->lookup_tensor(tensor_id);
    if (record.status == ONCPU) {
        cpu_manager->free(record.block_ptr);
    } else if (record.status == ONGPU) {
        gpu_manager->free(record.block_ptr);
    } else {
        DAO_WARNING("Tensor %s is not allocated");
    }
    global_memory_record->delete_tensor(tensor_id);
}

TensorRecord& Allocator::lookup_tensor(TensorUID tensor_id) {
    return global_memory_record->lookup_tensor(tensor_id);
}

/*

real_time_t Allocator::get_total_time() const {
    return std::max(compute_stream.current(), std::max(H2D_stream.current(), D2H_stream.current()));
}
*/

