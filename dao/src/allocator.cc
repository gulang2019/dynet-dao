#include <DAO/allocator.h>
#include <DAO/executor.h>
#include <DAO/memory_manager.h>
#include <dynet/globals.h>
#include <cuda_runtime.h>

#include <dynet/devices.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

inline void sync_raw(cudaStream_t dependency, cudaStream_t dependent) {
  // CUDACachingAllocator.cpp uses raw cuda events, as do we.
  cudaEvent_t event = nullptr;
  CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventRecord(event, dependency));
  CUDA_CHECK(cudaStreamWaitEvent(dependent, event));
  CUDA_CHECK(cudaEventDestroy(event));
}

namespace DAO {

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
        record.tensor_size = (uid)->d.size() * sizeof(float);
        record.block_ptr = NULL;
        record.last_access = 0; 
        record.name = uid->name;
    }
    return tensor_record_table[uid];
}

Allocator::Allocator(
        size_t cpu_mem, 
        size_t gpu_mem,
        size_t cpu_mem_limit,
        size_t gpu_mem_limit,
        size_t cpu_grow_size,
        size_t gpu_grow_size) {
    cpu_manager = std::make_unique<DoubleLinkedListStorage>(
        MemStorage::CPU, cpu_mem, logical_time, cpu_mem_limit, cpu_grow_size,
        DoubleLinkedListStorage::FIRST_FIT,
        DoubleLinkedListStorage::FIRST_FIT,
        1024, 256);
    gpu_manager = std::make_unique<DoubleLinkedListStorage>(
        MemStorage::GPU, gpu_mem, logical_time, gpu_mem_limit, gpu_grow_size, 
        DoubleLinkedListStorage::FIRST_FIT,
        DoubleLinkedListStorage::FIRST_FIT,
        1024, 256);
    global_memory_record = std::make_unique<GlobalMemRecord>();
    compute_stream = cudaStreamDefault; 
    CUDA_CHECK(cudaStreamCreate(&H2D_stream));
    CUDA_CHECK(cudaStreamCreate(&D2H_stream));
}


void* Allocator::prepare(TensorUID uid){
    TensorRecord& record = global_memory_record -> lookup_tensor(uid);
    //this->wait_for_event_if_have(record);
    if (record.status == ONCPU){
        std::shared_ptr<MemBlock> block = this->allocate_on_gpu(record.tensor_size);
        if (block != nullptr) {
            assert(!block->allocated && block->record == nullptr);
            CUDA_CHECK(cudaStreamWaitEvent(H2D_stream, record.event->get()));
            CUDA_CHECK(cudaMemcpyAsync(block->physical_location_start, record.block_ptr->physical_location_start, record.tensor_size, cudaMemcpyHostToDevice, H2D_stream));
            record.event = std::make_shared<SmartCudaEvent>(H2D_stream, "H2D::" + std::to_string(logical_time) + "::" + record.name);
            CUDA_CHECK(cudaStreamWaitEvent(compute_stream, record.event->get()));
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
    return record.block_ptr -> physical_location_start; 
}

std::shared_ptr<MemBlock> Allocator::allocate_on_gpu (size_t size) {
    std::shared_ptr<MemBlock> block = gpu_manager->allocate(size);
    if (block != nullptr) {
        return std::move(block);
    }
    std::vector<std::shared_ptr<MemBlock>> to_evict_blocks;
    std::shared_ptr<MemBlock> front, back;
    std::tie(front, back) = gpu_manager->evict(size, to_evict_blocks);
    if (front == nullptr) {
        display(std::cout);
        DAO_ERROR("GPU memory insufficient for sustaining allocation"); 
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
            evicted_block->record->tensor_id->v = nullptr;
            CUDA_CHECK(cudaStreamWaitEvent(D2H_stream, evict_record->event->get()));
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
    assert(((size_t)GPU_block->physical_location_start & 0xff) == 0);
    return std::move(GPU_block);
}

void Allocator::prepare(const DAO::Kernel& kernel) {
    std::unordered_set<TensorUID> tensor_ids(kernel._inputs.begin(), kernel._inputs.end());
    tensor_ids.insert(kernel._outputs.begin(), kernel._outputs.end());
    for (auto& tensor_id : tensor_ids) {
        DAO_ASSERT(tensor_id != nullptr, "TensorUID is nullptr");
        TensorRecord& record = global_memory_record -> lookup_tensor(tensor_id);
        while(!record.access_pattern.empty() && record.access_pattern.front() < logical_time) {
            record.access_pattern.pop();
        }
        if (record.access_pattern.empty() || record.access_pattern.front() > logical_time) {
            record.access_pattern.push(logical_time);
        }
    }
    for (auto& uid: tensor_ids) 
        uid->v = static_cast<float*>(prepare(uid));
    // display(std::cout);
}

void Allocator::complete(const DAO::Kernel& kernel) {
    std::unordered_set<TensorUID> tensor_ids(kernel._inputs.begin(), kernel._inputs.end());
    tensor_ids.insert(kernel._outputs.begin(), kernel._outputs.end());
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
    o << "<" << name << ",";
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
        DAO_WARNING("Tensor %s is not allocated", record.name.c_str());
    }
    global_memory_record->delete_tensor(tensor_id);
}

} // namespace DAO 