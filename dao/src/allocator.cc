#include <DAO/memory_manager.h>
#include <DAO/allocator.h>

#include <cuda_runtime.h>
#include <assert.h>
#include <unordered_set>
#include <sys/sysinfo.h>



#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


namespace DAO {

Allocator dao_allocator; 

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
    DAO_ASSERT(uid != NULL, "uid is NULL");    
    if (tensor_record_table.count(uid) == 0) {
        auto& record = tensor_record_table[uid];
        record.record_type = TensorRecord::INTERMIDIATE; 
        record.tensor_id = uid;
        record.status = UNINITIALIZED;
        record.last_access = (logical_time_t)(-1);
        // is this correct? 
        record.tensor_size = (uid)->d.size() * sizeof(float);
        record.block_ptr = NULL;
    }
    return tensor_record_table[uid];
}

void Allocator::init(
    size_t cpu_init_size,
    size_t gpu_init_size, 
    size_t cpu_grow_size,
    size_t gpu_grow_size, 
    DoubleLinkedListStorage::allocation_strategy_t allocation_strategy,
    DoubleLinkedListStorage::eviction_strategy_t evict_strategy
) {
    size_t gpu_free_mem, gpu_total_mem;
    cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);
    size_t cpu_free_mem, cpu_total_mem; 
    struct sysinfo info;
    sysinfo(&info);
    //cout << "GPU memory initial size" << device.mem_limit << endl;
    cpu_manager = std::make_unique<DoubleLinkedListStorage>(
        MemStorage::CPU, cpu_init_size, logical_time, info.freeram, cpu_grow_size,
        DoubleLinkedListStorage::allocation_strategy_t::FIRST_FIT,
        DoubleLinkedListStorage::eviction_strategy_t::EVI_FIRST_FIT,
        1024, 256u);
    gpu_manager = std::make_unique<DoubleLinkedListStorage>(
        MemStorage::GPU, gpu_init_size, logical_time, gpu_free_mem, gpu_grow_size, 
        allocation_strategy,
        evict_strategy,
        1024, 256u);
    global_memory_record = std::make_unique<GlobalMemRecord>();
    //initialize compute stream, H2D stream and D2H stream
    this->compute_stream = cudaStreamDefault;

    cudaSetDevice(DAO::default_device_id);
    CUDA_CHECK(cudaStreamCreate(&H2D_stream));
    CUDA_CHECK(cudaStreamCreate(&D2H_stream));

    DAO_INFO_LEVEL(1, "Init GPU device %d, Mem %d MB, CPU %d MB, Compute %d, H2D %d, D2H %d", DAO::default_device_id, (gpu_init_size >> 20), (cpu_init_size >> 20),  compute_stream, H2D_stream, D2H_stream);
}


void* Allocator::prepare(TensorUID uid, bool is_global){
    TensorRecord& record = global_memory_record -> lookup_tensor(uid);
    if (is_global) record.record_type = TensorRecord::GLOBAL;
    if (!is_global)
        DAO_ASSERT(record.access_pattern.front() == logical_time, "access pattern front is not logical time");
    //this->wait_for_event_if_have(record);
    DAO_ASSERT(gpu_manager->check_MemStorage(),  "gpu linked list invalid before prepare");
    DAO_ASSERT(cpu_manager->check_MemStorage(),  "cpu linked list invalid before prepare");
    if (record.status == ONCPU){
        std::shared_ptr<MemBlock> block = this->allocate_on_gpu(record.tensor_size);
        if (block != nullptr) {
            assert(!block->allocated && block->record == nullptr);
            if (record.event != nullptr)
                CUDA_CHECK(cudaStreamWaitEvent(H2D_stream, record.event->get()));
            CUDA_CHECK(cudaMemcpyAsync(block->physical_location_start, record.block_ptr->physical_location_start, record.tensor_size, cudaMemcpyHostToDevice, H2D_stream));
            record.event = std::make_shared<SmartCudaEvent>(H2D_stream, "H2D::" + std::to_string(logical_time) + "::" + record.name);
            if (record.event != nullptr)
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
    DAO_ASSERT(gpu_manager->check_MemStorage(),  "gpu linked list invalid after prepare");
    DAO_ASSERT(cpu_manager->check_MemStorage(),  "cpu linked list invalid after prepare");
    if (zero_init_tensors.count(uid)) {
        CUDA_CHECK(cudaMemsetAsync(record.block_ptr->physical_location_start, 0, record.tensor_size, compute_stream));
        zero_init_tensors.erase(uid);
    }
    statistics.max_cpu_usage = std::max(statistics.max_cpu_usage, cpu_manager->get_max_usage());
    statistics.max_gpu_usage = std::max(statistics.max_gpu_usage, gpu_manager->get_max_usage());
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
        DAO_ERROR("GPU memory insufficient when allocating %.3f MB", size / ((1 << 20) + 0.0)); 
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
            if (evict_record->event != nullptr)
                CUDA_CHECK(cudaStreamWaitEvent(D2H_stream, evict_record->event->get()));
            DAO_INFO_LEVEL(2, "copying %lu bytes from GPU %p to CPU %p using stream %d", evict_record->tensor_size, evicted_block->physical_location_start, CPU_block->physical_location_start, D2H_stream);
            CUDA_CHECK(cudaMemcpyAsync(CPU_block->physical_location_start, evicted_block->physical_location_start, evict_record->tensor_size, cudaMemcpyDeviceToHost, D2H_stream)); 
            evict_record->event = std::make_shared<SmartCudaEvent>(D2H_stream, "D2H::" + std::to_string(logical_time) + "::" + evict_record->name);
            if (evict_record->event != nullptr)
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
    DAO_INFO_LEVEL(1, "allocated %lu on GPU, evict %lu blocks", size, to_evict_blocks.size());
    return std::move(GPU_block);
}



void Allocator::prepare() {
    timer.start("prepare");
    std::unordered_set<TensorUID>& tensor_ids = all_accesses[logical_time];
    for (auto& tensor_id : tensor_ids) {
        tensor_id->v = (float*)prepare(tensor_id);
    }
    for (auto& tensor_id : tensor_ids) {
        if (debug_mode && tensor_values.count(tensor_id)) {
            std::vector<float> value(tensor_id->d.size());
            CUDA_CHECK(cudaMemcpyAsync(value.data(), tensor_id->v, value.size() * sizeof(float), cudaMemcpyDeviceToHost, compute_stream));
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));
            std::vector<float> ground_truth = tensor_values[tensor_id];
            for (int i = 0; i < value.size(); ++i) {
                DAO_ASSERT(abs(value[i] - ground_truth[i]) < 1e-3, "%f != %f", value[i], ground_truth[i]);
            }
        }
        DAO_ASSERT(check_on_gpu(tensor_id), "tensor not on gpu");
    }
    global_memory_record->self_check();
    timer.stop("prepare");
}


void Allocator::complete() {
    timer.start("complete");
    assert(gpu_manager->check_MemStorage());
    assert(cpu_manager->check_MemStorage());
    std::unordered_set<TensorUID>& tensor_ids = all_accesses[logical_time];
    std::shared_ptr<SmartCudaEvent> event = std::make_shared<SmartCudaEvent>(compute_stream, "compute::" + std::to_string(logical_time));
    for (auto& tensor_id : tensor_ids) {
        TensorRecord& record = global_memory_record -> lookup_tensor(tensor_id);
        assert(!record.access_pattern.empty() && record.access_pattern.front() == logical_time);
        record.access_pattern.pop();
        record.event = event; 
        if (record.last_access == logical_time) {
            DAO_INFO_LEVEL(1, "free tensor %lu", (size_t)tensor_id & 0xfff);
            free(tensor_id);            
        }
        else {
            if (debug_mode) {
                std::vector<float>& value = tensor_values[tensor_id];
                value.resize(tensor_id->d.size());
                CUDA_CHECK(cudaMemcpyAsync(value.data(), tensor_id->v, value.size() * sizeof(float), cudaMemcpyDeviceToHost, compute_stream));
                CUDA_CHECK(cudaStreamSynchronize(compute_stream));
            }
        }
    }
    logical_time++;
    assert(gpu_manager->check_MemStorage());
    assert(cpu_manager->check_MemStorage());
    timer.stop("complete");
}

void Allocator::display(std::ostream& o) const {
    o << "-------------Allocator----------------" << std::endl;
    if (statistics.max_cpu_usage < 1024)
        o << "max cpu usage: " << (statistics.max_cpu_usage) << "B" << std::endl;
    else if (statistics.max_cpu_usage < 1024 * 1024)
        o << "max cpu usage: " << (statistics.max_cpu_usage >> 10) << "KB" << std::endl;
    else if (statistics.max_cpu_usage < 1024 * 1024 * 1024)
        o << "max cpu usage: " << (statistics.max_cpu_usage >> 20) << "MB" << std::endl;
    else 
        o << "max cpu usage: " << (statistics.max_cpu_usage >> 30) << "GB" << std::endl;
    if (statistics.max_gpu_usage < 1024)
        o << "max gpu usage: " << (statistics.max_gpu_usage) << "B" << std::endl;
    else if (statistics.max_gpu_usage < 1024 * 1024)
        o << "max gpu usage: " << (statistics.max_gpu_usage >> 10) << "KB" << std::endl;
    else if (statistics.max_gpu_usage < 1024 * 1024 * 1024)
        o << "max gpu usage: " << (statistics.max_gpu_usage >> 20) << "MB" << std::endl;
    else 
        o << "max gpu usage: " << (statistics.max_gpu_usage >> 30) << "GB" << std::endl;
    if (verbose)
    {o << "Records: " << std::endl;
    global_memory_record->display(o);
    o << "CPU Memory: " << std::endl;
    cpu_manager->display(o);
    o << "GPU Memory: " << std::endl;
    gpu_manager->display(o);}
    timer.show(o);
    o << "------------Allocator END-------------" << std::endl;
}

Allocator::~Allocator() {
    // CUDA_CHECK(cudaStreamDestroy(H2D_stream));
    // CUDA_CHECK(cudaStreamDestroy(D2H_stream));
    global_memory_record.release();
    cpu_manager.release();
    gpu_manager.release();
}

void TensorRecord::display(std::ostream& o, bool display_block) const {
    o << "<";
    o << "ID: " << ((size_t)tensor_id & 0xfff) << ",";
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
    DAO_ASSERT(!freed_tensors.count(tensor_id), "tensor already freed or not exist");
    freed_tensors.insert(tensor_id);
    auto & record = global_memory_record->lookup_tensor(tensor_id);
    if (record.status == ONCPU) {
        cpu_manager->free(record.block_ptr);
    } else if (record.status == ONGPU) {
        gpu_manager->free(record.block_ptr);
    } else {
        DAO_WARNING("free a tensor is not allocated");
    }
    if (tensor_values.count(tensor_id)) {
        tensor_values.erase(tensor_id);
    }
    global_memory_record->delete_tensor(tensor_id);
}

void Allocator::free_intermidiates() {
    std::unordered_set<TensorUID> tensor_ids;
    for (auto& pair: global_memory_record->get_table()) {
        if (pair.second.record_type == TensorRecord::INTERMIDIATE) {
            tensor_ids.insert(pair.first);
        }
    }
    for (auto& tensor_id: tensor_ids) {
        free(tensor_id);
    }
}

std::vector<float> Allocator::get_values(TensorUID tensor_id) {
    std::vector<float> value(tensor_id->d.size());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto& record = global_memory_record->lookup_tensor(tensor_id);
    if (record.status == ONGPU) {
        CUDA_CHECK(cudaMemcpyAsync(value.data(), record.block_ptr->physical_location_start, value.size() * sizeof(float), cudaMemcpyDeviceToHost, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    }
    else if (record.status == ONCPU) {
        memcpy(value.data(), record.block_ptr->physical_location_start, value.size() * sizeof(float));
    }
    else {
        DAO_WARNING("get value from a tensor is not allocated");
        CUDA_CHECK(cudaMemcpyAsync(value.data(), tensor_id->v, value.size() * sizeof(float), cudaMemcpyDeviceToHost, compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    }
    return std::move(value);
}

TensorRecord& Allocator::lookup_tensor(TensorUID tensor_id) {
    return global_memory_record->lookup_tensor(tensor_id);
}

void Allocator::Register(Kernel&& kernel) {
    all_accesses.push_back(std::unordered_set<TensorUID>(kernel._inputs.begin(), kernel._inputs.end()));
    all_accesses.back().insert(kernel._outputs.begin(), kernel._outputs.end());
    all_accesses.back().insert(kernel._zeroed.begin(), kernel._zeroed.end());
    for (auto& tid: kernel._zeroed){
        DAO_ASSERT(zero_init_tensors.count(tid) == 0, "zero init tensor already exist");
    }
    zero_init_tensors.insert(kernel._zeroed.begin(), kernel._zeroed.end());
    global_memory_record->Register(all_accesses.back(), registered_time);
    registered_time++;
}

void GlobalMemRecord::Register(std::unordered_set<TensorUID>& tensor_ids, logical_time_t t) {
    for (auto& tid: tensor_ids) {
        auto & record = lookup_tensor(tid);
        record.access_pattern.push(t);
    }
}

void GlobalMemRecord::cal_last_access() {
    for (auto& pair: tensor_record_table) {
        auto& record = pair.second;
        DAO_COND_WARNING(record.access_pattern.empty(), "access pattern empty");
        if (record.record_type == TensorRecord::GLOBAL) 
            record.last_access = (logical_time_t)(-1);
        else 
            record.last_access = record.access_pattern.back();
    }
}

void Allocator::finish_register() {
    global_memory_record->cal_last_access();
}

void Allocator::set_compute_stream(cudaStream_t stream) {
    this->compute_stream = stream;
}

bool Allocator::check_on_gpu(const dynet::Tensor* tensor_id) const {
    auto& record = global_memory_record->lookup_tensor(const_cast<TensorUID>(tensor_id));
    DAO_ASSERT((dynet::Tensor*)tensor_id->v == (dynet::Tensor*)record.block_ptr->physical_location_start, "tensor_id->v != record.block_ptr->physical_location_start");
    DAO_ASSERT((((dynet::Tensor*)tensor_id)->d.size() * sizeof(float)) == record.tensor_size, "tensor_id->d.size() != record.tensor_size");
    DAO_ASSERT(record.block_ptr->physical_location_start + record.tensor_size <= record.block_ptr->physical_location_end, "tensor_id->v + record.tensor_size > record.block_ptr->physical_location_end");
    return record.status == ONGPU; 
}

void Allocator::set_global(TensorUID tensor_id) {
    auto& record = global_memory_record->lookup_tensor(tensor_id);
    record.record_type = TensorRecord::GLOBAL;
}

void GlobalMemRecord::self_check() const {
    if (!debug_mode) return; 
    std::vector<TensorUID> tensor_ids;
    for (auto & pair : tensor_record_table) {
        tensor_ids.push_back(pair.first);
    }

    for (int i = 0; i < tensor_ids.size(); i++) {
        auto& record = tensor_record_table.at(tensor_ids[i]);
        if (record.status != UNINITIALIZED){
            DAO_ASSERT(record.block_ptr && record.block_ptr->allocated, "record error");
            DAO_ASSERT(record.block_ptr->physical_location_start + record.tensor_size <= record.block_ptr->physical_location_end, "record error");
            DAO_ASSERT(((dynet::Tensor*)tensor_ids[i])->d.size() * sizeof(float) == record.tensor_size, "record error");
        }
    }

    std::sort(tensor_ids.begin(), tensor_ids.end(), [this](TensorUID a, TensorUID b) {
        auto& record_a = tensor_record_table.at(a);
        auto& record_b = tensor_record_table.at(b);
        if (record_a.status != record_b.status) 
            return record_a.status < record_b.status;
        if (record_a.status == UNINITIALIZED)
            return a < b;
        return record_a.block_ptr->physical_location_start < record_b.block_ptr->physical_location_start;
    });

    for (int i = 0; i < tensor_ids.size() - 1; ++i) {
        auto& record_a = tensor_record_table.at(tensor_ids[i]);
        auto& record_b = tensor_record_table.at(tensor_ids[i + 1]);
        if (record_a.status == ONGPU && record_b.status == ONGPU)
            DAO_ASSERT(record_a.block_ptr->physical_location_end <= record_b.block_ptr->physical_location_start, "tensor %lu and %lu overlap", (size_t)tensor_ids[i] & 0xfff, (size_t)tensor_ids[i + 1] & 0xfff);
        if (record_a.status == ONCPU && record_b.status == ONCPU)
            DAO_ASSERT(record_a.block_ptr->physical_location_end <= record_b.block_ptr->physical_location_start, "tensor %lu and %lu overlap", (size_t)tensor_ids[i] & 0xfff, (size_t)tensor_ids[i + 1] & 0xfff);
    }
}

} // namespace DAO