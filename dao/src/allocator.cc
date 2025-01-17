#include <DAO/memory_manager.h>
#include <DAO/allocator.h>

#include <cuda_runtime.h>
#include <assert.h>
#include <unordered_set>
#include <sys/sysinfo.h>
#include <fstream>


namespace DAO {

std::unique_ptr<Allocator> dao_allocator; 

Allocator* get_allocator() {
    if (dao_allocator == nullptr) {
        throw std::runtime_error("Allocator is not initialized");
    }
    return dao_allocator.get();
}

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


TensorRecord& GlobalMemRecord::lookup_tensor(TensorUID uid, TensorRecord::record_type_t record_type) {
    DAO_ASSERT(uid != NULL, "uid is NULL");    
    if (tensor_record_table.count(uid) == 0) {
        auto& record = tensor_record_table[uid];
        record.record_type = record_type; 
        record.tensor_id = uid;
        record.status = UNINITIALIZED;
        record.last_access = (logical_time_t)(-1);
        // is this correct? 
        record.tensor_size = (uid)->d.size() * sizeof(float);
        record.block_ptr = NULL;
        record.name = uid->name;
        assert(record.name.size());
    }
    return tensor_record_table[uid];
}

Allocator::Allocator(
    size_t cpu_init_size,
    size_t gpu_init_size, 
    size_t cpu_grow_size,
    size_t gpu_grow_size, 
    DoubleLinkedListStorage::allocation_strategy_t allocation_strategy,
    DoubleLinkedListStorage::eviction_strategy_t evict_strategy,
    cudaStream_t* p_compute_stream
) {
    size_t gpu_free_mem, gpu_total_mem;
    cudaMemGetInfo(&gpu_free_mem, &gpu_total_mem);
    struct sysinfo info;
    sysinfo(&info);
    //cout << "GPU memory initial size" << device.mem_limit << endl;
    cpu_manager = std::make_unique<DoubleLinkedListStorage>(
        MemStorage::CPU, cpu_init_size, logical_time, info.freeram, cpu_grow_size,
        DoubleLinkedListStorage::allocation_strategy_t::FIRST_FIT,
        DoubleLinkedListStorage::eviction_strategy_t::EVI_FIRST_FIT,
        1024, 64u);
    gpu_manager = std::make_unique<DoubleLinkedListStorage>(
        MemStorage::GPU, gpu_init_size, logical_time, gpu_free_mem, gpu_grow_size, 
        allocation_strategy,
        evict_strategy,
        1024, 64u);
    global_memory_record = std::make_unique<GlobalMemRecord>();
    //initialize compute stream, H2D stream and D2H stream
    if (p_compute_stream == nullptr)
        this->compute_stream = cudaStreamDefault;
    else this->compute_stream = *p_compute_stream;

    cudaSetDevice(DAO::default_device_id);
    CUDA_CHECK(cudaStreamCreate(&H2D_stream));
    CUDA_CHECK(cudaStreamCreate(&D2H_stream));

    DAO_INFO_LEVEL(0, "Init GPU device %d, Mem %ld MB, CPU %ld MB, Compute %p, H2D %p, D2H %p", DAO::default_device_id, (gpu_init_size >> 20), (cpu_init_size >> 20),  compute_stream, H2D_stream, D2H_stream);
}


Allocator::~Allocator() {
    CUDA_CHECK(cudaStreamDestroy(H2D_stream));
    CUDA_CHECK(cudaStreamDestroy(D2H_stream));
}

void* Allocator::prepare(TensorUID uid, TensorRecord::record_type_t record_type) {
    TensorRecord& record = global_memory_record -> lookup_tensor(uid, record_type);
    if (record_type == TensorRecord::INTERMIDIATE) { // this should be called from the prepare() only; 
        DAO_ASSERT(record.access_pattern.front() == logical_time, "access pattern front is not logical time");}
    //this->wait_for_event_if_have(record);
    // DAO_ASSERT(gpu_manager->check_MemStorage(),  "gpu linked list invalid before prepare");
    // DAO_ASSERT(cpu_manager->check_MemStorage(),  "cpu linked list invalid before prepare");
    if (record.status == ONCPU){
        std::shared_ptr<MemBlock> block = this->allocate_on_gpu(record.tensor_size);
        if (block != nullptr) {
            assert(!block->allocated && block->record == nullptr);
            if (record.event != nullptr)
                CUDA_CHECK(cudaStreamWaitEvent(H2D_stream, record.event->get()));
            CUDA_CHECK(cudaMemcpyAsync(block->physical_location_start, record.block_ptr->physical_location_start, record.tensor_size, cudaMemcpyHostToDevice, H2D_stream));
            timer.cumint("H2D", record.tensor_size);
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
    if (zero_init_tensors.count(uid)) {
        CUDA_CHECK(cudaMemsetAsync(record.block_ptr->physical_location_start, 0, record.tensor_size, compute_stream));
        zero_init_tensors.erase(uid);
    }
    max_cpu_usage = std::max(max_cpu_usage, cpu_manager->get_max_usage());
    max_gpu_usage = std::max(max_gpu_usage, gpu_manager->get_max_usage());
    if (offload_profiling == 1) {
        statistics.push_back({});
        auto & stat = statistics.back();
        global_memory_record->get_statistics(stat);
        stat.total_cpu_usage = cpu_manager->get_max_usage();
        stat.total_gpu_usage = gpu_manager->get_max_usage();
    }
    return record.block_ptr -> physical_location_start;
}

std::shared_ptr<MemBlock> Allocator::allocate_on_gpu (size_t size) {
    std::shared_ptr<MemBlock> block = gpu_manager->allocate(size);
    if (block != nullptr) {
        return block;
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
            DAO_INFO_LEVEL(2, "copying %lu bytes from GPU %p to CPU %p using stream %p", evict_record->tensor_size, evicted_block->physical_location_start, CPU_block->physical_location_start, D2H_stream);
            CUDA_CHECK(cudaMemcpyAsync(CPU_block->physical_location_start, evicted_block->physical_location_start, evict_record->tensor_size, cudaMemcpyDeviceToHost, D2H_stream)); 
            evict_record->event = std::make_shared<SmartCudaEvent>(D2H_stream, "D2H::" + std::to_string(logical_time) + "::" + evict_record->name);
            timer.cumint("D2H", evict_record->tensor_size);
            if (evict_record->event != nullptr)
                CUDA_CHECK(cudaStreamWaitEvent(compute_stream, evict_record->event->get()));

            CPU_block->record = evict_record; 
            CPU_block->allocated = true;
            evict_record->block_ptr = CPU_block;
            evict_record->status = ONCPU;
            evicted_block->record = nullptr;
            evicted_block->allocated = false;
            evict_record->tensor_id->v = nullptr;
        }
    }
    std::shared_ptr<MemBlock> GPU_block = gpu_manager -> mergeAndAllocate(size, front, back);
    DAO_INFO_LEVEL(1, "allocated %lu on GPU, evict %lu blocks", size, to_evict_blocks.size());
    return GPU_block;
}



void Allocator::prepare() {
    timer.start("prepare");
    std::unordered_set<TensorUID>& tensor_ids = all_accesses[logical_time];
    if (profiling) {
        accessed_tensors.insert(tensor_ids.begin(), tensor_ids.end());
    }
    for (auto& tensor_id : tensor_ids) {
        tensor_id->v = (float*)prepare(tensor_id, TensorRecord::INTERMIDIATE);
    }
    if (debug_mode) {
        for (auto& tensor_id : tensor_ids) {
        // if (tensor_values.count(tensor_id)) {
        //     std::vector<float> value(tensor_id->d.size());
        //     CUDA_CHECK(cudaMemcpyAsync(value.data(), tensor_id->v, value.size() * sizeof(float), cudaMemcpyDeviceToHost, compute_stream));
        //     CUDA_CHECK(cudaStreamSynchronize(compute_stream));
        //     std::vector<float> ground_truth = tensor_values[tensor_id];
        //     for (int i = 0; i < value.size(); ++i) {
        //         DAO_ASSERT(abs(value[i] - ground_truth[i]) < 1e-3, "%f != %f", value[i], ground_truth[i]);
        //     }
        // }
        DAO_ASSERT(check_on_gpu(tensor_id), "tensor not on gpu");
        }
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
            DAO_INFO_LEVEL(2, "free tensor %lu", (size_t)tensor_id & 0xfff);
            free(tensor_id);            
        }
        else {
            // if (debug_mode) {
            //     std::vector<float>& value = tensor_values[tensor_id];
            //     value.resize(tensor_id->d.size());
            //     CUDA_CHECK(cudaMemcpyAsync(value.data(), tensor_id->v, value.size() * sizeof(float), cudaMemcpyDeviceToHost, compute_stream));
            //     CUDA_CHECK(cudaStreamSynchronize(compute_stream));
            // }
        }
    }
    logical_time++;
    assert(gpu_manager->check_MemStorage());
    assert(cpu_manager->check_MemStorage());
    timer.stop("complete");
}

void Allocator::display(std::ostream& o) const {
    o << "-------------Allocator----------------" << std::endl;
    if (max_cpu_usage < 1024)
        o << "max cpu usage: " << (max_cpu_usage) << "B" << std::endl;
    else if (max_cpu_usage < 1024 * 1024)
        o << "max cpu usage: " << (max_cpu_usage >> 10) << "KB" << std::endl;
    else if (max_cpu_usage < 1024 * 1024 * 1024)
        o << "max cpu usage: " << (max_cpu_usage >> 20) << "MB" << std::endl;
    else 
        o << "max cpu usage: " << (max_cpu_usage >> 30) << "GB" << std::endl;
    if (max_gpu_usage < 1024)
        o << "max gpu usage: " << (max_gpu_usage) << "B" << std::endl;
    else if (max_gpu_usage < 1024 * 1024)
        o << "max gpu usage: " << (max_gpu_usage >> 10) << "KB" << std::endl;
    else if (max_gpu_usage < 1024 * 1024 * 1024)
        o << "max gpu usage: " << (max_gpu_usage >> 20) << "MB" << std::endl;
    else 
        o << "max gpu usage: " << (max_gpu_usage >> 30) << "GB" << std::endl;
    if (verbose)
    {o << "Records: " << std::endl;
    global_memory_record->display(o);}
    cpu_manager->display(o);
    gpu_manager->display(o);
    timer.show(o);
    o << "------------Allocator END-------------" << std::endl;
}

void TensorRecord::display(std::ostream& o, bool display_block) const {
    o << "<";
    o << name << ", ";
    std::string status_tag = status == ONCPU ? "CPU" : (status == ONGPU ? "GPU" : "UND");
    o << status_tag << ",";
    o << "N: " << tensor_size << ", ";
    if (block_ptr) {
        o << "P: " << block_ptr->physical_location_start << ",";
    }
    assert(status == UNINITIALIZED || block_ptr != nullptr);
    if (display_block){
        if (block_ptr) {
            o << "B: ";
            block_ptr->display(o);
        }
    }
    // size_t na = access_pattern.empty()? -1: access_pattern.front();
    // o << "A: " << na << ",";
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

void Allocator::reset() {
    // free all allocated tensors; 
    std::unordered_set<TensorUID> tensor_ids;
    for (auto& pair: global_memory_record->get_table()) {
        if (pair.second.record_type == TensorRecord::INTERMIDIATE) {
            tensor_ids.insert(pair.first);
        }
    }
    for (auto& tensor_id: tensor_ids) {
        free(tensor_id);
    }
    freed_tensors.clear();
    // clear up 
    assert(registered_time == all_accesses.size());
    DAO_COND_WARNING(logical_time != all_accesses.size(), "not all kernels are executed");
    logical_time = 0;
    registered_time = 0;
    max_gpu_usage = max_cpu_usage = 0;
    all_accesses.clear();
    zero_init_tensors.clear();
    profiling = false;
    accessed_tensors.clear();
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
    return value;
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
    last_kernel = std::move(kernel);
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
        if (record.record_type != TensorRecord::INTERMIDIATE) 
            record.last_access = (logical_time_t)(-1);
        else 
            record.last_access = record.access_pattern.back();
    }
}

void Allocator::finish_register() {
    global_memory_record->cal_last_access();
}

bool Allocator::check_on_gpu(const dynet::Tensor* tensor_id) const {
    auto& record = global_memory_record->lookup_tensor(const_cast<TensorUID>(tensor_id));
    DAO_ASSERT(record.status != UNINITIALIZED, "tensor not initialized");
    DAO_ASSERT((dynet::Tensor*)tensor_id->v == (dynet::Tensor*)record.block_ptr->physical_location_start, "tensor_id->v != record.block_ptr->physical_location_start");
    DAO_ASSERT((((dynet::Tensor*)tensor_id)->d.size() * sizeof(float)) == record.tensor_size, "tensor_id->d.size() != record.tensor_size");
    DAO_ASSERT((size_t)record.block_ptr->physical_location_start + record.tensor_size <= (size_t)record.block_ptr->physical_location_end, "tensor_id->v + record.tensor_size > record.block_ptr->physical_location_end");
    return record.status == ONGPU; 
}

void Allocator::set_record_type(TensorUID tensor_id, TensorRecord::record_type_t type) {
    global_memory_record->lookup_tensor(tensor_id).record_type = type;
}

void GlobalMemRecord::self_check() const {
    if (!debug_mode) return; 
    std::vector<TensorUID> tensor_ids;
    for (auto & pair : tensor_record_table) {
        tensor_ids.push_back(pair.first);
    }

    for (size_t i = 0; i < tensor_ids.size(); i++) {
        auto& record = tensor_record_table.at(tensor_ids[i]);
        if (record.status != UNINITIALIZED){
            DAO_ASSERT(record.block_ptr && record.block_ptr->allocated, "tensor %lu is not allocated", (size_t)tensor_ids[i] & 0xfff);
            DAO_ASSERT(record.block_ptr->physical_location_start + record.tensor_size <= record.block_ptr->physical_location_end, "record error");
            DAO_ASSERT(((dynet::Tensor*)tensor_ids[i])->d.size() * sizeof(float) == record.tensor_size, "%lu != %lu", ((dynet::Tensor*)tensor_ids[i])->d.size() * sizeof(float), record.tensor_size);
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

    for (size_t i = 0; i < tensor_ids.size() - 1; ++i) {
        auto& record_a = tensor_record_table.at(tensor_ids[i]);
        auto& record_b = tensor_record_table.at(tensor_ids[i + 1]);
        if (record_a.status == ONGPU && record_b.status == ONGPU)
            DAO_ASSERT(record_a.block_ptr->physical_location_end <= record_b.block_ptr->physical_location_start, "tensor %lu and %lu overlap", (size_t)tensor_ids[i] & 0xfff, (size_t)tensor_ids[i + 1] & 0xfff);
        if (record_a.status == ONCPU && record_b.status == ONCPU)
            DAO_ASSERT(record_a.block_ptr->physical_location_end <= record_b.block_ptr->physical_location_start, "tensor %lu and %lu overlap", (size_t)tensor_ids[i] & 0xfff, (size_t)tensor_ids[i + 1] & 0xfff);
    }
}

Kernel& Allocator::get_last_kernel() {
    return last_kernel;
}

void GlobalMemRecord::get_statistics(MemoryStatistics& stats) const {
    auto & breakdown = stats.breakdown;
    for (auto status: {UNINITIALIZED, ONCPU, ONGPU}) {
        for (auto record_type: {TensorRecord::INTERMIDIATE,
         TensorRecord::OPTIMIZER_STATE, TensorRecord::PARAMETER, TensorRecord::OUTPUT}) {
            breakdown[status][record_type] = 0;
        }
    }
    for (auto& kv: tensor_record_table) {
        auto& record = kv.second;
        breakdown[record.status][record.record_type] += record.tensor_size;
    }
}

void Allocator::dump_memory_breakdown(const std::string& filename) const {
    if (offload_profiling != 1) return; 
    std::ofstream file;
    if (filename.find(".csv") == std::string::npos) {
        file.open(filename + ".csv");
    }
    else file.open(filename);
    file << "cpu,cpu_total,cpu_intermediate,cpu_output,cpu_parameter,cpu_optimizer_state,gpu,gpu_total,gpu_intermediate,gpu_output,gpu_parameter,gpu_optimizer_state" << std::endl;
    for (auto & stats: statistics) {
        auto& breakdown = (stats.breakdown);
        size_t cpu_mem = 0;
        for (auto record_type: {TensorRecord::INTERMIDIATE, TensorRecord::OUTPUT, TensorRecord::PARAMETER, TensorRecord::OPTIMIZER_STATE}) {
            cpu_mem += breakdown.at(ONCPU).at(record_type);
        }
        size_t gpu_mem = 0;
        for (auto record_type: {TensorRecord::INTERMIDIATE, TensorRecord::OUTPUT, TensorRecord::PARAMETER, TensorRecord::OPTIMIZER_STATE}) {
            gpu_mem += breakdown.at(ONGPU).at(record_type);
        }
        file << "," << cpu_mem << "," << stats.total_cpu_usage << "," << breakdown.at(ONCPU).at(TensorRecord::INTERMIDIATE) << "," << breakdown.at(ONCPU).at(TensorRecord::OUTPUT) << ","
        << breakdown.at(ONCPU).at(TensorRecord::PARAMETER) << "," << breakdown.at(ONCPU).at(TensorRecord::OPTIMIZER_STATE) << "," << gpu_mem << "," << stats.total_gpu_usage << "," << breakdown.at(ONGPU).at(TensorRecord::INTERMIDIATE) << ","
        << breakdown.at(ONGPU).at(TensorRecord::OUTPUT) << "," << breakdown.at(ONGPU).at(TensorRecord::PARAMETER) << "," << breakdown.at(ONGPU).at(TensorRecord::OPTIMIZER_STATE) << std::endl;
    }
    file.close();
}

void Allocator::start_profiling() {
    profiling = true;
}

size_t Allocator::end_profiling() {
    size_t ret = 0;
    for (auto t: accessed_tensors) {
        ret += t->d.size() * sizeof(float);
    }
    profiling = false;
    accessed_tensors.clear();
    return ret;
}

} // namespace DAO