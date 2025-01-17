#include <deque>
#include <assert.h>
#include <climits>
#include <sys/sysinfo.h>
#include <iostream>

#include <DAO/memory_manager.h>
#include <DAO/allocator.h>
#include <DAO/globals.h>

#define ROUND_UP(x, s) ((x + s - 1) / s * s)

namespace DAO {

void VirtualMallocr::raw_free(void* ptr) {
    // do nothing
}

void* VirtualMallocr::raw_malloc(size_t size) {
    allocated += 1024; // for discontinuity 
    void* ret = (void*)allocated;
    allocated += size;
    return ret;
}

size_t VirtualMallocr::mem_limit() {
    return size_t(-1);
}

void CPURealMallocr::raw_free(void* ptr) {
    if (paged_memory.count(ptr)) {
        paged_memory.erase(ptr);
        free(ptr);
        return; 
    }
    CUDA_CHECK(cudaSetDevice(DAO::default_device_id));
    CUDA_CHECK(cudaFreeHost(ptr));
    return; 
}

void* CPURealMallocr::raw_malloc(size_t size) {
    allocated += size;     
    if (size <= pinned_thresholde) {
        void* ptr = nullptr;
        CUDA_CHECK(cudaSetDevice(DAO::default_device_id));
        CUDA_CHECK(cudaMallocHost(&ptr, size));
        return ptr;
    }
    void* ptr = malloc(size);
    paged_memory.insert(ptr);
    return ptr;
}

size_t CPURealMallocr::mem_limit() {
    CUDA_CHECK(cudaSetDevice(DAO::default_device_id));
    struct sysinfo info;
    sysinfo(&info);
    return info.freeram;
}

void GPURealMallocr::raw_free(void* ptr) {
    CUDA_CHECK(cudaSetDevice(DAO::default_device_id));
    CUDA_CHECK(cudaFree(ptr));   
}

void* GPURealMallocr::raw_malloc(size_t size) {
    CUDA_CHECK(cudaSetDevice(DAO::default_device_id));
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    allocated += size;
    return ptr;
}

size_t GPURealMallocr::mem_limit() {
    CUDA_CHECK(cudaSetDevice(DAO::default_device_id));
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total)); 
    return free;
}


MemStorage::MemStorage(device_type_t device_type, const std::string& name): 
    device_type(device_type), name(name) {
    if (device_type == CPU) {
        device_mallocr = std::make_unique<CPURealMallocr>();
    } else if (device_type == GPU) {
        device_mallocr = std::make_unique<GPURealMallocr>();
    } else if (device_type == VIRTUAL) {
        device_mallocr = std::make_unique<VirtualMallocr>();
    }
}


DoubleLinkedListStorage::DoubleLinkedListStorage(
    MemStorage::device_type_t device_type,
    size_t device_mem_size,
    logical_time_t& logical_time,
    size_t _mem_limit,
    size_t _grow_size, 
    allocation_strategy_t _allocation_strategy,
    eviction_strategy_t _eviction_strategy,
    size_t _split_threshold, 
    size_t _alignment): 
    MemStorage(device_type, "DoubleLinkedListStorage"), allocated(0), alignment(_alignment), 
    split_threshold(_split_threshold), allocation_strategy(_allocation_strategy),
    eviction_strategy(_eviction_strategy),mem_limit(_mem_limit), logical_time(logical_time){
    size_t initial_mem_size = ROUND_UP(device_mem_size, alignment);

    grow_size = ROUND_UP(_grow_size, alignment);
    if (mem_limit == (size_t)-1) 
        mem_limit = device_mallocr->mem_limit(); 

    start = std::make_shared<MemBlock>();
    end = std::make_shared<MemBlock>();
    start->allocated = end->allocated = true; 
    std::shared_ptr<MemBlock> first_block = std::make_shared<MemBlock>();
    start->next = first_block;
    first_block->prev = start;
    first_block->next = end;
    end->prev = first_block;
    void* ptr = device_mallocr->raw_malloc(device_mem_size + alignment);

    DAO_ASSERT(ptr, "Not enough memory for DoubleLinkedListStorage %lu MB", (device_mem_size + alignment) / (1 << 20));

    allocated_list.push_back(ptr);
    first_block->physical_location_start = (void*)ROUND_UP((unsigned long)ptr, alignment);
    allocated += device_mem_size + alignment;
    // CUDA_CHECK(cudaMalloc(&first_block->physical_location_start, device_mem_size));
    first_block->physical_location_end = (void*)((unsigned long)(first_block -> physical_location_start) + initial_mem_size);

    assert((unsigned long)(first_block -> physical_location_start) < (unsigned long)(first_block -> physical_location_end));

    first_block->payload_size = initial_mem_size; 
    
    start->physical_location_start = start->physical_location_end = first_block->physical_location_start;
    end->physical_location_start = end->physical_location_end = first_block->physical_location_end;

    std::string device_tag = (device_type == CPU) ? "CPU" : "GPU";
    MemStorage::name += "::" + device_tag;
    DAO_INFO("DoubleLinkedListStorage created on %s, device_mem_size = %lu, grow_size = %lu, mem_limit = %lu, S = %p, E = %p", device_tag.c_str(), device_mem_size, grow_size, mem_limit, start->physical_location_start, end->physical_location_end);
    DAO_ASSERT(this -> check_MemStorage(), "invalid initialization;");
}

std::shared_ptr<MemBlock> DoubleLinkedListStorage::allocate(size_t size) {
    timer.start("allocate");
    if (debug_mode) DAO_ASSERT(check_MemStorage(), "before allocation");
    size = ROUND_UP(size, alignment);
    // first check if there has available space
    std::shared_ptr<MemBlock> best_iter = nullptr;
    std::shared_ptr<MemBlock> iter = start;
    while(iter->next != nullptr) {
        if (iter->allocated == false && iter->payload_size >= size) {
            if (allocation_strategy == allocation_strategy_t::FIRST_FIT) {
                best_iter = iter;
                break; 
            } else if (allocation_strategy == allocation_strategy_t::BEST_FIT) {
                if (!best_iter || ((iter->payload_size - size) < (best_iter->next->payload_size - size))) {
                    best_iter = iter;
                }
            } else if (allocation_strategy == allocation_strategy_t::WORST_FIT) {
                DAO_ERROR("Not Implemented");
            }
        }
        iter = iter->next; 
    }
    auto to_grow = std::max(size, grow_size);
    if (grow_size && best_iter == nullptr && ((allocated + to_grow + alignment) <= mem_limit)) {    
        void* ptr = device_mallocr->raw_malloc(to_grow + alignment);
        if (ptr != nullptr) {
            allocated_list.push_back(ptr);
            // no available space, need to grow
            best_iter = std::make_shared<MemBlock>();
            best_iter->physical_location_start = (void*)ROUND_UP((unsigned long)ptr, alignment);
            allocated +=  to_grow + alignment;
            // CUDA_CHECK(cudaMalloc(&best_iter->physical_location_start, to_grow));
            best_iter->physical_location_end = (void*)((char*)(best_iter -> physical_location_start) + to_grow);
            best_iter->payload_size = to_grow;

            assert(best_iter->physical_location_end <= start->physical_location_start || 
            best_iter->physical_location_start >= end->physical_location_end);
            if (best_iter->physical_location_end <= start->physical_location_start) {
                // grow from front 
                start->next->prev = best_iter;
                best_iter->next = start->next;
                best_iter->prev = start;
                start->next = best_iter;
                start->physical_location_start = start->physical_location_end = best_iter->physical_location_start;
            }
            else {
                // grow from back.
                end->prev->next = best_iter;
                best_iter->prev = end->prev;
                best_iter->next = end;
                end->prev = best_iter;
                end->physical_location_start = end->physical_location_end = best_iter->physical_location_end;
            }
        }
    }
    timer.stop("allocate");
    if (best_iter == nullptr) return nullptr;
    // allocate and split 
    assert(best_iter->allocated == false && best_iter->payload_size >= size);
    splitAndAllocate(size, best_iter);    
    return best_iter; 
}

bool DoubleLinkedListStorage::splitAndAllocate(size_t size, std::shared_ptr<MemBlock>& block) {
    DAO_ASSERT(size % alignment == 0, "split size not aligned");
    bool split = split_cond(block->payload_size, size);
    if (split) {
        std::shared_ptr<MemBlock> new_block = std::make_shared<MemBlock>();
        new_block->allocated = false;
        new_block->physical_location_start = (void*)((char*)block->physical_location_start + size);
        new_block->physical_location_end = block->physical_location_end;
        new_block->payload_size = block->payload_size - size;

        new_block->prev = block;
        new_block->next = block->next;
        block->next->prev = new_block;
        block->next = new_block;
        block->payload_size = size;
        block->physical_location_end = (void*)((char*)block->physical_location_start + size);
    }
    return split; 
}


bool DoubleLinkedListStorage::split_cond(size_t block_size, size_t required_size) {
    return (block_size - required_size) >= (split_threshold);
}


std::pair<std::shared_ptr<MemBlock>,std::shared_ptr<MemBlock> > 
DoubleLinkedListStorage::evict(size_t size, std::vector<std::shared_ptr<MemBlock>>& blocks) {
    timer.start("evict");
    size = ROUND_UP(size, alignment);
    double best_score = 0;
    std::shared_ptr<MemBlock> front, back, best_front, best_back;
    best_front = best_back = nullptr;
    front = back = start->next;
    size_t total_size = 0;
    std::deque<size_t> block_sizes;
    std::deque<size_t> next_access; 
    // incremental update [front, back) to fit the need
    // invariant [front, back) 1. contiguous 2. not current block;
    while(front != end) { 
        while(back->next != nullptr // not the end 
            && total_size < size // 
            && ((back == front) || (back->prev->physical_location_end == back->physical_location_start))
            && (!back->allocated || (back->record->access_pattern.empty() || back->record->access_pattern.front() != logical_time))) {
            total_size += back->payload_size;
            if (back->allocated) {
                auto record = back->record;
                assert(record!=nullptr);
                block_sizes.push_back(record->tensor_size);
                if (record->access_pattern.empty())
                    next_access.push_back(INT_MAX);  // access at infinity
                else
                    next_access.push_back(record->access_pattern.front());
            }
            back = back->next; 
        }
        while(total_size >= (front->payload_size + size)) {
            assert(front->next != back);
            total_size -= front->payload_size;
            if (front->allocated) {
                if (front->record == nullptr) {
                    DAO_ERROR("front->record == nullptr");
                }
                auto record = front->record;
                assert(record!=nullptr);
                block_sizes.pop_front();
                next_access.pop_front();
            }
            front = front->next;
        }
        if (total_size >= size) {
            if (eviction_strategy == eviction_strategy_t::EVI_FIRST_FIT){
                best_front = front;
                best_back = back; 
                break;
            }
            else if (eviction_strategy == eviction_strategy_t::EVI_BEST_FIT){ 
                double score = eviction_score(size, total_size, block_sizes, next_access);
                if (score > best_score) {
                    best_front = front;
                    best_back = back;
                    best_score = score;
                }
            } else if (eviction_strategy ==  eviction_strategy_t::BELADY || 
                    eviction_strategy ==  eviction_strategy_t::WEIGHTED_BELADY) {
                double score = belady_score(front, back);
                if (score > best_score) {
                    best_front = front;
                    best_back = back;
                    best_score = score;
                }
            }
                
            else if (eviction_strategy ==  eviction_strategy_t::LRU || 
                    eviction_strategy ==  eviction_strategy_t::WEIGHTED_LRU){
                double score = lru_score(front, back);
                if (score > best_score) {
                    best_front = front;
                    best_back = back;
                    best_score = score;
                }
            }
        }
        
        if (back->next == nullptr) break;
        if (back->allocated 
            && !back->record->access_pattern.empty()
            && back->record->access_pattern.front() == logical_time){
            front = back = back->next; 
            total_size = 0; 
            block_sizes.clear();
            next_access.clear();
        }
        else if(back->prev->physical_location_end != back->physical_location_start) {
            front = back;
            total_size = 0;
            block_sizes.clear();
            next_access.clear();
        }
        else {
            if (front->allocated){
                block_sizes.pop_front();
                next_access.pop_front();
            }
            total_size -= front->payload_size;
            front = front->next; 
        }
    }

    if (best_front != nullptr) {
        auto iter = best_front; 
        while(iter != best_back) {
            if (iter->allocated) {
                blocks.push_back(iter);
            }
            iter = iter->next; 
        }
    }
    timer.stop("evict");
    return std::make_pair(best_front, best_back);
}


double DoubleLinkedListStorage::eviction_score(
        size_t required_size,
        size_t total_size, 
        const std::deque<size_t>& sizes,
        const std::deque<size_t>& next_accesses
){
    assert(eviction_strategy == eviction_strategy_t::BELADY || eviction_strategy == eviction_strategy_t::WEIGHTED_BELADY);
    /**
     * \Sigma_{i=1}{n} \frac{next_access_i}{size_i} / sqrt(n) * \frac{required_size}{total_size}
    */
    assert(sizes.size() == next_accesses.size());
    double score = 0;
    if (!sizes.size()) score = INT_MAX;
    else {
        for (int i = 0; i < sizes.size(); i++) {
            score += (double)(next_accesses[i] - logical_time) * sizes[i];
        }
        score /= sqrt(sizes.size());
    }
    score *= (double)required_size / total_size;
    return score; 
}




double DoubleLinkedListStorage::belady_score(
    const std::shared_ptr<MemBlock>&front, 
    const std::shared_ptr<MemBlock>&back
) {
    assert(eviction_strategy == eviction_strategy_t::BELADY || eviction_strategy == eviction_strategy_t::WEIGHTED_BELADY);
    size_t total_size = 0;
    auto iter = front;
    auto score = 0.0;
    int n_blocks = 0;
    while(iter != back) {
        auto size = iter->payload_size;
        total_size += size;
        auto next_access = logical_time + 1024; // a large enough number
        if (iter->allocated){
            assert(iter->record!=nullptr);
            if (!iter->record->access_pattern.empty())
                next_access = iter->record->access_pattern.front();
        }
        iter = iter->next; 
        auto param = (double)(next_access - logical_time);
        if (eviction_strategy == eviction_strategy_t::WEIGHTED_BELADY)
            param *= size;
        score += param;
        n_blocks+=1;
    }
    assert(total_size > 0 && n_blocks > 0);
    if (eviction_strategy == eviction_strategy_t::WEIGHTED_BELADY)
        score /= total_size;
    else score /= n_blocks;
    return score; 
}



double DoubleLinkedListStorage::lru_score(
    const std::shared_ptr<MemBlock>&front, 
    const std::shared_ptr<MemBlock>&back
) {
    assert(eviction_strategy == eviction_strategy_t::LRU || eviction_strategy == eviction_strategy_t::WEIGHTED_LRU);
    size_t total_size = 0;
    auto iter = front;
    auto score = 0.0;
    int n_blocks = 0;
    while(iter != back) {
        auto size = iter->payload_size;
        total_size += size;
        auto last_access = std::max(0, (int)logical_time - 1024);
        if (iter->allocated){
            assert(iter->record!=nullptr);
            last_access = iter->record->access_pattern.last();
        }
        iter = iter->next; 
        auto param = (double)(logical_time - last_access);
        if (eviction_strategy == eviction_strategy_t::WEIGHTED_LRU)
            param *= size;
        score += param;
        n_blocks+=1;
    }
    assert(n_blocks>0);
    if (eviction_strategy == eviction_strategy_t::WEIGHTED_LRU)
        score /= total_size;
    else score /= n_blocks;
    return score; 
}




std::shared_ptr<MemBlock> DoubleLinkedListStorage::merge(
    const std::shared_ptr<MemBlock>& front, 
    const std::shared_ptr<MemBlock>& back){
    assert(front != start);
    std::shared_ptr<MemBlock> new_block = std::make_shared<MemBlock>();
    new_block->physical_location_start = front->physical_location_start;
    new_block->physical_location_end = back->prev->physical_location_end;
    new_block->payload_size = (char*)new_block->physical_location_end - (char*)new_block->physical_location_start;

    new_block->prev = front->prev;
    new_block->next = back;

    auto iter = front;
    size_t total_size = 0; 
    while(iter != back) {
        assert(iter->allocated == false);
        assert(iter->record == nullptr);
        assert(iter->physical_location_end == iter->next->physical_location_start||iter->next == back);
        total_size += iter->payload_size;
        iter->prev.reset();
        iter = iter->next;
        iter->prev->next.reset();
    }

    back->prev = new_block;
    new_block->prev->next = new_block;    

    assert(total_size == new_block->payload_size);
    return new_block;
}

std::shared_ptr<MemBlock> DoubleLinkedListStorage::mergeAndAllocate(
    size_t size, 
    const std::shared_ptr<MemBlock>& front, 
    const std::shared_ptr<MemBlock>& back) {
    timer.start("mergeAndAllocate");
    size = ROUND_UP(size, alignment);
    std::shared_ptr<MemBlock> new_block = merge(front, back);
    splitAndAllocate(size, new_block);
    DAO_ASSERT((size) % (this -> alignment) == 0, "block split size not aligned");
    DAO_ASSERT((unsigned long)(new_block -> physical_location_start) % (this -> alignment) == 0, "new block start location not properly aligned");
    DAO_ASSERT((unsigned long)(new_block -> physical_location_end) % (this -> alignment) == 0, "new block end location not properly aligned");
    timer.stop("mergeAndAllocate");
    return new_block;
}

void DoubleLinkedListStorage::free(const std::shared_ptr<MemBlock>& block) {
    timer.start("free");
    assert(block->allocated == true);
    block->allocated = false;
    block->record = nullptr;
    std::shared_ptr<MemBlock> front, back;
    front = block;
    back = block->next;
    while (front->prev->allocated == false
    && front->prev->physical_location_end == front->physical_location_start) {
        front = front->prev;
    }
    while (back->allocated == false && 
    back->prev->physical_location_end == back->physical_location_start) {
        back = back->next;
    }
    if (!((front->prev->allocated || front->prev->physical_location_end != front->physical_location_start)
        && (back->allocated || back->prev->physical_location_end != back->physical_location_start))){
        display(std::cout, front, back, false);
        printf("%d, %d, %d, %d\n", front->prev->allocated, front->prev->physical_location_end != front->physical_location_start, back->allocated, back->prev->physical_location_end != back->physical_location_start);
        DAO_WARNING("%s front and back are not contiguous ", name.c_str());
    }
    if (front != block || back != block->next) {
        merge(front, back);
    }
    timer.stop("free");
}

void DoubleLinkedListStorage::display(std::ostream& o) const {
    o << "-----------" <<  name << "-----------" << std::endl;
    timer.show(o);
    if (verbose) {
    display(o, start, end);
    auto iter = start;
    assert(iter->prev == nullptr);
    while(iter != end) {
        assert(iter->payload_size == (char*)iter->physical_location_end - (char*)iter->physical_location_start);
        assert(iter->prev == nullptr || iter->prev->physical_location_end <= iter->physical_location_start);
        assert((iter->allocated == false && iter->record == nullptr) || (iter == start || iter == end || (iter->allocated == true && iter->record != nullptr)));
        assert(iter->next->prev == iter);
        iter = iter->next; 
    }
    assert(iter == end);
    assert(iter->next == nullptr);}
    o << "----------"<< name << " END-----------" << std::endl;
}

void DoubleLinkedListStorage::display(std::ostream& o, 
    const std::shared_ptr<MemBlock>& front, 
    const std::shared_ptr<MemBlock>& back,
    bool exclusive) const {
    auto iter = front;
    while(iter != back) {
        iter->display(o);
        o << std::endl;
        iter = iter->next;
    }
    if (!exclusive){ 
        iter->display(o);
        o << std::endl;
    }
}

DoubleLinkedListStorage::~DoubleLinkedListStorage(){
    auto iter = start->next;
    start->next.reset();
    while(iter != end) {
        iter->prev.reset();
        iter = iter->next;
        iter->prev->next.reset();
    }
    assert(iter == end && iter->next == nullptr);
    iter->prev.reset();
    start.reset();
    end.reset();
    iter.reset();

    for(auto ptr: allocated_list) {
        device_mallocr->raw_free(ptr);
    }
}

void MemBlock::display(std::ostream& o) const {
    o << "[A:" << allocated << ", ";
    o << "S:" << payload_size << ", ";
    if (record) {
        o << "R:";
        record->display(o, false);
        assert(record->block_ptr.get() == this);
    }
    if (record && allocated && !record->access_pattern.empty()) {
        o << "T:" << record->access_pattern.front() << ", "; 
    }
    o << "B:" << (physical_location_start) << ", ";
    o << "E:" << (physical_location_end) << ", ";
    o << "P:" << ((size_t)prev.get() & 0xfff) << ", ";
    o << "N:" << ((size_t)next.get() & 0xfff) << ", ";
    o << "M:" << ((size_t)this & 0xfff) << "]";
}

bool DoubleLinkedListStorage::check_MemStorage() {
    if (!debug_mode) return true;
    //check if the double linked list is valid
    //std::cout << "storage check" << std::endl;
    if (this -> start -> physical_location_start != this -> start -> physical_location_end) {
        std::cout << "header block invalid" << std::endl;
        return false;
    }
    if (this -> end -> physical_location_start != this -> end -> physical_location_end) {
        std::cout << "tail block invalid" << std::endl;
        return false;
    }
    std::shared_ptr<MemBlock> tracer = this -> start -> next;
    if (tracer -> prev != this -> start) {
        std::cout << "starting block not well connected with the first block" << std::endl;
        return false;
    }
    size_t discontinous_counter = 0;
    while (tracer != this -> end) {
        size_t block_size = ((size_t) tracer -> physical_location_end) - ((size_t) tracer -> physical_location_start);
        size_t payload_size = tracer -> payload_size;
        if (block_size % (this -> alignment) != 0) {
            std::cout << "start of the block " << tracer -> physical_location_start << std::endl;
            std::cout << "end of the block " << tracer -> physical_location_end << std::endl;
            std::cout << "block size not aligned properly: " << block_size << std::endl;
            return false;
        } 
        if (payload_size > block_size) {
            std::cout << "payload size larger than block size" << std::endl;
            return false;
        }
        if (tracer -> next != NULL) {
            if (tracer -> next -> prev != tracer) {
                std::cout << "next block" << tracer -> next << std::endl;
                std::cout << "current block" << tracer << std::endl;
                std::cout << "double linked list broken" << std::endl;
                return false;
            }
            /*
            if (!(tracer -> allocated) && !(tracer -> next-> allocated))  {
                if ((unsigned long) (tracer -> physical_location_end) ==  (unsigned long) (tracer -> next -> physical_location_start)) {
                    std::cout << "consequtive free blocks without merging" << std::endl;
                    //this -> display(std::std::cout, this -> start, this -> end, false);
                    return false;
                }
            }
            */
            if ((unsigned long) (tracer -> physical_location_end) !=  (unsigned long) (tracer -> next -> physical_location_start)) {
                discontinous_counter += 1;
            }
        }
        tracer = tracer -> next;
    }
    if (discontinous_counter != allocated_list.size() - 1) {
        std::cout << "broken block list: more discontinous block address than allocated list" << std::endl;
        return false;
    }
    //std::cout << "start checking each list" << std::endl;
    return true;
    //check if block size if valid
}

size_t DoubleLinkedListStorage::get_max_usage() const {
    if (grow_size && DAO::offload_profiling) {
        auto it = start->next;
        size_t max_usage = 0;        
        while(it != nullptr) {
            auto last_allocated = it;
            void* s = it->physical_location_start;
            while (
                it->next != nullptr && 
                it->physical_location_end == it->next->physical_location_start) {
                if (it->allocated) {
                    last_allocated = it;
                }
                it = it->next;
            }
            max_usage += (char*)last_allocated->physical_location_end - (char*)s;
            it = it->next;
        }
        return max_usage; 
    }
    DAO_COND_WARNING(!grow_size && !DAO::offload_profiling, "inaccurate max usage");
    auto it = end->prev;
    while(it!=start && it->allocated == false) {
        it = it->prev;
    }
    return (char*)it->physical_location_end - (char*)start->physical_location_start;
}

} // namespace DAO