#ifndef MEM_MANAGER_H
#define MEM_MANAGER_H

#include <unordered_map>
#include <list>
#include <vector>
#include <memory>
#include <utility>
#include <deque>
#include <iostream>

#include "globals.h"

using namespace std;

struct TensorRecord;


struct MemBlock {
    bool allocated = false;
    size_t payload_size = 0;
    void* physical_location_start = nullptr;
    void* physical_location_end = nullptr;
    TensorRecord* record = nullptr;
    std::shared_ptr<MemBlock> prev = nullptr;
    std::shared_ptr<MemBlock> next = nullptr;
    void display(std::ostream& o) const;
};

struct DeviceRawMallocr {
    virtual void* raw_malloc(size_t size) = 0;
    virtual void raw_free(void*) = 0;
    virtual size_t mem_limit() = 0;
};

struct VirtualMallocr: public DeviceRawMallocr {
    void* raw_malloc(size_t size) override;
    void raw_free(void*) override;
    size_t mem_limit() override;
    size_t allocated = 1024;
};

class MemStorage{
protected: 
    std::unique_ptr<DeviceRawMallocr> device_mallocr;
    std::string name; 
public:
    enum device_type_t {
        CPU,
        GPU,
        VIRTUAL
    }device_type;
    MemStorage(device_type_t device_type, const std::string& name = "MemStorage");
    virtual std::shared_ptr<MemBlock> allocate(size_t size) = 0;
    virtual void free(const std::shared_ptr<MemBlock>& block) = 0;
    /**
     * @brief Evict a block of memory from the storage
     * @param size The size of the block to be evicted
     * @return A pair of pointers to the start and end of the evicted block
     * @return blocks A list of blocks that are evicted
    */
    virtual std::pair<std::shared_ptr<MemBlock>,std::shared_ptr<MemBlock> > 
    evict(size_t size, std::vector<std::shared_ptr<MemBlock>>& blocks) = 0;
    /**
     * @brief Merge a list of blocks into one block
     * @param start The start of the list of blocks, inclusive
     * @param end The end of the list of blocks, exclusive
    */
    virtual std::shared_ptr<MemBlock> merge(const std::shared_ptr<MemBlock>& front, const std::shared_ptr<MemBlock>& back) = 0;
    /**
     * @brief Merge a list of blocks into one block and allocate a new block
     * @param size The size of the new block
     * @param start The start of the list of blocks, inclusive
     * @param end The end of the list of blocks, exclusive
    */
    virtual std::shared_ptr<MemBlock> mergeAndAllocate(size_t size, const std::shared_ptr<MemBlock>& front, const std::shared_ptr<MemBlock>& back) = 0;
    virtual void display(std::ostream& o) const = 0; 
    virtual ~MemStorage() = default;
};

class DoubleLinkedListStorage : public MemStorage {
public: 
    enum allocation_strategy_t {
        FIRST_FIT,
        BEST_FIT,
        WORST_FIT
    };

    enum eviction_strategy_t {
        EVI_FIRST_FIT, 
        EVI_BEST_FIT,
        BELADY,
        WEIGHTED_BELADY,
        LRU,
        WEIGHTED_LRU 
    };
    
    DoubleLinkedListStorage(
        MemStorage::device_type_t device_type,
        size_t device_mem_size, 
        logical_time_t& logical_time,
        size_t mem_limit = size_t(-1),
        size_t grow_size = (1 << 20),
        allocation_strategy_t allocation_strategy = allocation_strategy_t::FIRST_FIT,
        eviction_strategy_t eviction_strategy = eviction_strategy_t::EVI_FIRST_FIT,
        size_t split_threshold = (1 << 10),
        size_t alignment = 128u);
    ~DoubleLinkedListStorage();

    inline double eviction_score(
        size_t required_size,
        size_t total_size, 
        const std::deque<size_t>& sizes,
        const std::deque<size_t>& next_accesses
    );
    inline double lru_score(
        const std::shared_ptr<MemBlock>& front, 
        const std::shared_ptr<MemBlock>& back
    );
    inline double belady_score(
        const std::shared_ptr<MemBlock>& front, 
        const std::shared_ptr<MemBlock>& back
    );
    inline bool split_cond(size_t block_size, size_t required_size);
    std::shared_ptr<MemBlock> allocate(size_t size) override;
    void free(const std::shared_ptr<MemBlock>& block) override;
    std::pair<std::shared_ptr<MemBlock>,std::shared_ptr<MemBlock> > 
    evict(size_t size, std::vector<std::shared_ptr<MemBlock>>& blocks) override;
    std::shared_ptr<MemBlock> merge(const std::shared_ptr<MemBlock>& front, const std::shared_ptr<MemBlock>& back) override;
    std::shared_ptr<MemBlock> mergeAndAllocate(size_t size, const std::shared_ptr<MemBlock>& front, const std::shared_ptr<MemBlock>& back) override;
    void display(std::ostream& o) const override;

private: 
    inline void display(std::ostream& o, const std::shared_ptr<MemBlock>& front, const std::shared_ptr<MemBlock>& back, bool exclusive = false) const;
    bool splitAndAllocate(size_t size, std::shared_ptr<MemBlock>& block);
    allocation_strategy_t allocation_strategy;
    eviction_strategy_t eviction_strategy;
    size_t alignment;
    size_t split_threshold;  
    size_t allocated;
    size_t grow_size; 
    size_t mem_limit;
    std::shared_ptr<MemBlock> start;
    std::shared_ptr<MemBlock> end; 
    std::vector<void*> allocated_list;
    const logical_time_t& logical_time; 
};

#endif 